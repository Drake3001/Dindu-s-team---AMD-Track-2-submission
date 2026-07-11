from __future__ import annotations

from pathlib import Path
import cv2
import numpy as np
import random

from src.pipeline_2.important_frames.detector import DetectorState
from src.preprocessing.vlm_output.grid import frames_to_grid_b64
from src.preprocessing.types import Frame, VideoMetadata, PreprocessingError

def _effective_fps(fps: float, duration: float, frame_count: int) -> float:
    if fps > 0:
        return fps
    if duration > 0 and frame_count > 0:
        return frame_count / duration
    return 30.0

def _resize_max_dim(image: np.ndarray, max_dim: int) -> np.ndarray:
    h, w = image.shape[:2]
    longest = max(h, w)
    if longest <= max_dim:
        return image
    scale = max_dim / float(longest)
    new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

def extract_smart_grids(
    video_path: Path | str,
    max_frames: int = 32,
    context_frames: int = 4,
    max_dim: int = 512,
    grid_cols: int = 4,
    grid_rows: int = 4,
) -> dict:
    """
    Extracts a prioritized list of important frames and background context frames
    in a SINGLE pass, downscales them in-memory, and formats them into grids.
    """
    video_path = Path(video_path)
    if not video_path.is_file():
        raise PreprocessingError(f"Video file not found: {video_path}")

    # 1. Read metadata
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise PreprocessingError(f"Could not open video: {video_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    duration = frame_count / fps if fps > 0 else 0.0

    metadata = VideoMetadata(
        path=str(video_path),
        duration_sec=duration,
        fps=fps,
        frame_count=frame_count,
        width=width,
        height=height,
    )
    
    native_fps = _effective_fps(fps, duration, frame_count)

    # 2. Single Pass Streaming
    state = DetectorState()
    important_frames_list: list[Frame] = []
    unimportant_frames_list: list[Frame] = []
    
    # Padding logic for streaming:
    # We want +/- 15 frames of padding around every trigger.
    padding = 15
    past_buffer: list[Frame] = [] # Holds the last 'padding' frames
    save_countdown = 0            # How many future frames to save unconditionally
    seen_important_indices = set() # To prevent duplicates

    frame_idx = 0
    try:
        while True:
            ok, img = cap.read()
            if not ok or img is None:
                break
                
            is_important, diff = state.process_frame(img)
            ts = frame_idx / native_fps
            current_frame = Frame(index=frame_idx, timestamp=ts, image=_resize_max_dim(img, max_dim), score=diff)
            
            if is_important:
                # If we just triggered, save all frames in the past buffer
                for pf in past_buffer:
                    if pf.index not in seen_important_indices:
                        important_frames_list.append(pf)
                        seen_important_indices.add(pf.index)
                
                # Save the current frame
                if current_frame.index not in seen_important_indices:
                    important_frames_list.append(current_frame)
                    seen_important_indices.add(current_frame.index)
                
                # Tell the loop to save the NEXT `padding` frames unconditionally
                save_countdown = padding
                
            elif save_countdown > 0:
                # We didn't trigger, but we are inside the +padding window of a recent trigger
                if current_frame.index not in seen_important_indices:
                    important_frames_list.append(current_frame)
                    seen_important_indices.add(current_frame.index)
                save_countdown -= 1
                
            else:
                # Completely unimportant frame
                if random.random() < 0.1:
                    unimportant_frames_list.append(current_frame)
                    
            # Maintain the sliding window for past padding
            past_buffer.append(current_frame)
            if len(past_buffer) > padding:
                past_buffer.pop(0)
                    
            frame_idx += 1
    finally:
        cap.release()

    if frame_idx == 0:
        raise PreprocessingError(f"No frames could be extracted from {video_path}")

    # 3. Budget Application
    actual_context = min(context_frames, len(unimportant_frames_list))
    
    chosen_context = []
    if actual_context > 0:
        chosen_context = random.sample(unimportant_frames_list, actual_context)

    important_budget = max_frames - actual_context
    chosen_important = []
    if len(important_frames_list) > 0:
        if len(important_frames_list) <= important_budget:
            chosen_important = list(important_frames_list)
        else:
            # Flattened Normal Distribution Frame Sampling
            
            # Find Top Peaks (up to 3 distinct peaks spaced by at least 1.5s to cover prolonged events)
            sorted_by_score = sorted(important_frames_list, key=lambda f: f.score, reverse=True)
            peaks = []
            for f in sorted_by_score:
                if all(abs(f.timestamp - p.timestamp) > 1.5 for p in peaks):
                    peaks.append(f)
                if len(peaks) >= 3:
                    break
            
            # Calculate standard deviation (wider spread to cover prolonged events)
            sigma = duration / 4.0
            if sigma <= 0:
                sigma = 1.0 # fallback
                
            # Compute Gaussian weights for all frames
            weights = np.zeros(len(important_frames_list), dtype=np.float64)
            for i, f in enumerate(important_frames_list):
                pdf_sum = sum(np.exp(-0.5 * ((f.timestamp - p.timestamp) / sigma) ** 2) for p in peaks)
                weights[i] = pdf_sum
                
            # Normalize gaussian weights
            weight_sum = np.sum(weights)
            if weight_sum > 0:
                weights /= weight_sum
            else:
                weights = np.ones(len(important_frames_list)) / len(important_frames_list)
                
            # Flatten the distribution: Mix 50% Gaussian with 50% Uniform
            uniform_weight = 1.0 / len(important_frames_list)
            weights = 0.5 * weights + 0.5 * uniform_weight
            weights /= np.sum(weights) # re-normalize exactly to 1.0
                
            # Sample exactly important_budget frames WITHOUT replacement
            sampled_indices = np.random.choice(
                len(important_frames_list), 
                size=important_budget, 
                replace=False, 
                p=weights
            )
            
            for idx in sampled_indices:
                chosen_important.append(important_frames_list[idx])

    # Combine and sort chronologically
    final_frames = sorted(chosen_context + chosen_important, key=lambda f: f.index)

    if not final_frames:
        raise PreprocessingError("No frames selected for grid.")

    # 4. Build grids
    grids_b64 = frames_to_grid_b64(final_frames, cols=grid_cols, rows=grid_rows)
    
    return {
        "metadata": metadata,
        "grids_b64": grids_b64,
        "frame_timestamps": [f.timestamp for f in final_frames],
        "sampled_count": len(final_frames)
    }
