from __future__ import annotations

from pathlib import Path
import cv2
import numpy as np
import random

from pipeline_2.important_frames.api import DetectorState
from preprocessing.vlm_output.grid import frames_to_grid_b64
from preprocessing.types import Frame, VideoMetadata, PreprocessingError

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

    frame_idx = 0
    try:
        while True:
            ok, img = cap.read()
            if not ok or img is None:
                break
                
            is_important = state.process_frame(img)
            
            ts = frame_idx / native_fps
            if is_important:
                important_frames_list.append(
                    Frame(index=frame_idx, timestamp=ts, image=_resize_max_dim(img, max_dim))
                )
            else:
                # To save memory during the loop, we randomly drop a vast majority 
                # of unimportant frames, because we only need ~4 of them for context anyway.
                if random.random() < 0.1:
                    unimportant_frames_list.append(
                        Frame(index=frame_idx, timestamp=ts, image=_resize_max_dim(img, max_dim))
                    )
                    
            frame_idx += 1
    finally:
        cap.release()

    if frame_idx == 0:
        raise PreprocessingError(f"No frames could be extracted from {video_path}")

    # 3. Budget Application
    actual_context = min(context_frames, len(unimportant_frames_list))
    
    chosen_context = []
    if actual_context > 0:
        step = max(1.0, len(unimportant_frames_list) / actual_context)
        for i in range(actual_context):
            chosen_context.append(unimportant_frames_list[int(i * step)])

    important_budget = max_frames - actual_context
    chosen_important = []
    if len(important_frames_list) > 0:
        if len(important_frames_list) <= important_budget:
            chosen_important = list(important_frames_list)
        else:
            step = len(important_frames_list) / important_budget
            for i in range(important_budget):
                chosen_important.append(important_frames_list[int(i * step)])

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
