from __future__ import annotations

from pathlib import Path
import random

import cv2
import numpy as np

from pipeline_2.grid_extractor.ffmpeg_reader import iter_frames, probe_metadata
from pipeline_2.important_frames.detector import DetectorState
from preprocessing.types import Frame, PreprocessingError, VideoMetadata
from preprocessing.vlm_output.grid import frames_to_grid_b64


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
    max_frames: int = 48,
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

    metadata = probe_metadata(video_path)
    native_fps = _effective_fps(metadata.fps, metadata.duration_sec, metadata.frame_count)

    state = DetectorState()
    important_frames_list: list[Frame] = []
    unimportant_frames_list: list[Frame] = []

    padding = 15
    past_buffer: list[Frame] = []
    save_countdown = 0
    seen_important_indices: set[int] = set()

    frame_idx = 0
    for img in iter_frames(video_path, max_dim=max_dim, metadata=metadata):
        is_important, diff = state.process_frame(img)
        ts = frame_idx / native_fps
        current_frame = Frame(
            index=frame_idx,
            timestamp=ts,
            image=_resize_max_dim(img, max_dim),
            score=diff,
        )

        if is_important:
            for pf in past_buffer:
                if pf.index not in seen_important_indices:
                    important_frames_list.append(pf)
                    seen_important_indices.add(pf.index)

            if current_frame.index not in seen_important_indices:
                important_frames_list.append(current_frame)
                seen_important_indices.add(current_frame.index)

            save_countdown = padding

        elif save_countdown > 0:
            if current_frame.index not in seen_important_indices:
                important_frames_list.append(current_frame)
                seen_important_indices.add(current_frame.index)
            save_countdown -= 1

        else:
            if random.random() < 0.1:
                unimportant_frames_list.append(current_frame)

        past_buffer.append(current_frame)
        if len(past_buffer) > padding:
            past_buffer.pop(0)

        frame_idx += 1

    if frame_idx == 0:
        raise PreprocessingError(f"No frames could be extracted from {video_path}")

    duration = metadata.duration_sec
    if duration <= 0 and frame_idx > 0 and native_fps > 0:
        duration = frame_idx / native_fps

    metadata = VideoMetadata(
        path=metadata.path,
        duration_sec=duration,
        fps=metadata.fps,
        frame_count=frame_idx,
        width=metadata.width,
        height=metadata.height,
    )

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
            sorted_by_score = sorted(important_frames_list, key=lambda f: f.score, reverse=True)
            peaks = []
            for f in sorted_by_score:
                if all(abs(f.timestamp - p.timestamp) > 1.5 for p in peaks):
                    peaks.append(f)
                if len(peaks) >= 3:
                    break

            sigma = duration / 4.0
            if sigma <= 0:
                sigma = 1.0

            weights = np.zeros(len(important_frames_list), dtype=np.float64)
            for i, f in enumerate(important_frames_list):
                pdf_sum = sum(
                    np.exp(-0.5 * ((f.timestamp - p.timestamp) / sigma) ** 2) for p in peaks
                )
                weights[i] = pdf_sum

            weight_sum = np.sum(weights)
            if weight_sum > 0:
                weights /= weight_sum
            else:
                weights = np.ones(len(important_frames_list)) / len(important_frames_list)

            uniform_weight = 1.0 / len(important_frames_list)
            weights = 0.5 * weights + 0.5 * uniform_weight
            weights /= np.sum(weights)

            sampled_indices = np.random.choice(
                len(important_frames_list),
                size=important_budget,
                replace=False,
                p=weights,
            )

            for idx in sampled_indices:
                chosen_important.append(important_frames_list[idx])

    final_frames = sorted(chosen_context + chosen_important, key=lambda f: f.index)

    if not final_frames:
        raise PreprocessingError("No frames selected for grid.")

    grids_b64 = frames_to_grid_b64(final_frames, cols=grid_cols, rows=grid_rows)

    return {
        "metadata": metadata,
        "grids_b64": grids_b64,
        "frame_timestamps": [f.timestamp for f in final_frames],
        "sampled_count": len(final_frames),
    }
