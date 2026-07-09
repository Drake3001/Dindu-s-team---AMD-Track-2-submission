"""
Video preprocessing pipeline.

Turns an already-downloaded video file into a small set of resized frames,
optionally prunes near-duplicates, and builds in-memory base64 grid images
for downstream VLM consumption.

Downloading is NOT this module's responsibility - callers pass a Path to
the local file. Other modules should not import this file directly - go
through `api.py`.
"""

from __future__ import annotations

import gc
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
import structlog

from .pruning_strat import AbsDiffPruner, FramePruner
from .vlm_output import frames_to_grid_b64

log = structlog.get_logger(__name__)

DEFAULT_MAX_FRAMES = 240
DEFAULT_MAX_DIM = 512
DEFAULT_TARGET_FPS = 1.0
DEFAULT_PRUNE_THRESHOLD = 5.0
DEFAULT_GRID_COLS = 4
DEFAULT_GRID_ROWS = 4

from .types import Frame, PreprocessingError, VideoMetadata


@dataclass
class PreprocessResult:
    task_id: str
    source: str
    metadata: VideoMetadata
    sampled_count: int
    post_pruned_count: int
    grids_b64: List[str]
    frame_timestamps: List[float]


# --------------------------------------------------------------------------
# Metadata
# --------------------------------------------------------------------------

def read_metadata(video_path: Path) -> VideoMetadata:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise PreprocessingError(f"Could not open video: {video_path}")
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        duration = frame_count / fps if fps > 0 else 0.0
        return VideoMetadata(
            path=str(video_path),
            duration_sec=duration,
            fps=fps,
            frame_count=frame_count,
            width=width,
            height=height,
        )
    finally:
        cap.release()


# --------------------------------------------------------------------------
# Resizing
# --------------------------------------------------------------------------

def _resize_max_dim(image: np.ndarray, max_dim: int) -> np.ndarray:
    h, w = image.shape[:2]
    longest = max(h, w)
    if longest <= max_dim:
        return image
    scale = max_dim / float(longest)
    new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)


def _effective_fps(metadata: VideoMetadata) -> float:
    if metadata.fps > 0:
        return metadata.fps
    if metadata.duration_sec > 0 and metadata.frame_count > 0:
        return metadata.frame_count / metadata.duration_sec
    return 0.0


# --------------------------------------------------------------------------
# Sample + downscale (single OpenCV pass)
# --------------------------------------------------------------------------

def sample_and_downscale(
    video_path: Path,
    fps: float = DEFAULT_TARGET_FPS,
    max_dim: int = DEFAULT_MAX_DIM,
    max_frames: Optional[int] = DEFAULT_MAX_FRAMES,
) -> tuple[VideoMetadata, list[Frame]]:
    """Sample frames at target fps and downscale in a single cv2 decode pass."""
    metadata = read_metadata(video_path)
    native_fps = _effective_fps(metadata)
    if native_fps <= 0:
        raise PreprocessingError(f"Could not determine fps for {video_path}")

    if fps <= 0:
        raise PreprocessingError("target fps must be positive")

    step = max(1, round(native_fps / fps))

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise PreprocessingError(f"Could not open video for sampling: {video_path}")

    frames: list[Frame] = []
    try:
        if step > 15 and metadata.frame_count > 0:
            target_indices = []
            curr_idx = 0
            while curr_idx < metadata.frame_count:
                target_indices.append(curr_idx)
                curr_idx += step
                if max_frames is not None and len(target_indices) >= max_frames:
                    break

            for target_idx in target_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, target_idx)
                ok, img = cap.read()
                if not ok or img is None:
                    break
                ts = target_idx / native_fps
                frames.append(
                    Frame(
                        index=len(frames),
                        timestamp=ts,
                        image=_resize_max_dim(img, max_dim),
                    )
                )
        else:
            idx = 0
            while True:
                ok, img = cap.read()
                if not ok or img is None:
                    break
                if idx % step == 0:
                    ts = idx / native_fps
                    frames.append(
                        Frame(
                            index=len(frames),
                            timestamp=ts,
                            image=_resize_max_dim(img, max_dim),
                        )
                    )
                    if max_frames is not None and len(frames) >= max_frames:
                        break
                idx += 1
    finally:
        cap.release()

    if not frames:
        raise PreprocessingError(f"No frames could be extracted from {video_path}")

    return metadata, frames


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------

def preprocess_video(
    task_id: str,
    video_path: Path,
    *,
    fps: float = DEFAULT_TARGET_FPS,
    max_dim: int = DEFAULT_MAX_DIM,
    prune_threshold: float = DEFAULT_PRUNE_THRESHOLD,
    grid_cols: int = DEFAULT_GRID_COLS,
    grid_rows: int = DEFAULT_GRID_ROWS,
    max_frames: Optional[int] = DEFAULT_MAX_FRAMES,
    pruner: Optional[FramePruner] = None,
) -> PreprocessResult:
    """Full pipeline: sample+downscale -> prune -> in-memory base64 grids."""
    video_path = Path(video_path)
    if not video_path.is_file():
        raise PreprocessingError(f"Video file not found: {video_path}")

    log.info(
        "preprocessing_started",
        task_id=task_id,
        path=str(video_path),
        fps=fps,
        max_dim=max_dim,
        prune_threshold=prune_threshold,
        grid_cols=grid_cols,
        grid_rows=grid_rows,
    )

    metadata, sampled = sample_and_downscale(
        video_path,
        fps=fps,
        max_dim=max_dim,
        max_frames=max_frames,
    )
    sampled_count = len(sampled)

    pruner = pruner or AbsDiffPruner(threshold=prune_threshold)
    post_pruned = pruner.prune(sampled)
    sampled.clear()
    post_pruned_count = len(post_pruned)

    grids_b64 = frames_to_grid_b64(post_pruned, cols=grid_cols, rows=grid_rows)
    timestamps = [f.timestamp for f in post_pruned]
    post_pruned.clear()
    gc.collect()

    log.info(
        "preprocessing_complete",
        task_id=task_id,
        duration_sec=metadata.duration_sec,
        sampled_count=sampled_count,
        post_pruned_count=post_pruned_count,
        num_grids=len(grids_b64),
    )

    return PreprocessResult(
        task_id=task_id,
        source=str(video_path),
        metadata=metadata,
        sampled_count=sampled_count,
        post_pruned_count=post_pruned_count,
        grids_b64=grids_b64,
        frame_timestamps=timestamps,
    )


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("usage: python -m preprocessing.preprocessing <task_id> <video_path>")
        sys.exit(1)
    tid, path = sys.argv[1], sys.argv[2]
    try:
        res = preprocess_video(tid, Path(path))
        print(
            f"{tid}: {res.post_pruned_count} frames -> {len(res.grids_b64)} grids, "
            f"duration={res.metadata.duration_sec:.1f}s, "
            f"size={res.metadata.width}x{res.metadata.height}"
        )
    except PreprocessingError as e:
        print(f"{tid}: FAILED - {e}")
        sys.exit(1)
