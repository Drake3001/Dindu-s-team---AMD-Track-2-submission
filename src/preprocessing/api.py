"""
Public interface for the preprocessing module.

Downstream stages (VLM description, LLM caption generation) should import
only from here - not from `preprocessing.py` directly - so the extraction
strategy can be changed without touching callers.

This module does NOT download anything. Callers pass a Path to the local
video file. Processed output is held in memory as base64 grid images.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Union

from .preprocessing import (
    DEFAULT_GRID_COLS,
    DEFAULT_GRID_ROWS,
    DEFAULT_MAX_DIM,
    DEFAULT_MAX_FRAMES,
    DEFAULT_PRUNE_THRESHOLD,
    DEFAULT_TARGET_FPS,
    PreprocessResult,
    preprocess_video,
)
from .types import PreprocessingError
from .vlm_output import GridImage

__all__ = ["PreprocessingError", "PreprocessedVideo", "preprocess"]


@dataclass
class PreprocessedVideo:
    task_id: str
    duration_sec: float
    width: int
    height: int
    sampled_count: int
    post_pruned_count: int
    num_grids: int
    grids: List[GridImage]
    frame_timestamps: List[float]

    @property
    def grids_b64(self) -> List[str]:
        return [grid.b64 for grid in self.grids]


def preprocess(
    task_id: str,
    video_path: Union[str, Path],
    *,
    fps: float = DEFAULT_TARGET_FPS,
    max_dim: int = DEFAULT_MAX_DIM,
    prune_threshold: float = DEFAULT_PRUNE_THRESHOLD,
    grid_cols: int = DEFAULT_GRID_COLS,
    grid_rows: int = DEFAULT_GRID_ROWS,
    max_frames: Optional[int] = DEFAULT_MAX_FRAMES,
) -> PreprocessedVideo:
    """Sample, prune, and grid-encode a local video for VLM consumption.

    Raises PreprocessingError if the file can't be read or zero frames could
    be extracted, so the caller can record that task as failed instead of
    silently emitting empty captions.
    """
    result: PreprocessResult = preprocess_video(
        task_id=task_id,
        video_path=Path(video_path),
        fps=fps,
        max_dim=max_dim,
        prune_threshold=prune_threshold,
        grid_cols=grid_cols,
        grid_rows=grid_rows,
        max_frames=max_frames,
    )

    return PreprocessedVideo(
        task_id=task_id,
        duration_sec=result.metadata.duration_sec,
        width=result.metadata.width,
        height=result.metadata.height,
        sampled_count=result.sampled_count,
        post_pruned_count=result.post_pruned_count,
        num_grids=len(result.grids),
        grids=result.grids,
        frame_timestamps=result.frame_timestamps,
    )
