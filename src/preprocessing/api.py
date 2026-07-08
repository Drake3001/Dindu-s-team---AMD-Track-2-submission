"""
Public interface for the preprocessing module.

Downstream stages (VLM description, LLM caption generation) should import
only from here - not from `preprocessing.py` directly - so the extraction
strategy can be changed without touching callers.

This module does NOT download anything. It expects the video for a task
to already exist under /video (handled by another component) and writes
its output frames under /preprocessed_input.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Union

import cv2

from .preprocessing import (
    DEFAULT_MAX_DIM,
    DEFAULT_MAX_FRAMES,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_VIDEO_DIR,
    PreprocessingError,
    PreprocessResult,
    preprocess_video,
)

__all__ = ["PreprocessingError", "PreprocessedVideo", "preprocess"]


@dataclass
class PreprocessedVideo:
    task_id: str
    duration_sec: float
    width: int
    height: int
    num_frames: int
    frames_b64: List[str]        # base64 JPEGs, ready to hand to a VLM
    frame_timestamps: List[float]
    saved_paths: List[str]       # empty if save_to_disk=False


def _frame_to_b64(image, quality: int = 85) -> str:
    ok, buf = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise PreprocessingError("Failed to encode frame to JPEG")
    return base64.b64encode(buf.tobytes()).decode("utf-8")


def preprocess(
    task_id: str,
    video_url: str,
    max_frames: int = DEFAULT_MAX_FRAMES,
    max_dim: int = DEFAULT_MAX_DIM,
    strategy: str = "adaptive",
    video_dir: Union[str, Path] = DEFAULT_VIDEO_DIR,
    output_dir: Union[str, Path] = DEFAULT_OUTPUT_DIR,
    save_to_disk: bool = True,
) -> PreprocessedVideo:
    """Locate the already-downloaded clip for this task in `video_dir`,
    sample + resize it, and return frames ready for a VLM.

    `video_url` is the task's original URL from tasks.json - it is used
    only to derive the expected local filename (basename of the URL, or
    `{task_id}.<ext>` as a fallback), never fetched over the network.

    Raises PreprocessingError if the file can't be found or read, or zero
    frames could be extracted, so the caller can record that task as
    failed instead of silently emitting empty captions.
    """
    result: PreprocessResult = preprocess_video(
        task_id=task_id,
        video_ref=video_url,
        max_frames=max_frames,
        max_dim=max_dim,
        strategy=strategy,
        video_dir=Path(video_dir),
        output_dir=Path(output_dir),
        save_to_disk=save_to_disk,
    )

    frames_b64 = [_frame_to_b64(f.image) for f in result.frames]

    return PreprocessedVideo(
        task_id=task_id,
        duration_sec=result.metadata.duration_sec,
        width=result.metadata.width,
        height=result.metadata.height,
        num_frames=len(result.frames),
        frames_b64=frames_b64,
        frame_timestamps=[f.timestamp for f in result.frames],
        saved_paths=result.saved_paths,
    )
