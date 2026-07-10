"""Public API for the important_frames module.

Usage::

    from important_frames.api import get_important_frames

    frames = get_important_frames("videos/v1.mp4")
    # → numpy array of 0-based frame indices
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .detector import (
    DEFAULT_ALPHA,
    DEFAULT_MAX_DIM,
    DEFAULT_PADDING,
    DEFAULT_THRESHOLD,
    DEFAULT_DIFF_ALPHA,
    DEFAULT_DIFF_MULTIPLIER,
    detect_important_frames,
)


def get_important_frames(
    video_path: str | Path,
    *,
    alpha: float = DEFAULT_ALPHA,
    threshold: float = DEFAULT_THRESHOLD,
    max_dim: int = DEFAULT_MAX_DIM,
    padding: int = DEFAULT_PADDING,
    diff_alpha: float = DEFAULT_DIFF_ALPHA,
    diff_multiplier: float = DEFAULT_DIFF_MULTIPLIER,
) -> np.ndarray:
    """Return a 1-D numpy array of important frame indices for *video_path*.

    This is the single public entry-point for the module.  See
    :func:`~important_frames.detector.detect_important_frames` for full
    parameter documentation.

    Parameters
    ----------
    video_path:
        Path to a video file readable by OpenCV.
    alpha:
        EMA smoothing factor (0 < α ≤ 1).
    threshold:
        Mean absolute difference threshold (0–255 scale).
    max_dim:
        Longest-edge cap for greyscale downscaling.
    padding:
        Extra context frames on each side of every threshold crossing.
    diff_alpha:
        Alpha for the frame-to-frame difference EMA.
    diff_multiplier:
        Multiplier for the thresholding logic.

    Returns
    -------
    numpy.ndarray
        1-D array (``dtype=np.intp``) of 0-based frame indices.
    """
    result = detect_important_frames(
        video_path,
        alpha=alpha,
        threshold=threshold,
        max_dim=max_dim,
        padding=padding,
        diff_alpha=diff_alpha,
        diff_multiplier=diff_multiplier,
    )
    return result.important_frames
