"""Data types for the important_frames module."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


class ImportantFramesError(Exception):
    """Raised when the important-frame detection pipeline fails."""


@dataclass(frozen=True)
class DetectionResult:
    """Output of the important-frame detection pipeline.

    Attributes:
        important_frames: 1-D int array of 0-based frame indices deemed important.
        total_frames: Total number of frames in the source video.
        fps: Original frames-per-second of the source video.
    """

    important_frames: np.ndarray
    total_frames: int
    fps: float
