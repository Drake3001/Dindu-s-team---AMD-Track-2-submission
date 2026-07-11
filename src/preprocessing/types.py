from __future__ import annotations

from dataclasses import dataclass

import numpy as np


class PreprocessingError(Exception):
    """Raised for any unrecoverable failure in the preprocessing stage."""


@dataclass
class Frame:
    index: int            # position within the sampled sequence
    timestamp: float      # seconds into the source video
    image: np.ndarray     # BGR ndarray (opencv convention)
    score: float = 0.0    # action intensity score from the detector


@dataclass
class VideoMetadata:
    path: str
    duration_sec: float
    fps: float
    frame_count: int
    width: int
    height: int
