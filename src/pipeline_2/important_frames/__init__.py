"""Important-frame detection via EMA-based change detection.

Public API::

    from important_frames import get_important_frames, detect_important_frames

    # Quick – returns just the numpy array of frame indices:
    frames = get_important_frames("videos/v1.mp4")

    # Detailed – returns a DetectionResult with metadata:
    result = detect_important_frames("videos/v1.mp4", alpha=0.05, threshold=8.0)
    result.important_frames   # np.ndarray
    result.total_frames       # int
    result.fps                # float
"""

from .api import get_important_frames
from .detector import detect_important_frames
from .types import DetectionResult, ImportantFramesError

__all__ = [
    "DetectionResult",
    "ImportantFramesError",
    "detect_important_frames",
    "get_important_frames",
]
