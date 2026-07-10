"""EMA-based important-frame detector.

Pipeline per frame:
  1. Decode frame from video.
  2. Convert to greyscale.
  3. Downscale to *max_dim* on the longest edge.
  4. Compare against an Exponential Moving Average (EMA) of previous
     greyscale frames.  If the mean absolute difference exceeds
     *threshold*, the frame is flagged as important.
  5. Update the EMA:  ema ← α·current + (1 − α)·ema

Because the EMA is asymptotic it reacts *after* a sudden event starts
and settles *after* it ends.  To compensate, every threshold-crossing
frame is padded with extra context frames on both sides, and
overlapping regions are merged into contiguous spans.
"""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np

from .types import DetectionResult, ImportantFramesError

log = logging.getLogger(__name__)

# ── defaults ────────────────────────────────────────────────────────────
DEFAULT_FAST_ALPHA: float = 0.05
"""Fast EMA smoothing factor. Tracks the immediate state of the scene."""

DEFAULT_SLOW_ALPHA: float = 0.001
"""Slow EMA smoothing factor. Tracks the long-term background."""

DEFAULT_THRESHOLD: float = 30.0
"""Base pixel-difference threshold (0–255 scale)."""

DEFAULT_DIFF_ALPHA: float = 0.05
"""EMA smoothing factor for the difference score itself."""

DEFAULT_DIFF_MULTIPLIER: float = 0.0
"""Multiplier for the difference baseline. A frame is flagged if its
difference exceeds: `(baseline_diff * multiplier) + base_threshold`."""

DEFAULT_MAX_DIM: int = 128
"""Longest-edge cap used when downscaling greyscale frames for the EMA
comparison.  Smaller = faster but less spatial detail."""

DEFAULT_PADDING: int = 15
"""Number of extra context frames on each side of every threshold
crossing.  At 30 fps this gives ±500 ms — enough to capture the
approach and aftermath of an event despite the EMA's asymptotic lag."""


# ── helpers ─────────────────────────────────────────────────────────────

def _to_gray(frame: np.ndarray) -> np.ndarray:
    """Convert a BGR frame to single-channel greyscale."""
    if frame.ndim == 2:
        return frame
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


def _downscale(gray: np.ndarray, max_dim: int) -> np.ndarray:
    """Downscale *gray* so its longest edge is at most *max_dim*.

    Returns the image unchanged when it already fits.
    """
    h, w = gray.shape[:2]
    longest = max(h, w)
    if longest <= max_dim:
        return gray
    scale = max_dim / longest
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    return cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_AREA)


def _pad_and_merge(
    indices: list[int],
    padding: int,
    total_frames: int,
) -> np.ndarray:
    """Expand each index by ±*padding*, merge overlaps, return sorted array.

    Every important index is widened into the range
    ``[idx - padding, idx + padding]``, clamped to
    ``[0, total_frames - 1]``.  Overlapping or adjacent ranges are
    merged into contiguous spans so the final array contains no
    duplicates.
    """
    if not indices:
        return np.array([], dtype=np.intp)

    if padding <= 0:
        return np.array(sorted(set(indices)), dtype=np.intp)

    # Build closed [lo, hi] intervals.
    last = total_frames - 1
    intervals: list[tuple[int, int]] = []
    for idx in indices:
        lo = max(0, idx - padding)
        hi = min(last, idx + padding)
        intervals.append((lo, hi))

    # Sort and merge overlapping / adjacent intervals.
    intervals.sort()
    merged: list[tuple[int, int]] = [intervals[0]]
    for lo, hi in intervals[1:]:
        prev_lo, prev_hi = merged[-1]
        if lo <= prev_hi + 1:          # overlapping or adjacent
            merged[-1] = (prev_lo, max(prev_hi, hi))
        else:
            merged.append((lo, hi))

    # Expand merged intervals into individual frame indices.
    result: list[int] = []
    for lo, hi in merged:
        result.extend(range(lo, hi + 1))

    return np.array(result, dtype=np.intp)


# ── core detector ───────────────────────────────────────────────────────

class DetectorState:
    """Maintains the EMA and adaptive threshold state for processing video frames one by one."""
    def __init__(
        self,
        fast_alpha: float = DEFAULT_FAST_ALPHA,
        slow_alpha: float = DEFAULT_SLOW_ALPHA,
        threshold: float = DEFAULT_THRESHOLD,
        max_dim: int = DEFAULT_MAX_DIM,
        diff_alpha: float = DEFAULT_DIFF_ALPHA,
        diff_multiplier: float = DEFAULT_DIFF_MULTIPLIER,
    ):
        self.fast_alpha = fast_alpha
        self.slow_alpha = slow_alpha
        self.threshold = threshold
        self.max_dim = max_dim
        self.diff_alpha = diff_alpha
        self.diff_multiplier = diff_multiplier
        self.fast_ema: np.ndarray | None = None
        self.slow_ema: np.ndarray | None = None
        self.ema_diff: float = 15.0 # Initialize high to suppress startup junk

    def process_frame(self, frame: np.ndarray) -> tuple[bool, float]:
        """Evaluate a single RGB or grayscale frame. Returns (is_important, diff_score)."""
        gray = _to_gray(frame)
        small = _downscale(gray, self.max_dim).astype(np.float32)
        # Apply heavy blur to obliterate high-frequency noise like rippling water
        small = cv2.GaussianBlur(small, (11, 11), 0)

        if self.fast_ema is None:
            # First frame always counts as important and seeds the EMAs.
            self.fast_ema = small.copy()
            self.slow_ema = small.copy()
            return True, 0.0

        diff = float(np.max(np.abs(self.fast_ema - self.slow_ema)))
        dynamic_threshold = (self.ema_diff * self.diff_multiplier) + self.threshold
        is_important = diff >= dynamic_threshold
        
        # Score the structural complexity of the movement to prioritize crashes over camera shakes
        diff_mat_uint8 = np.abs(self.fast_ema - self.slow_ema).astype(np.uint8)
        lap_var = float(cv2.Laplacian(diff_mat_uint8, cv2.CV_64F).var())

        # Always update the EMAs so they track gradual/normal changes.
        self.fast_ema = self.fast_alpha * small + (1.0 - self.fast_alpha) * self.fast_ema
        self.slow_ema = self.slow_alpha * small + (1.0 - self.slow_alpha) * self.slow_ema
        self.ema_diff = self.diff_alpha * diff + (1.0 - self.diff_alpha) * self.ema_diff

        return is_important, lap_var


def detect_important_frames(
    video_path: str | Path,
    *,
    fast_alpha: float = DEFAULT_FAST_ALPHA,
    slow_alpha: float = DEFAULT_SLOW_ALPHA,
    threshold: float = DEFAULT_THRESHOLD,
    max_dim: int = DEFAULT_MAX_DIM,
    padding: int = DEFAULT_PADDING,
    diff_alpha: float = DEFAULT_DIFF_ALPHA,
    diff_multiplier: float = DEFAULT_DIFF_MULTIPLIER,
) -> DetectionResult:
    """Scan every frame of *video_path* and return the important ones.

    Parameters
    ----------
    video_path:
        Path to a video file readable by OpenCV.
    alpha:
        EMA smoothing factor (0 < α ≤ 1).  Lower values make the EMA
        slower to adapt, so brief bursts of change are more visible.
    threshold:
        Minimum base difference (0–255) at the 98th percentile between a
        frame and the current EMA to consider the frame *important*.
    max_dim:
        Longest edge to downscale greyscale frames to before comparison.
    padding:
        Number of extra context frames to include on each side of every
        threshold-crossing frame.  Compensates for the EMA's asymptotic
        lag — e.g. a car crash is captured from approach through
        aftermath.  Set to 0 to disable.
    diff_alpha:
        Smoothing factor for tracking the baseline difference score.
    diff_multiplier:
        Multiplier for the baseline difference to form the dynamic threshold.

    Returns
    -------
    DetectionResult
        Contains a 1-D ``numpy.ndarray`` of important frame indices
        (``dtype=np.intp``), the total frame count, and the source FPS.

    Raises
    ------
    ImportantFramesError
        If the video cannot be opened or contains no decodable frames.
    """
    video_path = Path(video_path)
    if not video_path.is_file():
        raise ImportantFramesError(f"Video file not found: {video_path}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ImportantFramesError(f"Cannot open video: {video_path}")

    try:
        fps: float = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total: int = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        ema: np.ndarray | None = None
        ema_diff: float = 0.0
        important: list[int] = []

        state = DetectorState(
            fast_alpha=fast_alpha,
            slow_alpha=slow_alpha,
            threshold=threshold,
            max_dim=max_dim,
            diff_alpha=diff_alpha,
            diff_multiplier=diff_multiplier,
        )

        frame_idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            if state.process_frame(frame):
                important.append(frame_idx)

            frame_idx += 1
    finally:
        cap.release()

    if frame_idx == 0:
        raise ImportantFramesError(
            f"No frames could be decoded from: {video_path}"
        )

    # Use actual decoded count if metadata was wrong / unavailable.
    if total <= 0:
        total = frame_idx

    # Pad each crossing with context frames and merge overlapping spans.
    result_array = _pad_and_merge(important, padding, total)

    log.info(
        "important_frames.detect",
        extra={
            "video": str(video_path),
            "total_frames": total,
            "raw_crossings": len(important),
            "important_count": len(result_array),
            "padding": padding,
            "fast_alpha": fast_alpha,
            "slow_alpha": slow_alpha,
            "threshold": threshold,
        },
    )

    return DetectionResult(
        important_frames=result_array,
        total_frames=total,
        fps=fps,
    )
