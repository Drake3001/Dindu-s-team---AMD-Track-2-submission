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
DEFAULT_ALPHA: float = 0.01
"""EMA smoothing factor.  Lower → slower adaptation, keeps a stable
background model that ignores gradual motion.  At 0.01 the half-life is
~69 frames (~2.3 s at 30 fps), so only abrupt departures break through."""

DEFAULT_THRESHOLD: float = 10.0
"""Base pixel-difference threshold (0–255 scale) for the 98th percentile.
If at least 2% of the frame deviates from the EMA by this amount + the
adaptive baseline, it is flagged as important."""

DEFAULT_DIFF_ALPHA: float = 0.05
"""EMA smoothing factor for the difference score itself. Used to track
the 'normal' amount of chaos in the scene (e.g. constant traffic)."""

DEFAULT_DIFF_MULTIPLIER: float = 1.5
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

def detect_important_frames(
    video_path: str | Path,
    *,
    alpha: float = DEFAULT_ALPHA,
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

        frame_idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            gray = _to_gray(frame)
            small = _downscale(gray, max_dim).astype(np.float32)

            if ema is None:
                # First frame always counts as important and seeds the EMA.
                ema = small.copy()
                ema_diff = 0.0
                important.append(frame_idx)
            else:
                diff = float(np.percentile(np.abs(small - ema), 98))
                
                # Adaptive threshold logic
                dynamic_threshold = (ema_diff * diff_multiplier) + threshold
                if diff >= dynamic_threshold:
                    important.append(frame_idx)
                
                # Always update the EMAs so they track gradual/normal changes.
                ema = alpha * small + (1.0 - alpha) * ema
                ema_diff = diff_alpha * diff + (1.0 - diff_alpha) * ema_diff

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
            "alpha": alpha,
            "threshold": threshold,
        },
    )

    return DetectionResult(
        important_frames=result_array,
        total_frames=total,
        fps=fps,
    )
