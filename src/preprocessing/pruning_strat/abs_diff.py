from __future__ import annotations

import cv2
import numpy as np

from preprocessing.types import Frame


class AbsDiffPruner:
    """Keep frames that differ enough from the last kept frame; drop near-duplicates."""

    def __init__(self, threshold: float = 5.0) -> None:
        self.threshold = threshold

    def _to_gray_small(self, image: np.ndarray) -> np.ndarray:
        return cv2.cvtColor(cv2.resize(image, (64, 36)), cv2.COLOR_BGR2GRAY)

    def prune(self, frames: list[Frame]) -> list[Frame]:
        if not frames:
            return []

        kept: list[Frame] = []
        last_gray: np.ndarray | None = None

        for frame in frames:
            gray = self._to_gray_small(frame.image)
            if last_gray is None:
                kept.append(frame)
                last_gray = gray
                continue

            diff = float(np.mean(cv2.absdiff(gray, last_gray)))
            if diff >= self.threshold:
                kept.append(frame)
                last_gray = gray

        return [
            Frame(index=i, timestamp=f.timestamp, image=f.image)
            for i, f in enumerate(kept)
        ]
