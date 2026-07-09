from __future__ import annotations

from typing import Protocol

from preprocessing.types import Frame


class FramePruner(Protocol):
    """Strategy for reducing a sampled frame sequence."""

    def prune(self, frames: list[Frame]) -> list[Frame]:
        """Return a pruned copy of the frame list with re-indexed positions."""
        ...
