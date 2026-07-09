from __future__ import annotations

import base64

import cv2
import numpy as np

from preprocessing.types import Frame, PreprocessingError


def _encode_grid_b64(grid: np.ndarray, quality: int) -> str:
    ok, buf = cv2.imencode(".jpg", grid, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        raise PreprocessingError("Failed to encode grid to JPEG")
    return base64.b64encode(buf.tobytes()).decode("utf-8")


def _cell_size(frames: list[Frame]) -> tuple[int, int]:
    max_h = max(f.image.shape[0] for f in frames)
    max_w = max(f.image.shape[1] for f in frames)
    return max_h, max_w


def _fit_into_cell(image: np.ndarray, cell_h: int, cell_w: int) -> np.ndarray:
    h, w = image.shape[:2]
    scale = min(cell_w / w, cell_h / h)
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

    canvas = np.zeros((cell_h, cell_w, 3), dtype=np.uint8)
    y0 = (cell_h - new_h) // 2
    x0 = (cell_w - new_w) // 2
    canvas[y0 : y0 + new_h, x0 : x0 + new_w] = resized
    return canvas


def _build_grid(frames: list[Frame], cols: int, rows: int) -> np.ndarray:
    cell_h, cell_w = _cell_size(frames)
    grid_h, grid_w = cell_h * rows, cell_w * cols
    canvas = np.zeros((grid_h, grid_w, 3), dtype=np.uint8)

    for slot, frame in enumerate(frames):
        row = slot // cols
        col = slot % cols
        y0 = row * cell_h
        x0 = col * cell_w
        canvas[y0 : y0 + cell_h, x0 : x0 + cell_w] = _fit_into_cell(frame.image, cell_h, cell_w)

    return canvas


def frames_to_grid_b64(
    frames: list[Frame],
    cols: int = 4,
    rows: int = 4,
    quality: int = 85,
) -> list[str]:
    """Tile frames into one or more grid montages and return base64 JPEG strings."""
    if not frames:
        return []
    if cols <= 0 or rows <= 0:
        raise PreprocessingError("grid cols and rows must be positive")

    capacity = cols * rows
    grids: list[str] = []

    for start in range(0, len(frames), capacity):
        chunk = frames[start : start + capacity]
        grid = _build_grid(chunk, cols, rows)
        grids.append(_encode_grid_b64(grid, quality))

    return grids
