import unittest

import numpy as np

from preprocessing.types import Frame
from preprocessing.vlm_output.grid import GridImage, frames_to_b64_list, frames_to_grid_b64


class GridImageTests(unittest.TestCase):
    def test_frames_to_grid_b64_reports_partial_last_grid(self) -> None:
        frames = [
            Frame(index=i, timestamp=float(i), image=np.zeros((32, 48, 3), dtype=np.uint8))
            for i in range(10)
        ]

        grids = frames_to_grid_b64(frames, cols=4, rows=4)

        self.assertEqual(len(grids), 1)
        grid = grids[0]
        self.assertIsInstance(grid, GridImage)
        self.assertEqual(grid.frame_count, 10)
        self.assertEqual(grid.capacity, 16)
        self.assertEqual(grid.empty_cells, 6)
        self.assertEqual(grid.cols, 4)
        self.assertEqual(grid.rows, 4)
        self.assertGreater(grid.width_px, 0)
        self.assertGreater(grid.height_px, 0)
        self.assertTrue(grid.b64)

    def test_frames_to_grid_b64_splits_into_multiple_grids(self) -> None:
        frames = [
            Frame(index=i, timestamp=float(i), image=np.zeros((16, 16, 3), dtype=np.uint8))
            for i in range(20)
        ]

        grids = frames_to_grid_b64(frames, cols=4, rows=4)

        self.assertEqual(len(grids), 2)
        self.assertEqual(grids[0].frame_count, 16)
        self.assertEqual(grids[0].empty_cells, 0)
        self.assertEqual(grids[1].frame_count, 4)
        self.assertEqual(grids[1].empty_cells, 12)

    def test_frames_to_grid_b64_empty_list(self) -> None:
        self.assertEqual(frames_to_grid_b64([]), [])

    def test_frames_to_b64_list_encodes_each_frame(self) -> None:
        frames = [
            Frame(index=i, timestamp=float(i), image=np.zeros((16, 16, 3), dtype=np.uint8))
            for i in range(3)
        ]

        encoded = frames_to_b64_list(frames)

        self.assertEqual(len(encoded), 3)
        self.assertTrue(all(frame_b64 for frame_b64 in encoded))

    def test_frames_to_b64_list_empty_list(self) -> None:
        self.assertEqual(frames_to_b64_list([]), [])


if __name__ == "__main__":
    unittest.main()
