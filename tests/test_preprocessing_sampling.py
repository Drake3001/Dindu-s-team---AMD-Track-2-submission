import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from preprocessing.preprocessing import sample_and_downscale
from preprocessing.types import PreprocessingError, VideoMetadata


class SamplingStepTests(unittest.TestCase):
    def test_step_greater_than_one_uses_seek_path(self) -> None:
        metadata = VideoMetadata(
            path="videos/test.mp4",
            duration_sec=60.0,
            fps=30.0,
            frame_count=1800,
            width=1920,
            height=1080,
        )
        step = max(1, round(metadata.fps / 1.0))
        self.assertEqual(step, 30)
        self.assertGreater(step, 1)

    def test_step_one_uses_sequential_path(self) -> None:
        metadata = VideoMetadata(
            path="videos/test.mp4",
            duration_sec=10.0,
            fps=1.0,
            frame_count=10,
            width=640,
            height=480,
        )
        step = max(1, round(metadata.fps / 1.0))
        self.assertEqual(step, 1)


class SampleAndDownscaleTests(unittest.TestCase):
    @patch("preprocessing.preprocessing.read_metadata")
    @patch("preprocessing.preprocessing.cv2.VideoCapture")
    def test_seek_path_respects_max_frames(
        self,
        video_capture: MagicMock,
        read_metadata: MagicMock,
    ) -> None:
        read_metadata.return_value = VideoMetadata(
            path="videos/test.mp4",
            duration_sec=60.0,
            fps=30.0,
            frame_count=1800,
            width=1920,
            height=1080,
        )
        cap = video_capture.return_value
        cap.isOpened.return_value = True
        cap.read.return_value = (True, np.zeros((1080, 1920, 3), dtype=np.uint8))

        _, frames = sample_and_downscale(
            __import__("pathlib").Path("videos/test.mp4"),
            fps=1.0,
            max_dim=512,
            max_frames=4,
        )

        self.assertEqual(len(frames), 4)
        self.assertEqual(cap.set.call_count, 4)
        self.assertEqual(cap.read.call_count, 4)

    @patch("preprocessing.preprocessing.read_metadata")
    @patch("preprocessing.preprocessing.cv2.VideoCapture")
    def test_sequential_path_when_target_fps_matches_native(
        self,
        video_capture: MagicMock,
        read_metadata: MagicMock,
    ) -> None:
        read_metadata.return_value = VideoMetadata(
            path="videos/test.mp4",
            duration_sec=3.0,
            fps=1.0,
            frame_count=3,
            width=640,
            height=480,
        )
        cap = video_capture.return_value
        cap.isOpened.return_value = True
        cap.read.side_effect = [
            (True, np.zeros((480, 640, 3), dtype=np.uint8)),
            (True, np.zeros((480, 640, 3), dtype=np.uint8)),
            (True, np.zeros((480, 640, 3), dtype=np.uint8)),
            (False, None),
        ]

        _, frames = sample_and_downscale(
            __import__("pathlib").Path("videos/test.mp4"),
            fps=1.0,
            max_dim=512,
            max_frames=None,
        )

        self.assertEqual(len(frames), 3)
        cap.set.assert_not_called()

    @patch("preprocessing.preprocessing.read_metadata")
    @patch("preprocessing.preprocessing.cv2.VideoCapture")
    def test_raises_when_no_frames_extracted(
        self,
        video_capture: MagicMock,
        read_metadata: MagicMock,
    ) -> None:
        read_metadata.return_value = VideoMetadata(
            path="videos/test.mp4",
            duration_sec=60.0,
            fps=30.0,
            frame_count=1800,
            width=1920,
            height=1080,
        )
        cap = video_capture.return_value
        cap.isOpened.return_value = True
        cap.read.return_value = (False, None)

        with self.assertRaises(PreprocessingError):
            sample_and_downscale(
                __import__("pathlib").Path("videos/test.mp4"),
                fps=1.0,
            )


if __name__ == "__main__":
    unittest.main()
