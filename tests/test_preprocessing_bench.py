import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from preprocessing.bench import _b64_sizes, main
from preprocessing.types import PreprocessingError


class PreprocessingBenchB64Tests(unittest.TestCase):
    def test_b64_sizes_reports_per_grid_and_totals(self) -> None:
        # "YQ==" decodes to 1 byte; base64 string is 4 chars
        sizes = _b64_sizes(["YQ==", "YWI="])

        self.assertEqual(len(sizes["per_grid"]), 2)
        self.assertEqual(sizes["per_grid"][0]["b64_bytes"], 4)
        self.assertEqual(sizes["per_grid"][0]["decoded_bytes"], 1)
        self.assertEqual(sizes["per_grid"][1]["b64_bytes"], 4)
        self.assertEqual(sizes["per_grid"][1]["decoded_bytes"], 2)
        self.assertEqual(sizes["total_b64_bytes"], 8)
        self.assertEqual(sizes["total_decoded_bytes"], 3)
        self.assertEqual(sizes["avg_b64_bytes"], 4)

    def test_b64_sizes_empty_list(self) -> None:
        sizes = _b64_sizes([])
        self.assertEqual(sizes["per_grid"], [])
        self.assertEqual(sizes["total_b64_bytes"], 0)
        self.assertEqual(sizes["avg_b64_bytes"], 0)


class PreprocessingBenchFailureTests(unittest.TestCase):
    @patch("preprocessing.bench._write_report")
    @patch("preprocessing.bench.process_task")
    @patch("preprocessing.bench.load_input")
    def test_main_writes_all_tasks_when_one_fails(
        self,
        load_input: Mock,
        process_task_mock: Mock,
        write_report: Mock,
    ) -> None:
        load_input.return_value = [
            {"task_id": "v1", "video_url": "https://example.test/v1.mp4"},
            {"task_id": "v2", "video_url": "https://example.test/v2.mp4"},
        ]
        process_task_mock.side_effect = [
            {
                "task_id": "v1",
                "status": "ok",
                "timings_s": {"total": 1.0},
                "counts": {"sampled": 1, "post_pruned": 1, "grids": 1},
                "base64_sizes": {"total_b64_kb": 10.0},
            },
            PreprocessingError("preprocess failed"),
        ]
        write_report.return_value = Path("output/processing/bench_test.json")

        main(runs=1)

        report = write_report.call_args.args[1]
        tasks = report["tasks"]
        self.assertEqual(len(tasks), 2)
        self.assertEqual(tasks[0]["status"], "ok")
        self.assertEqual(tasks[1]["status"], "failed")
        self.assertIn("preprocess failed", tasks[1]["error"])


if __name__ == "__main__":
    unittest.main()
