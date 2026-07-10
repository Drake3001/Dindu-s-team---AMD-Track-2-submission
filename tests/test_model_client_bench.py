import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from model_client.bench import _write_report, main, process_task
from model_client.config import ModelConfig
from model_client.prompts import Prompt
from model_client.types import ModelRequestError

TEST_PROMPT = Prompt(name="test", system="System", user="User")


class ModelClientBenchTests(unittest.TestCase):
    @patch("model_client.bench.preprocess_video")
    @patch("model_client.bench._resolve_video_path")
    def test_process_task_reports_pipeline_stages(
        self,
        resolve_video_path: Mock,
        preprocess_video: Mock,
    ) -> None:
        video_path = Path("videos/v1.mp4")
        resolve_video_path.return_value = video_path
        preprocess_video.return_value = _preprocess_result(["grid1", "grid2"])
        model_client = _model_client(["first response", "second response"])

        result = process_task(
            {"task_id": "v1", "video_url": "https://example.test/v1.mp4"},
            model_client,
            Path("videos"),
            True,
            1.0,
            384,
            5.0,
            4,
            4,
            8,
            [TEST_PROMPT],
            False,
        )

        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["video_path"], str(video_path))
        self.assertEqual(result["counts"]["grids"], 2)
        self.assertEqual(result["counts"]["model_requests"], 2)
        self.assertEqual(result["model"]["provider"], "openrouter")
        self.assertEqual(result["outputs"][0]["prompt"], "test")
        self.assertIn("elapsed_s", result["outputs"][0])
        self.assertEqual(result["outputs"][0]["response_chars"], len("first response"))
        self.assertNotIn("response", result["outputs"][0])
        self.assertIn("valid_json", result["outputs"][0])
        self.assertIn("model_request", result["timings_s"])

    @patch("model_client.bench.preprocess_video")
    @patch("model_client.bench._resolve_video_path")
    def test_process_task_can_include_full_responses(
        self,
        resolve_video_path: Mock,
        preprocess_video: Mock,
    ) -> None:
        resolve_video_path.return_value = Path("videos/v1.mp4")
        preprocess_video.return_value = _preprocess_result(["grid1"])
        model_client = _model_client(["full response"])

        result = process_task(
            {"task_id": "v1", "video_url": "https://example.test/v1.mp4"},
            model_client,
            Path("videos"),
            True,
            1.0,
            384,
            5.0,
            4,
            4,
            8,
            [TEST_PROMPT],
            True,
        )

        self.assertEqual(result["outputs"][0]["response"], "full response")

    @patch("model_client.bench.preprocess_video")
    @patch("model_client.bench._resolve_video_path")
    def test_process_task_preserves_outputs_when_later_grid_fails(
        self,
        resolve_video_path: Mock,
        preprocess_video: Mock,
    ) -> None:
        resolve_video_path.return_value = Path("videos/v1.mp4")
        preprocess_video.return_value = _preprocess_result(["grid1", "grid2"])
        model_client = _model_client(["first response", ModelRequestError("model failed")])

        result = process_task(
            {"task_id": "v1", "video_url": "https://example.test/v1.mp4"},
            model_client,
            Path("videos"),
            True,
            1.0,
            384,
            5.0,
            4,
            4,
            8,
            [TEST_PROMPT],
            False,
        )

        self.assertEqual(result["status"], "partial")
        self.assertEqual(len(result["outputs"]), 2)
        self.assertEqual(result["outputs"][0]["status"], "ok")
        self.assertEqual(result["outputs"][1]["status"], "failed")
        self.assertIn("model failed", result["outputs"][1]["error"])

    def test_write_report_uses_vlm_output_subdir(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            report_path = _write_report(Path(tmpdir), {"ok": True})
            self.assertTrue(report_path.is_file())

        self.assertIn("vlm_output", str(report_path))
        self.assertTrue(report_path.name.startswith("bench_"))

    @patch("model_client.bench.load_prompts")
    @patch("model_client.bench._write_report")
    @patch("model_client.bench.process_task")
    @patch("model_client.bench.create_model_client")
    @patch("model_client.bench.load_input")
    def test_main_records_failed_tasks(
        self,
        load_input: Mock,
        create_model_client: Mock,
        process_task_mock: Mock,
        write_report: Mock,
        load_prompts: Mock,
    ) -> None:
        load_input.return_value = [
            {"task_id": "v1", "video_url": "https://example.test/v1.mp4"},
        ]
        load_prompts.return_value = [TEST_PROMPT]
        create_model_client.return_value = _model_client([])
        process_task_mock.side_effect = ModelRequestError("model failed")
        write_report.return_value = Path("output/vlm_output/bench_test.json")

        main(runs=1)

        report = write_report.call_args.args[1]
        task = report["tasks"][0]
        self.assertEqual(task["status"], "failed")
        self.assertEqual(task["task_id"], "v1")
        self.assertIn("model failed", task["error"])

    @patch("model_client.bench.load_prompts")
    @patch("model_client.bench._write_report")
    @patch("model_client.bench.process_task")
    @patch("model_client.bench.create_model_client")
    @patch("model_client.bench.load_input")
    def test_main_writes_all_tasks_when_one_fails(
        self,
        load_input: Mock,
        create_model_client: Mock,
        process_task_mock: Mock,
        write_report: Mock,
        load_prompts: Mock,
    ) -> None:
        load_input.return_value = [
            {"task_id": "v1", "video_url": "https://example.test/v1.mp4"},
            {"task_id": "v2", "video_url": "https://example.test/v2.mp4"},
        ]
        load_prompts.return_value = [TEST_PROMPT]
        create_model_client.return_value = _model_client([])
        process_task_mock.side_effect = [
            {
                "task_id": "v1",
                "status": "ok",
                "timings_s": {"total": 1.0},
                "counts": {"grids": 1, "model_requests": 1},
            },
            ModelRequestError("model failed"),
        ]
        write_report.return_value = Path("output/vlm_output/bench_test.json")

        main(runs=1)

        report = write_report.call_args.args[1]
        tasks = report["tasks"]
        self.assertEqual(len(tasks), 2)
        self.assertEqual(tasks[0]["status"], "ok")
        self.assertEqual(tasks[1]["status"], "failed")
        self.assertIn("model failed", tasks[1]["error"])


def _model_client(responses: list[str]) -> Mock:
    client = Mock()
    client.config = ModelConfig(
        provider="openrouter",
        api_key="secret",
        model="vision/test",
        base_url="https://openrouter.ai/api/v1",
        timeout_seconds=60.0,
        temperature=0.7,
        max_tokens=512,
    )
    client.generate_from_frame_grid_base64.side_effect = responses
    return client


def _preprocess_result(grids_b64: list[str]) -> SimpleNamespace:
    grids = [
        SimpleNamespace(
            b64=b64,
            frame_count=8,
            capacity=16,
            empty_cells=8,
            cols=4,
            rows=4,
            width_px=512,
            height_px=512,
        )
        for b64 in grids_b64
    ]
    return SimpleNamespace(
        metadata=SimpleNamespace(
            duration_sec=6.0,
            fps=25.0,
            frame_count=150,
            width=1920,
            height=1080,
        ),
        sampled_count=6,
        post_pruned_count=6,
        grids=grids,
        grids_b64=grids_b64,
    )
