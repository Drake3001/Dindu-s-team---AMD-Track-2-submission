import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

from file_io.show_output import main, print_task_outputs, resolve_report_path


class ShowOutputTests(unittest.TestCase):
    def test_resolve_report_path_uses_vlm_output_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            report_dir = Path(tmpdir) / "output" / "vlm_output"
            report_dir.mkdir(parents=True)
            report_path = report_dir / "bench_test.json"
            report_path.write_text("{}", encoding="utf-8")

            original = Path("output") / "vlm_output"
            with mock.patch("file_io.show_output.DEFAULT_VLM_OUTPUT_DIR", report_dir):
                resolved = resolve_report_path("bench_test.json")

            self.assertEqual(resolved, report_path)

    def test_print_task_outputs_pretty_prints_valid_json(self) -> None:
        report = {
            "tasks": [
                {
                    "task_id": "v1",
                    "outputs": [
                        {
                            "prompt": "concise_factual",
                            "valid_json": True,
                            "response": '{"setting": "street", "subjects": ["car"]}',
                        }
                    ],
                }
            ]
        }

        buffer = io.StringIO()
        with redirect_stdout(buffer):
            print_task_outputs(report, "v1")

        output = buffer.getvalue()
        self.assertIn("--- v1 / concise_factual (valid_json=True)", output)
        self.assertIn('"setting": "street"', output)
        self.assertIn('"subjects": [\n    "car"\n  ]', output)

    def test_print_task_outputs_uses_preview_when_response_missing(self) -> None:
        report = {
            "tasks": [
                {
                    "task_id": "v1",
                    "outputs": [
                        {
                            "prompt": "detailed_chronological",
                            "valid_json": False,
                            "response_preview": "partial text",
                        }
                    ],
                }
            ]
        }

        buffer = io.StringIO()
        with redirect_stdout(buffer):
            print_task_outputs(report, "v1")

        output = buffer.getvalue()
        self.assertIn("partial text", output)
        self.assertIn("--include_responses=True", output)

    def test_main_returns_error_for_unknown_task(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            report_path = Path(tmpdir) / "bench_test.json"
            report_path.write_text(
                json.dumps({"tasks": [{"task_id": "v1", "outputs": []}]}),
                encoding="utf-8",
            )

            buffer = io.StringIO()
            with redirect_stdout(buffer):
                code = main(str(report_path), "missing")

        self.assertEqual(code, 1)
        self.assertIn("Available task ids: v1", buffer.getvalue())

    def test_main_reads_multi_run_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            report_path = Path(tmpdir) / "bench_test.json"
            report_path.write_text(
                json.dumps(
                    {
                        "runs": [
                            {
                                "run": 2,
                                "tasks": [
                                    {
                                        "task_id": "v2",
                                        "outputs": [
                                            {
                                                "prompt": "test",
                                                "valid_json": False,
                                                "response": "run two response",
                                            }
                                        ],
                                    }
                                ],
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )

            buffer = io.StringIO()
            with redirect_stdout(buffer):
                code = main(str(report_path), "v2")

        self.assertEqual(code, 0)
        self.assertIn("run 2", buffer.getvalue())
        self.assertIn("run two response", buffer.getvalue())


if __name__ == "__main__":
    unittest.main()
