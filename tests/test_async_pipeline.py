import asyncio
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from model_client.prompts import Prompt
from workflow.async_pipeline import PipelineConfig, run_bench_tasks, run_workflow_tasks

TEST_PROMPT = Prompt(name="test", system="System", user="User")


class AsyncPipelineTests(unittest.IsolatedAsyncioTestCase):
    @patch("workflow.async_pipeline.async_generate_caption")
    @patch("workflow.async_pipeline._preprocess_video")
    @patch("workflow.async_pipeline._download_video")
    async def test_run_workflow_tasks_preserves_task_order(
        self,
        download_video: AsyncMock,
        preprocess_video: AsyncMock,
        async_generate_caption: AsyncMock,
    ) -> None:
        download_video.side_effect = [
            "videos/v1.mp4",
            "videos/v2.mp4",
        ]
        preprocess_video.side_effect = [
            _preprocessed_payload("v1"),
            _preprocessed_payload("v2"),
        ]
        async_generate_caption.side_effect = ["caption-a", "caption-b"]

        vlm_client = AsyncMock()
        vlm_client.generate_from_frame_grids = AsyncMock(return_value='{"ok": true}')
        caption_client = AsyncMock()
        caption_clients = {"formal": caption_client}

        tasks = [
            {"task_id": "v1", "video_url": "https://example.test/v1.mp4", "styles": ["formal"]},
            {"task_id": "v2", "video_url": "https://example.test/v2.mp4", "styles": ["formal"]},
        ]

        with patch("workflow.async_pipeline.ProcessPoolExecutor", return_value=MagicMock()):
            results = await run_workflow_tasks(
                tasks,
                vlm_client=vlm_client,
                caption_clients=caption_clients,
                caption_params_for_style=lambda _style: {"temperature": None, "max_tokens": None, "timeout_seconds": None},
                analysis_prompt=TEST_PROMPT,
                videos_dir=Path("videos"),
                styles_resolver=lambda task: task["styles"],
                config=PipelineConfig(max_preprocess_workers=1),
            )

        self.assertEqual([result["task_id"] for result in results], ["v1", "v2"])
        self.assertEqual(results[0]["captions"]["formal"], "caption-a")
        self.assertEqual(results[1]["captions"]["formal"], "caption-b")
        vlm_client.generate_from_frame_grids.assert_awaited()
        self.assertEqual(async_generate_caption.await_count, 2)
        for call in async_generate_caption.await_args_list:
            self.assertIs(call.args[0], caption_client)

    @patch("workflow.async_pipeline.run_bench_task")
    async def test_run_bench_tasks_runs_all_tasks(
        self,
        run_bench_task: AsyncMock,
    ) -> None:
        run_bench_task.side_effect = [
            {"task_id": "v1", "status": "ok"},
            {"task_id": "v2", "status": "ok"},
        ]

        model_client = AsyncMock()
        tasks = [
            {"task_id": "v1", "video_url": "https://example.test/v1.mp4"},
            {"task_id": "v2", "video_url": "https://example.test/v2.mp4"},
        ]

        with patch("workflow.async_pipeline.ProcessPoolExecutor", return_value=MagicMock()):
            results = await run_bench_tasks(
                tasks,
                model_client=model_client,
                videos_dir=Path("videos"),
                skip_download=True,
                prompts=[TEST_PROMPT],
                include_responses=False,
                preprocess_kwargs={"fps": 1.0},
            )

        self.assertEqual(len(results), 2)
        self.assertEqual(run_bench_task.await_count, 2)

    @patch("workflow.async_pipeline.async_generate_caption", new_callable=AsyncMock)
    async def test_tasks_overlap_across_stages(
        self,
        async_generate_caption: AsyncMock,
    ) -> None:
        events: list[str] = []
        async_generate_caption.return_value = "caption"

        async def fake_download(task, *_args, **_kwargs):
            task_id = task["task_id"]
            events.append(f"download_start:{task_id}")
            if task_id == "v1":
                await asyncio.sleep(0.08)
            else:
                await asyncio.sleep(0.01)
            events.append(f"download_end:{task_id}")
            return f"videos/{task_id}.mp4"

        async def fake_preprocess(task_id, *_args, **_kwargs):
            events.append(f"preprocess_start:{task_id}")
            await asyncio.sleep(0.02)
            events.append(f"preprocess_end:{task_id}")
            return _preprocessed_payload(task_id)

        vlm_client = AsyncMock()
        vlm_client.generate_from_frame_grids = AsyncMock(return_value='{"ok": true}')
        caption_client = AsyncMock()
        caption_clients = {"formal": caption_client}

        tasks = [
            {"task_id": "v1", "video_url": "https://example.test/v1.mp4", "styles": ["formal"]},
            {"task_id": "v2", "video_url": "https://example.test/v2.mp4", "styles": ["formal"]},
        ]

        with (
            patch("workflow.async_pipeline._download_video", side_effect=fake_download),
            patch("workflow.async_pipeline._preprocess_video", side_effect=fake_preprocess),
            patch("workflow.async_pipeline.ProcessPoolExecutor", return_value=MagicMock()),
        ):
            await run_workflow_tasks(
                tasks,
                vlm_client=vlm_client,
                caption_clients=caption_clients,
                caption_params_for_style=lambda _style: {"temperature": None, "max_tokens": None, "timeout_seconds": None},
                analysis_prompt=TEST_PROMPT,
                videos_dir=Path("videos"),
                styles_resolver=lambda task: task["styles"],
                config=PipelineConfig(max_preprocess_workers=2, max_inference_workers=2),
            )

        self.assertIn("download_start:v1", events)
        self.assertIn("download_start:v2", events)
        preprocess_v2_start = events.index("preprocess_start:v2")
        download_v1_end = events.index("download_end:v1")
        self.assertLess(
            preprocess_v2_start,
            download_v1_end,
            "v2 should begin preprocessing before v1 finishes downloading",
        )
        vlm_client.generate_from_frame_grids.assert_awaited()
        caption_client.generate_from_frame_grids.assert_not_awaited()


def _preprocessed_payload(task_id: str) -> dict:
    return {
        "task_id": task_id,
        "video_path": f"videos/{task_id}.mp4",
        "metadata": {
            "duration_sec": 6.0,
            "fps": 25.0,
            "frame_count": 150,
            "width": 1920,
            "height": 1080,
        },
        "grids_b64": ["grid1"],
        "grids_meta": [
            {
                "frame_count": 8,
                "cols": 4,
                "rows": 4,
                "empty_cells": 8,
                "width_px": 512,
                "height_px": 512,
            }
        ],
        "sampled_count": 6,
        "post_pruned_count": 6,
        "frame_timestamps": [0.0],
        "grids_count": 1,
    }


if __name__ == "__main__":
    unittest.main()
