"""
End-to-end benchmark for video preprocessing plus VLM calls.

Usage:
    uv run bench2
    uv run bench2 --skip_download=True
    uv run bench2 --fps 1 --max_dim 384 --max_frames 8
    uv run bench2 --prompt detailed_chronological
    uv run bench2 --include_responses=True
"""

from __future__ import annotations

import asyncio
import gc
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import fire
import structlog

from file_io.api import configure_logging, load_input
from file_io.download import download_for_task, expected_video_path
from model_client.api import create_async_model_client, create_model_client
from model_client.config import ModelConfig
from model_client.prompts import Prompt, load_prompts
from model_client.types import ModelRequestError
from preprocessing.preprocessing import (
    DEFAULT_GRID_COLS,
    DEFAULT_GRID_ROWS,
    DEFAULT_MAX_DIM,
    DEFAULT_MAX_FRAMES,
    DEFAULT_PRUNE_THRESHOLD,
    DEFAULT_TARGET_FPS,
    preprocess_video,
)
from preprocessing.types import PreprocessingError
from workflow.async_pipeline import PipelineConfig, parallelism_params, run_bench_tasks

log = structlog.get_logger(__name__)

DEFAULT_TASKS_PATH = Path("input/tasks.json")
DEFAULT_VIDEOS_DIR = Path("videos")
DEFAULT_OUTPUT_DIR = Path("output")
BENCH_SUBDIR = "vlm_output"
RESPONSE_PREVIEW_CHARS = 240


def _write_report(output_dir: Path, report: dict) -> Path:
    bench_dir = output_dir / BENCH_SUBDIR
    bench_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    report_path = bench_dir / f"bench_{timestamp}.json"
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
        f.write("\n")
    return report_path


def _resolve_video_path(task: dict, videos_dir: Path, skip_download: bool) -> Path:
    task_id = task["task_id"]
    video_url = task["video_url"]

    if skip_download:
        path = expected_video_path(task_id, video_url, videos_dir)
        if not path.is_file():
            raise PreprocessingError(f"Expected video not found: {path}")
        return path

    return download_for_task(task_id, video_url, videos_dir)


def _model_info(config: ModelConfig) -> dict:
    return {
        "provider": config.provider,
        "model": config.model,
        "temperature": config.temperature,
        "max_tokens": config.max_tokens,
    }


def _output_record(
    prompt_name: str,
    response: str,
    elapsed_s: float,
    include_response: bool,
    grids_sent: int,
) -> dict:
    valid_json = False
    try:
        json.loads(response)
        valid_json = True
    except json.JSONDecodeError:
        pass

    record = {
        "prompt": prompt_name,
        "elapsed_s": elapsed_s,
        "status": "ok",
        "response_chars": len(response),
        "response_preview": response[:RESPONSE_PREVIEW_CHARS],
        "valid_json": valid_json,
        "grids_sent": grids_sent,
    }
    if include_response:
        record["response"] = response
    return record


def _failed_output_record(
    prompt_name: str,
    elapsed_s: float,
    error: Exception,
    grids_sent: int,
) -> dict:
    return {
        "prompt": prompt_name,
        "elapsed_s": elapsed_s,
        "status": "failed",
        "error": str(error),
        "grids_sent": grids_sent,
    }


def _task_status_from_outputs(outputs: list[dict]) -> str:
    if not outputs:
        return "failed"
    failures = sum(1 for output in outputs if output.get("status") == "failed")
    if failures == 0:
        return "ok"
    if failures == len(outputs):
        return "failed"
    return "partial"


def process_task(
    task: dict,
    model_client,
    videos_dir: Path,
    skip_download: bool,
    fps: float,
    max_dim: int,
    prune_threshold: float,
    grid_cols: int,
    grid_rows: int,
    max_frames: int | None,
    prompts: list[Prompt],
    include_responses: bool,
) -> dict:
    task_id = task["task_id"]
    t0 = time.perf_counter()

    video_path = _resolve_video_path(task, videos_dir, skip_download)
    t1 = time.perf_counter()

    preprocessed = preprocess_video(
        task_id=task_id,
        video_path=video_path,
        fps=fps,
        max_dim=max_dim,
        prune_threshold=prune_threshold,
        grid_cols=grid_cols,
        grid_rows=grid_rows,
        max_frames=max_frames,
    )
    t2 = time.perf_counter()

    grids_b64 = [grid.b64 for grid in preprocessed.grids]
    grids_meta = [
        {
            "frame_count": grid.frame_count,
            "cols": grid.cols,
            "rows": grid.rows,
            "empty_cells": grid.empty_cells,
            "width_px": grid.width_px,
            "height_px": grid.height_px,
        }
        for grid in preprocessed.grids
    ]
    grids_sent = len(grids_b64)

    outputs = []
    for prompt in prompts:
        call_start = time.perf_counter()
        if not grids_b64:
            elapsed_s = round(time.perf_counter() - call_start, 4)
            error = ModelRequestError("No grid images available for model request")
            outputs.append(
                _failed_output_record(prompt.name, elapsed_s, error, grids_sent)
            )
            log.error(
                "model_grid_request_failed",
                task_id=task_id,
                prompt=prompt.name,
                error=str(error),
            )
            continue

        try:
            response = model_client.generate_from_frame_grids(
                grids_b64,
                prompt.system,
                prompt.user,
                grids_meta=grids_meta,
            )
        except ModelRequestError as exc:
            elapsed_s = round(time.perf_counter() - call_start, 4)
            outputs.append(
                _failed_output_record(prompt.name, elapsed_s, exc, grids_sent)
            )
            log.error(
                "model_grid_request_failed",
                task_id=task_id,
                prompt=prompt.name,
                error=str(exc),
            )
            continue

        elapsed_s = round(time.perf_counter() - call_start, 4)
        outputs.append(
            _output_record(
                prompt.name,
                response,
                elapsed_s,
                include_responses,
                grids_sent,
            )
        )
    t3 = time.perf_counter()

    gc.collect()
    t4 = time.perf_counter()

    meta = preprocessed.metadata
    return {
        "task_id": task_id,
        "video_url": task["video_url"],
        "video_path": str(video_path),
        "metadata": {
            "duration_sec": meta.duration_sec,
            "fps": meta.fps,
            "frame_count": meta.frame_count,
            "width": meta.width,
            "height": meta.height,
            "resolution": f"{meta.width}x{meta.height}",
        },
        "counts": {
            "sampled": preprocessed.sampled_count,
            "post_pruned": preprocessed.post_pruned_count,
            "grids": len(preprocessed.grids),
            "model_requests": len(outputs),
        },
        "timings_s": {
            "resolve_video": round(t1 - t0, 4),
            "preprocess": round(t2 - t1, 4),
            "model_request": round(t3 - t2, 4),
            "cleanup": round(t4 - t3, 4),
            "total": round(t4 - t0, 4),
        },
        "model": _model_info(model_client.config),
        "outputs": outputs,
        "status": _task_status_from_outputs(outputs),
    }


def _handle_bench_task_error(task: dict, error: Exception, model_config: ModelConfig) -> dict:
    task_id = task.get("task_id", "unknown")
    log.error("model_task_failed", task_id=task_id, error=str(error))
    return {
        "task_id": task_id,
        "video_url": task.get("video_url"),
        "model": _model_info(model_config),
        "status": "failed",
        "error": str(error),
    }


def _log_completed_tasks(
    run_results: list[dict],
    *,
    run_idx: int,
    runs: int,
) -> None:
    for result in run_results:
        status = result.get("status")
        if status in {"ok", "partial"}:
            timings = result.get("timings_s", {})
            counts = result.get("counts", {})
            log.info(
                "model_task_complete",
                run=run_idx + 1,
                runs=runs,
                task_id=result.get("task_id"),
                status=status,
                total_s=timings.get("total"),
                grids=counts.get("grids"),
                model_requests=counts.get("model_requests"),
            )


async def _run_parallel_bench(
    task_list: list[dict],
    *,
    videos_path: Path,
    skip_download: bool,
    prompts: list[Prompt],
    include_responses: bool,
    preprocess_kwargs: dict,
    pipeline_config: PipelineConfig,
    model_config: ModelConfig,
) -> list[dict]:
    model_client = create_async_model_client(
        provider=model_config.provider,
        api_key=model_config.api_key,
        model=model_config.model,
        base_url=model_config.base_url,
        timeout_seconds=model_config.timeout_seconds,
        temperature=model_config.temperature,
        max_tokens=model_config.max_tokens,
    )

    def on_error(task: dict, error: Exception) -> dict:
        return _handle_bench_task_error(task, error, model_config)

    return await run_bench_tasks(
        task_list,
        model_client=model_client,
        videos_dir=videos_path,
        skip_download=skip_download,
        prompts=prompts,
        include_responses=include_responses,
        preprocess_kwargs=preprocess_kwargs,
        config=pipeline_config,
        on_error=on_error,
    )


def main(
    tasks: str = str(DEFAULT_TASKS_PATH),
    videos_dir: str = str(DEFAULT_VIDEOS_DIR),
    output_dir: str = str(DEFAULT_OUTPUT_DIR),
    fps: float = DEFAULT_TARGET_FPS,
    max_dim: int = DEFAULT_MAX_DIM,
    prune_threshold: float = DEFAULT_PRUNE_THRESHOLD,
    grid_cols: int = DEFAULT_GRID_COLS,
    grid_rows: int = DEFAULT_GRID_ROWS,
    max_frames: int | None = DEFAULT_MAX_FRAMES,
    skip_download: bool = False,
    runs: int = 1,
    prompt: str | None = None,
    include_responses: bool = False,
    temperature: float | None = None,
    max_tokens: int | None = None,
    timeout_seconds: float | None = None,
    sequential: bool = False,
    max_download_workers: int = 2,
    max_preprocess_workers: int | None = None,
    max_inference_workers: int = 3,
) -> None:
    """Benchmark preprocessing plus model calls for all tasks in a JSON file."""
    configure_logging()

    if runs < 1:
        raise ValueError("runs must be >= 1")

    tasks_path = Path(tasks)
    videos_path = Path(videos_dir)
    out_path = Path(output_dir)
    task_list = load_input(tasks_path)
    prompts = load_prompts([prompt] if prompt else None)
    pipeline_config = PipelineConfig(
        max_download_workers=max_download_workers,
        max_preprocess_workers=max_preprocess_workers,
        max_inference_workers=max_inference_workers,
    )
    preprocess_kwargs = {
        "fps": fps,
        "max_dim": max_dim,
        "prune_threshold": prune_threshold,
        "grid_cols": grid_cols,
        "grid_rows": grid_rows,
        "max_frames": max_frames,
    }

    params = {
        "tasks": str(tasks_path),
        "videos_dir": str(videos_path),
        "output_dir": str(out_path),
        "fps": fps,
        "max_dim": max_dim,
        "prune_threshold": prune_threshold,
        "grid_cols": grid_cols,
        "grid_rows": grid_rows,
        "max_frames": max_frames,
        "skip_download": skip_download,
        "runs": runs,
        "prompts": [p.name for p in prompts],
        "include_responses": include_responses,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "timeout_seconds": timeout_seconds,
        "sequential": sequential,
        "parallelism": parallelism_params(pipeline_config, len(task_list)),
    }

    log.info("model_benchmark_started", **params, num_tasks=len(task_list))

    all_runs: list[dict] = []
    for run_idx in range(runs):
        if sequential:
            model_client = create_model_client(
                temperature=temperature,
                max_tokens=max_tokens,
                timeout_seconds=timeout_seconds,
            )
            run_results: list[dict] = []
            for task in task_list:
                try:
                    result = process_task(
                        task,
                        model_client,
                        videos_path,
                        skip_download,
                        fps,
                        max_dim,
                        prune_threshold,
                        grid_cols,
                        grid_rows,
                        max_frames,
                        prompts,
                        include_responses,
                    )
                except (
                    PreprocessingError,
                    ModelRequestError,
                    KeyError,
                    FileNotFoundError,
                    OSError,
                ) as exc:
                    result = _handle_bench_task_error(task, exc, model_client.config)

                run_results.append(result)
            _log_completed_tasks(run_results, run_idx=run_idx, runs=runs)
        else:
            sync_model_client = create_model_client(
                temperature=temperature,
                max_tokens=max_tokens,
                timeout_seconds=timeout_seconds,
            )
            run_results = asyncio.run(
                _run_parallel_bench(
                    task_list,
                    videos_path=videos_path,
                    skip_download=skip_download,
                    prompts=prompts,
                    include_responses=include_responses,
                    preprocess_kwargs=preprocess_kwargs,
                    pipeline_config=pipeline_config,
                    model_config=sync_model_client.config,
                )
            )
            _log_completed_tasks(run_results, run_idx=run_idx, runs=runs)

        all_runs.append({"run": run_idx + 1, "tasks": run_results})

    report = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "params": params,
        "runs": all_runs,
    }

    if runs == 1:
        report["tasks"] = all_runs[0]["tasks"]

    report_path = _write_report(out_path, report)
    log.info("model_benchmark_report_written", path=str(report_path))


def cli() -> None:
    fire.Fire(main)


if __name__ == "__main__":
    cli()
