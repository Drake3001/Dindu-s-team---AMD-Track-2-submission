"""
Timing benchmark for the preprocessing pipeline.

Reads tasks from a JSON file, optionally downloads videos renamed to
task ids, then preprocesses each video sequentially while measuring
per-phase timings. Writes a JSON report under output/processing/.

Usage:
    uv run bench
    uv run bench --tasks input/tasks.json --skip_download=True
    uv run bench --fps 2 --max_dim 384 --prune_threshold 8
"""

from __future__ import annotations

import gc
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import fire
import structlog

from file_io.api import configure_logging, load_input
from file_io.download import download_for_task, expected_video_path

from .preprocessing import (
    DEFAULT_GRID_COLS,
    DEFAULT_GRID_ROWS,
    DEFAULT_MAX_DIM,
    DEFAULT_MAX_FRAMES,
    DEFAULT_PRUNE_THRESHOLD,
    DEFAULT_TARGET_FPS,
    preprocess_video,
)
from .types import PreprocessingError

log = structlog.get_logger(__name__)

DEFAULT_TASKS_PATH = Path("input/tasks.json")
DEFAULT_VIDEOS_DIR = Path("videos")
DEFAULT_OUTPUT_DIR = Path("output")
BENCH_SUBDIR = "processing"


def _write_report(output_dir: Path, report: dict) -> Path:
    bench_dir = output_dir / BENCH_SUBDIR
    bench_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    report_path = bench_dir / f"bench_{timestamp}.json"
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
        f.write("\n")
    return report_path


def _resolve_video_path(
    task: dict,
    videos_dir: Path,
    skip_download: bool,
) -> Path:
    task_id = task["task_id"]
    video_url = task["video_url"]

    if skip_download:
        path = expected_video_path(task_id, video_url, videos_dir)
        if not path.is_file():
            raise PreprocessingError(f"Expected video not found: {path}")
        return path

    return download_for_task(task_id, video_url, videos_dir)


def process_task(
    task: dict,
    videos_dir: Path,
    skip_download: bool,
    fps: float,
    max_dim: int,
    prune_threshold: float,
    grid_cols: int,
    grid_rows: int,
    max_frames: int | None,
) -> dict:
    task_id = task["task_id"]
    t0 = time.perf_counter()

    video_path = _resolve_video_path(task, videos_dir, skip_download)
    t1 = time.perf_counter()

    result = preprocess_video(
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

    gc.collect()
    t3 = time.perf_counter()

    meta = result.metadata
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
            "sampled": result.sampled_count,
            "post_pruned": result.post_pruned_count,
            "grids": len(result.grids_b64),
        },
        "timings_s": {
            "resolve_video": round(t1 - t0, 4),
            "preprocess": round(t2 - t1, 4),
            "cleanup": round(t3 - t2, 4),
            "total": round(t3 - t0, 4),
        },
        "status": "ok",
    }


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
) -> None:
    """Benchmark preprocessing for all tasks in a JSON file."""
    configure_logging()

    if runs < 1:
        raise ValueError("runs must be >= 1")

    tasks_path = Path(tasks)
    videos_path = Path(videos_dir)
    out_path = Path(output_dir)

    task_list = load_input(tasks_path)
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
    }

    log.info("benchmark_started", **params, num_tasks=len(task_list))

    all_runs: list[dict] = []
    for run_idx in range(runs):
        run_results: list[dict] = []
        for task in task_list:
            try:
                result = process_task(
                    task,
                    videos_path,
                    skip_download,
                    fps,
                    max_dim,
                    prune_threshold,
                    grid_cols,
                    grid_rows,
                    max_frames,
                )
            except (PreprocessingError, KeyError, FileNotFoundError) as exc:
                result = {
                    "task_id": task.get("task_id", "unknown"),
                    "video_url": task.get("video_url"),
                    "status": "failed",
                    "error": str(exc),
                }
                log.error("task_failed", task_id=result["task_id"], error=str(exc))

            run_results.append(result)
            if result.get("status") == "ok":
                log.info(
                    "task_complete",
                    run=run_idx + 1,
                    runs=runs,
                    task_id=result["task_id"],
                    total_s=result["timings_s"]["total"],
                    sampled=result["counts"]["sampled"],
                    post_pruned=result["counts"]["post_pruned"],
                    grids=result["counts"]["grids"],
                )

        all_runs.append({"run": run_idx + 1, "tasks": run_results})

    report = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "params": params,
        "runs": all_runs,
    }

    if runs == 1:
        report["tasks"] = all_runs[0]["tasks"]

    report_path = _write_report(out_path, report)
    log.info("benchmark_report_written", path=str(report_path))


def cli() -> None:
    fire.Fire(main)


if __name__ == "__main__":
    cli()
