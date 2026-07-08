"""
Timing benchmark for the preprocessing pipeline.

Measures how long it takes to preprocess a single already-downloaded video,
broken down by phase (metadata read / frame sampling / disk save), so you
can see where the time actually goes.

Usage:
    uv run bench --task_id v1 --video clip1.mp4
    uv run bench --task_id v1 --video clip1.mp4 --runs 5 --save=False
    uv run bench --task_id v1 --video clip1.mp4 --strategy uniform --max_frames 8
"""

from __future__ import annotations

import time
from pathlib import Path

import fire
import structlog

from file_io.api import configure_logging

from .preprocessing import (
    DEFAULT_MAX_DIM,
    DEFAULT_MAX_FRAMES,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_VIDEO_DIR,
    PreprocessingError,
    compute_frame_budget,
    read_metadata,
    resolve_video,
    sample_frames_adaptive,
    sample_frames_uniform,
    save_frames,
)

log = structlog.get_logger(__name__)


def run_once(
    video_path: Path,
    task_id: str,
    max_frames: int,
    max_dim: int,
    strategy: str,
    output_dir: Path,
    save: bool,
) -> dict:
    t0 = time.perf_counter()
    metadata = read_metadata(video_path)
    t1 = time.perf_counter()

    frame_budget = compute_frame_budget(metadata.duration_sec, max_frames=max_frames)
    sampler = sample_frames_uniform if strategy == "uniform" else sample_frames_adaptive
    frames = sampler(video_path, metadata, frame_budget, max_dim)
    t2 = time.perf_counter()

    if save:
        save_frames(frames, output_dir / task_id, task_id)
    t3 = time.perf_counter()

    return {
        "duration_sec": metadata.duration_sec,
        "resolution": f"{metadata.width}x{metadata.height}",
        "frame_budget": frame_budget,
        "num_frames": len(frames),
        "metadata_read_s": t1 - t0,
        "sampling_s": t2 - t1,
        "save_s": t3 - t2,
        "total_s": t3 - t0,
    }


def main(
    task_id: str,
    video: str,
    video_dir: str = str(DEFAULT_VIDEO_DIR),
    output_dir: str = str(DEFAULT_OUTPUT_DIR),
    max_frames: int = DEFAULT_MAX_FRAMES,
    max_dim: int = DEFAULT_MAX_DIM,
    strategy: str = "adaptive",
    runs: int = 3,
    save: bool = True,
) -> None:
    """Benchmark preprocessing timing for a single video."""
    configure_logging()

    if strategy not in {"uniform", "adaptive"}:
        raise ValueError("strategy must be 'uniform' or 'adaptive'")

    try:
        video_path = resolve_video(video, Path(video_dir), task_id=task_id)
    except PreprocessingError as e:
        log.error("video_not_found", error=str(e))
        raise SystemExit(1)

    log.info(
        "benchmark_started",
        video=str(video_path),
        task_id=task_id,
        strategy=strategy,
        max_frames=max_frames,
        max_dim=max_dim,
        runs=runs,
    )
    results = []
    for i in range(runs):
        result = run_once(
            video_path,
            task_id,
            max_frames,
            max_dim,
            strategy,
            Path(output_dir),
            save,
        )
        results.append(result)
        log.info(
            "benchmark_run",
            run=i + 1,
            runs=runs,
            total_s=round(result["total_s"], 3),
            metadata_read_s=round(result["metadata_read_s"], 3),
            sampling_s=round(result["sampling_s"], 3),
            save_s=round(result["save_s"], 3),
            num_frames=result["num_frames"],
            frame_budget=result["frame_budget"],
            duration_sec=round(result["duration_sec"], 1),
            resolution=result["resolution"],
        )

    if len(results) > 1:
        avg_total = sum(r["total_s"] for r in results) / len(results)
        avg_sampling = sum(r["sampling_s"] for r in results) / len(results)
        log.info(
            "benchmark_summary",
            avg_total_s=round(avg_total, 3),
            avg_sampling_s=round(avg_sampling, 3),
            runs=runs,
        )


def cli() -> None:
    fire.Fire(main)


if __name__ == "__main__":
    cli()
