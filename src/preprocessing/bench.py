"""
Timing benchmark for the preprocessing pipeline.

Measures how long it takes to preprocess a single already-downloaded video,
broken down by phase (metadata read / frame sampling / disk save), so you
can see where the time actually goes.

Usage:
    uv run python -m preprocessing.bench --task-id v1 --video clip1.mp4
    uv run python -m preprocessing.bench --task-id v1 --video clip1.mp4 --runs 5 --no-save
    uv run python -m preprocessing.bench --task-id v1 --video clip1.mp4 --strategy uniform --max-frames 8
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

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


def run_once(video_path: Path, task_id: str, max_frames: int, max_dim: int,
             strategy: str, output_dir: Path, save: bool) -> dict:
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


def main() -> None:
    configure_logging()
    parser = argparse.ArgumentParser(description="Benchmark preprocessing timing for a single video")
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--video", required=True, help="filename inside /video, or a full path")
    parser.add_argument("--video-dir", default=str(DEFAULT_VIDEO_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--max-frames", type=int, default=DEFAULT_MAX_FRAMES)
    parser.add_argument("--max-dim", type=int, default=DEFAULT_MAX_DIM)
    parser.add_argument("--strategy", choices=["uniform", "adaptive"], default="adaptive")
    parser.add_argument("--runs", type=int, default=3,
                         help="repeat N times and report avg (run 1 may be slower - cold disk cache)")
    parser.add_argument("--no-save", action="store_true", help="skip writing frames to disk")
    args = parser.parse_args()

    try:
        video_path = resolve_video(args.video, Path(args.video_dir), task_id=args.task_id)
    except PreprocessingError as e:
        log.error("video_not_found", error=str(e))
        raise SystemExit(1)

    log.info(
        "benchmark_started",
        video=str(video_path),
        task_id=args.task_id,
        strategy=args.strategy,
        max_frames=args.max_frames,
        max_dim=args.max_dim,
        runs=args.runs,
    )
    results = []
    for i in range(args.runs):
        r = run_once(video_path, args.task_id, args.max_frames, args.max_dim,
                      args.strategy, Path(args.output_dir), not args.no_save)
        results.append(r)
        log.info(
            "benchmark_run",
            run=i + 1,
            runs=args.runs,
            total_s=round(r["total_s"], 3),
            metadata_read_s=round(r["metadata_read_s"], 3),
            sampling_s=round(r["sampling_s"], 3),
            save_s=round(r["save_s"], 3),
            num_frames=r["num_frames"],
            frame_budget=r["frame_budget"],
            duration_sec=round(r["duration_sec"], 1),
            resolution=r["resolution"],
        )

    if len(results) > 1:
        avg_total = sum(r["total_s"] for r in results) / len(results)
        avg_sampling = sum(r["sampling_s"] for r in results) / len(results)
        log.info(
            "benchmark_summary",
            avg_total_s=round(avg_total, 3),
            avg_sampling_s=round(avg_sampling, 3),
            runs=args.runs,
        )


if __name__ == "__main__":
    main()
