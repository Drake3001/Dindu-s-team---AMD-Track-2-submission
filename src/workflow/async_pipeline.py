from __future__ import annotations

import asyncio
import os
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import structlog

from file_io.download import download_for_task, expected_video_path
from model_client.caption_generation import async_generate_caption
from model_client.client import AsyncModelClient
from model_client.prompts import Prompt
from model_client.response_parsing import parse_json_from_model_response
from model_client.types import ModelRequestError
from pipeline_2.grid_extractor import extract_smart_grids
from preprocessing.preprocessing import preprocess_video
from preprocessing.types import PreprocessingError

log = structlog.get_logger(__name__)


@dataclass(frozen=True)
class PipelineConfig:
    max_download_workers: int = 2
    max_preprocess_workers: int | None = None
    max_inference_workers: int = 3


def default_preprocess_workers(num_tasks: int) -> int:
    return min(num_tasks, os.cpu_count() or 2)


def parallelism_params(config: PipelineConfig, num_tasks: int) -> dict[str, int]:
    preprocess_workers = config.max_preprocess_workers
    if preprocess_workers is None:
        preprocess_workers = default_preprocess_workers(num_tasks)
    return {
        "max_download_workers": config.max_download_workers,
        "max_preprocess_workers": preprocess_workers,
        "max_inference_workers": config.max_inference_workers,
    }


def _resolve_video_path_sync(
    task_id: str,
    video_url: str,
    videos_dir: Path,
    skip_download: bool,
) -> str:
    if skip_download:
        path = expected_video_path(task_id, video_url, videos_dir)
        if not path.is_file():
            raise PreprocessingError(f"Expected video not found: {path}")
        return str(path)
    return str(download_for_task(task_id, video_url, videos_dir))


def _preprocess_worker(args: tuple[str, str, dict[str, Any]]) -> dict[str, Any]:
    task_id, video_path, preprocess_kwargs = args
    kwargs = dict(preprocess_kwargs)
    strategy = kwargs.pop("strategy", "smart")
    upload_mode = kwargs.pop("upload_mode", "grid")

    if strategy == "smart":
        result = extract_smart_grids(
            video_path,
            max_frames=kwargs.get("max_frames", 32),
            context_frames=kwargs.get("context_frames", 4),
            max_dim=kwargs.get("max_dim", 512),
            grid_cols=kwargs.get("grid_cols", 4),
            grid_rows=kwargs.get("grid_rows", 4),
            upload_mode=upload_mode,
        )
        meta = result["metadata"]
        return _build_preprocess_payload(
            task_id=task_id,
            video_path=str(video_path),
            metadata=meta,
            upload_mode=upload_mode,
            grids=result.get("grids_b64") or [],
            frames_b64=result.get("frames_b64"),
            sampled_count=result["sampled_count"],
            post_pruned_count=result["sampled_count"],
            frame_timestamps=result["frame_timestamps"],
        )

    result = preprocess_video(
        task_id=task_id,
        video_path=Path(video_path),
        upload_mode=upload_mode,
        **kwargs,
    )
    meta = result.metadata
    return _build_preprocess_payload(
        task_id=task_id,
        video_path=str(video_path),
        metadata=meta,
        upload_mode=upload_mode,
        grids=result.grids,
        frames_b64=result.frames_b64,
        sampled_count=result.sampled_count,
        post_pruned_count=result.post_pruned_count,
        frame_timestamps=result.frame_timestamps,
    )


def _build_preprocess_payload(
    *,
    task_id: str,
    video_path: str,
    metadata: Any,
    upload_mode: str,
    grids: list[Any],
    frames_b64: list[str] | None,
    sampled_count: int,
    post_pruned_count: int,
    frame_timestamps: list[float],
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "task_id": task_id,
        "video_path": video_path,
        "metadata": {
            "duration_sec": metadata.duration_sec,
            "fps": metadata.fps,
            "frame_count": metadata.frame_count,
            "width": metadata.width,
            "height": metadata.height,
        },
        "upload_mode": upload_mode,
        "sampled_count": sampled_count,
        "post_pruned_count": post_pruned_count,
        "frame_timestamps": frame_timestamps,
    }

    if upload_mode == "frames":
        payload["frames_b64"] = frames_b64 or []
        payload["frames_count"] = len(frames_b64 or [])
        payload["grids_b64"] = []
        payload["grids_meta"] = []
        payload["grids_count"] = 0
        return payload

    grid_images = grids
    if grid_images and hasattr(grid_images[0], "b64"):
        payload["grids_b64"] = [grid.b64 for grid in grid_images]
        payload["grids_meta"] = [
            {
                "frame_count": grid.frame_count,
                "cols": grid.cols,
                "rows": grid.rows,
                "empty_cells": grid.empty_cells,
                "width_px": grid.width_px,
                "height_px": grid.height_px,
            }
            for grid in grid_images
        ]
        payload["grids_count"] = len(grid_images)
    else:
        payload["grids_b64"] = []
        payload["grids_meta"] = []
        payload["grids_count"] = 0

    payload["frames_b64"] = []
    payload["frames_count"] = 0
    return payload


async def _download_video(
    task: dict[str, Any],
    videos_dir: Path,
    skip_download: bool,
    download_sem: asyncio.Semaphore,
    loop: asyncio.AbstractEventLoop,
) -> str:
    async with download_sem:
        return await loop.run_in_executor(
            None,
            _resolve_video_path_sync,
            task["task_id"],
            task["video_url"],
            videos_dir,
            skip_download,
        )


async def _preprocess_video(
    task_id: str,
    video_path: str,
    preprocess_kwargs: dict[str, Any],
    process_pool: ProcessPoolExecutor,
    loop: asyncio.AbstractEventLoop,
) -> dict[str, Any]:
    return await loop.run_in_executor(
        process_pool,
        _preprocess_worker,
        (task_id, video_path, preprocess_kwargs),
    )


async def run_workflow_task(
    task: dict[str, Any],
    *,
    vlm_client: AsyncModelClient,
    caption_clients: dict[str, AsyncModelClient],
    caption_params_for_style: Any,
    analysis_prompt: Prompt,
    videos_dir: Path,
    styles: list[str],
    download_sem: asyncio.Semaphore,
    inference_sem: asyncio.Semaphore,
    process_pool: ProcessPoolExecutor,
    loop: asyncio.AbstractEventLoop,
    preprocess_kwargs: dict[str, Any] | None = None,
    skip_download: bool = False,
) -> dict[str, Any]:
    task_id = task["task_id"]
    t0 = time.perf_counter()

    video_path = await _download_video(
        task, videos_dir, skip_download, download_sem, loop
    )
    t1 = time.perf_counter()

    preprocessed = await _preprocess_video(
        task_id,
        video_path,
        preprocess_kwargs or {},
        process_pool,
        loop,
    )
    t2 = time.perf_counter()

    upload_mode = preprocessed.get("upload_mode", "grid")

    async with inference_sem:
        if upload_mode == "frames":
            vlm_response = await vlm_client.generate_from_individual_frames(
                preprocessed["frames_b64"],
                analysis_prompt.system,
                analysis_prompt.user,
                frame_timestamps=preprocessed["frame_timestamps"],
            )
        else:
            vlm_response = await vlm_client.generate_from_frame_grids(
                preprocessed["grids_b64"],
                analysis_prompt.system,
                analysis_prompt.user,
                grids_meta=preprocessed["grids_meta"],
            )
    analysis = parse_json_from_model_response(vlm_response)
    t3 = time.perf_counter()

    async def _caption(style: str) -> tuple[str, str]:
        client = caption_clients.get(style)
        if client is None:
            raise ModelRequestError(f"No caption client configured for style '{style}'")
        params = caption_params_for_style(style)
        async with inference_sem:
            text = await async_generate_caption(
                client,
                analysis,
                style,
                temperature=params.get("temperature"),
                max_tokens=params.get("max_tokens"),
                timeout_seconds=params.get("timeout_seconds"),
            )
        return style, text

    caption_pairs = await asyncio.gather(*[_caption(style) for style in styles])
    captions = dict(caption_pairs)
    t4 = time.perf_counter()

    key_seconds = sorted(round(float(t), 1) for t in preprocessed.get("frame_timestamps", []))

    log.info(
        "workflow_task_complete",
        task_id=task_id,
        resolve_video_s=round(t1 - t0, 4),
        preprocess_s=round(t2 - t1, 4),
        vlm_s=round(t3 - t2, 4),
        captions_s=round(t4 - t3, 4),
        total_s=round(t4 - t0, 4),
        key_seconds=key_seconds,
        key_frame_count=len(key_seconds),
    )

    return {"task_id": task_id, "captions": captions}


async def run_bench_task(
    task: dict[str, Any],
    *,
    model_client: AsyncModelClient,
    videos_dir: Path,
    skip_download: bool,
    prompts: list[Prompt],
    include_responses: bool,
    download_sem: asyncio.Semaphore,
    inference_sem: asyncio.Semaphore,
    process_pool: ProcessPoolExecutor,
    loop: asyncio.AbstractEventLoop,
    preprocess_kwargs: dict[str, Any],
) -> dict[str, Any]:
    from model_client.bench import _failed_output_record, _model_info, _output_record, _task_status_from_outputs

    task_id = task["task_id"]
    t0 = time.perf_counter()

    video_path = await _download_video(task, videos_dir, skip_download, download_sem, loop)
    t1 = time.perf_counter()

    preprocessed = await _preprocess_video(
        task_id,
        video_path,
        preprocess_kwargs,
        process_pool,
        loop,
    )
    t2 = time.perf_counter()

    grids_b64 = preprocessed["grids_b64"]
    grids_meta = preprocessed["grids_meta"]
    grids_sent = len(grids_b64)
    meta = preprocessed["metadata"]

    outputs = []
    for prompt in prompts:
        call_start = time.perf_counter()
        if not grids_b64:
            elapsed_s = round(time.perf_counter() - call_start, 4)
            error = ModelRequestError("No grid images available for model request")
            outputs.append(_failed_output_record(prompt.name, elapsed_s, error, grids_sent))
            log.error(
                "model_grid_request_failed",
                task_id=task_id,
                prompt=prompt.name,
                error=str(error),
            )
            continue

        try:
            async with inference_sem:
                response = await model_client.generate_from_frame_grids(
                    grids_b64,
                    prompt.system,
                    prompt.user,
                    grids_meta=grids_meta,
                )
        except ModelRequestError as exc:
            elapsed_s = round(time.perf_counter() - call_start, 4)
            outputs.append(_failed_output_record(prompt.name, elapsed_s, exc, grids_sent))
            log.error(
                "model_grid_request_failed",
                task_id=task_id,
                prompt=prompt.name,
                error=str(exc),
            )
            continue

        elapsed_s = round(time.perf_counter() - call_start, 4)
        outputs.append(
            _output_record(prompt.name, response, elapsed_s, include_responses, grids_sent)
        )
    t3 = time.perf_counter()
    t4 = time.perf_counter()

    return {
        "task_id": task_id,
        "video_url": task["video_url"],
        "video_path": video_path,
        "metadata": {
            "duration_sec": meta["duration_sec"],
            "fps": meta["fps"],
            "frame_count": meta["frame_count"],
            "width": meta["width"],
            "height": meta["height"],
            "resolution": f"{meta['width']}x{meta['height']}",
        },
        "counts": {
            "sampled": preprocessed["sampled_count"],
            "post_pruned": preprocessed["post_pruned_count"],
            "grids": preprocessed["grids_count"],
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


async def run_workflow_tasks(
    tasks: list[dict[str, Any]],
    *,
    vlm_client: AsyncModelClient,
    caption_clients: dict[str, AsyncModelClient],
    caption_params_for_style: Any,
    analysis_prompt: Prompt,
    videos_dir: Path,
    styles_resolver: Any,
    config: PipelineConfig | None = None,
    preprocess_kwargs: dict[str, Any] | None = None,
    skip_download: bool = False,
    on_error: Any | None = None,
) -> list[dict[str, Any]]:
    cfg = config or PipelineConfig()
    loop = asyncio.get_running_loop()
    download_sem = asyncio.Semaphore(cfg.max_download_workers)
    inference_sem = asyncio.Semaphore(cfg.max_inference_workers)
    preprocess_workers = cfg.max_preprocess_workers or default_preprocess_workers(len(tasks))

    results: list[dict[str, Any] | None] = [None] * len(tasks)

    with ProcessPoolExecutor(max_workers=preprocess_workers) as process_pool:

        async def _run_one(index: int, task: dict[str, Any]) -> None:
            try:
                styles = styles_resolver(task)
                results[index] = await run_workflow_task(
                    task,
                    vlm_client=vlm_client,
                    caption_clients=caption_clients,
                    caption_params_for_style=caption_params_for_style,
                    analysis_prompt=analysis_prompt,
                    videos_dir=videos_dir,
                    styles=styles,
                    download_sem=download_sem,
                    inference_sem=inference_sem,
                    process_pool=process_pool,
                    loop=loop,
                    preprocess_kwargs=preprocess_kwargs,
                    skip_download=skip_download,
                )
            except Exception as error:
                if on_error is not None:
                    results[index] = on_error(task, error)
                else:
                    raise

        await asyncio.gather(*[_run_one(i, task) for i, task in enumerate(tasks)])

    return [r for r in results if r is not None]


async def run_bench_tasks(
    tasks: list[dict[str, Any]],
    *,
    model_client: AsyncModelClient,
    videos_dir: Path,
    skip_download: bool,
    prompts: list[Prompt],
    include_responses: bool,
    preprocess_kwargs: dict[str, Any],
    config: PipelineConfig | None = None,
    on_error: Any | None = None,
) -> list[dict[str, Any]]:
    cfg = config or PipelineConfig()
    loop = asyncio.get_running_loop()
    download_sem = asyncio.Semaphore(cfg.max_download_workers)
    inference_sem = asyncio.Semaphore(cfg.max_inference_workers)
    preprocess_workers = cfg.max_preprocess_workers or default_preprocess_workers(len(tasks))

    results: list[dict[str, Any] | None] = [None] * len(tasks)

    with ProcessPoolExecutor(max_workers=preprocess_workers) as process_pool:

        async def _run_one(index: int, task: dict[str, Any]) -> None:
            try:
                results[index] = await run_bench_task(
                    task,
                    model_client=model_client,
                    videos_dir=videos_dir,
                    skip_download=skip_download,
                    prompts=prompts,
                    include_responses=include_responses,
                    download_sem=download_sem,
                    inference_sem=inference_sem,
                    process_pool=process_pool,
                    loop=loop,
                    preprocess_kwargs=preprocess_kwargs,
                )
            except Exception as error:
                if on_error is not None:
                    results[index] = on_error(task, error)
                else:
                    raise

        await asyncio.gather(*[_run_one(i, task) for i, task in enumerate(tasks)])

    return [r for r in results if r is not None]
