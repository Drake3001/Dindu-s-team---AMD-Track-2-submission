from pathlib import Path
from typing import Any

import structlog

from file_io.api import configure_logging, load_input, save_output
from file_io.download import download_for_task
from model_client import (
    create_model_client,
    generate_caption,
    list_caption_styles,
    parse_json_from_model_response,
)
from model_client.prompts import load_prompt
from model_client.types import ModelRequestError, ModelResponseError
from preprocessing.api import PreprocessedVideo, PreprocessingError, preprocess

log = structlog.get_logger(__name__)

DEFAULT_ANALYSIS_PROMPT = "detailed_chronological"
DEFAULT_VIDEOS_DIR = "videos"
DEFAULT_OUTPUT_DIR = "output"


def _grid_metadata(video: PreprocessedVideo) -> list[dict[str, int]]:
    return [
        {
            "frame_count": grid.frame_count,
            "cols": grid.cols,
            "rows": grid.rows,
            "empty_cells": grid.empty_cells,
            "width_px": grid.width_px,
            "height_px": grid.height_px,
        }
        for grid in video.grids
    ]


def _task_styles(task: dict[str, Any]) -> list[str]:
    styles = task.get("styles")
    if styles is None:
        return list_caption_styles()
    if not isinstance(styles, list) or not all(isinstance(style, str) for style in styles):
        raise ValueError("task styles must be a list of strings")
    return styles


def _generate_captions(
    model_client,
    analysis: Any,
    styles: list[str],
) -> list[dict[str, str]]:
    captions = []
    for style in styles:
        caption = generate_caption(model_client, analysis, style)
        captions.append(
            {
                "style": style,
                "caption": caption,
            }
        )
    return captions


def process_task(
    task: dict[str, Any],
    *,
    model_client,
    analysis_prompt,
    videos_dir: Path,
) -> dict[str, Any]:
    task_id = task["task_id"]
    video_url = task["video_url"]
    styles = _task_styles(task)

    video_path = download_for_task(task_id, video_url, videos_dir)
    video = preprocess(task_id, video_path)

    vlm_response = model_client.generate_from_frame_grids(
        video.grids_b64,
        analysis_prompt.system,
        analysis_prompt.user,
        grids_meta=_grid_metadata(video),
    )
    analysis = parse_json_from_model_response(vlm_response)
    captions = _generate_captions(model_client, analysis, styles)

    return {
        "task_id": task_id,
        "video_url": video_url,
        "video_path": str(video_path),
        "analysis_prompt": analysis_prompt.name,
        "analysis": analysis,
        "captions": captions,
        "preprocessing": {
            "duration_sec": video.duration_sec,
            "resolution": f"{video.width}x{video.height}",
            "sampled_count": video.sampled_count,
            "post_pruned_count": video.post_pruned_count,
            "num_grids": video.num_grids,
        },
        "status": "ok",
    }


def main() -> None:
    configure_logging()

    project_root = Path(__file__).resolve().parent.parent
    input_path = project_root / "input" / "tasks.json"
    videos_dir = project_root / DEFAULT_VIDEOS_DIR
    output_dir = project_root / DEFAULT_OUTPUT_DIR

    tasks = load_input(input_path)
    model_client = create_model_client()
    analysis_prompt = load_prompt(DEFAULT_ANALYSIS_PROMPT)

    results = []
    for task in tasks:
        task_id = task.get("task_id", "unknown")
        try:
            result = process_task(
                task,
                model_client=model_client,
                analysis_prompt=analysis_prompt,
                videos_dir=videos_dir,
            )
            log.info("workflow_task_complete", task_id=task_id)
        except (
            KeyError,
            ValueError,
            FileNotFoundError,
            OSError,
            PreprocessingError,
            ModelRequestError,
            ModelResponseError,
        ) as error:
            result = {
                "task_id": task_id,
                "video_url": task.get("video_url"),
                "status": "failed",
                "error": str(error),
            }
            log.error("workflow_task_failed", task_id=task_id, error=str(error))
        results.append(result)

    output_path = save_output(results, output_dir, filename_prefix="workflow_results")
    print(f"Wrote {len(results)} workflow results to {output_path}")


if __name__ == "__main__":
    main()
