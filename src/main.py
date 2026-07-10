import json
from pathlib import Path
from typing import Any

import structlog

from file_io.api import configure_logging, load_input
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
CONTAINER_INPUT_PATH = Path("/input/tasks.json")
CONTAINER_OUTPUT_PATH = Path("/output/results.json")


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
) -> dict[str, str]:
    captions = {}
    for style in styles:
        captions[style] = generate_caption(model_client, analysis, style)
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
        "captions": captions,
    }


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _input_path(project_root: Path) -> Path:
    if CONTAINER_INPUT_PATH.is_file():
        return CONTAINER_INPUT_PATH
    return project_root / "input" / "tasks.json"


def _output_path(project_root: Path) -> Path:
    if CONTAINER_OUTPUT_PATH.parent.is_dir():
        return CONTAINER_OUTPUT_PATH
    return project_root / "output" / "results.json"


def _empty_captions(task: dict[str, Any]) -> dict[str, str]:
    try:
        styles = _task_styles(task)
    except ValueError:
        styles = list_caption_styles()
    return {style: "" for style in styles}


def _write_results(results: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
        f.write("\n")


def main() -> None:
    configure_logging()

    project_root = _project_root()
    input_path = _input_path(project_root)
    videos_dir = project_root / DEFAULT_VIDEOS_DIR
    output_path = _output_path(project_root)

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
                "captions": _empty_captions(task),
            }
            log.error("workflow_task_failed", task_id=task_id, error=str(error))
        results.append(result)

    _write_results(results, output_path)
    print(f"Wrote {len(results)} workflow results to {output_path}")


if __name__ == "__main__":
    main()
