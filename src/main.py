import asyncio
import json
from pathlib import Path
from typing import Any

import structlog

from file_io.api import configure_logging, load_input
from model_client.api import create_async_model_client
from model_client.caption_generation import list_caption_styles
from model_client.prompts import load_prompt
from model_client.types import ModelRequestError, ModelResponseError
from preprocessing.types import PreprocessingError
from workflow.async_pipeline import PipelineConfig, run_workflow_tasks

log = structlog.get_logger(__name__)

DEFAULT_ANALYSIS_PROMPT = "detailed_chronological"
DEFAULT_VIDEOS_DIR = "videos"
CONTAINER_INPUT_PATH = Path("/input/tasks.json")
CONTAINER_OUTPUT_PATH = Path("/output/results.json")


def _task_styles(task: dict[str, Any]) -> list[str]:
    styles = task.get("styles")
    if styles is None:
        return list_caption_styles()
    if not isinstance(styles, list) or not all(isinstance(style, str) for style in styles):
        raise ValueError("task styles must be a list of strings")
    return styles


def _empty_captions(task: dict[str, Any]) -> dict[str, str]:
    try:
        styles = _task_styles(task)
    except ValueError:
        styles = list_caption_styles()
    return {style: "" for style in styles}


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


def _write_results(results: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
        f.write("\n")


def _handle_task_error(task: dict[str, Any], error: Exception) -> dict[str, Any]:
    task_id = task.get("task_id", "unknown")
    log.error("workflow_task_failed", task_id=task_id, error=str(error))
    return {
        "task_id": task_id,
        "captions": _empty_captions(task),
    }


async def _run_workflow(
    tasks: list[dict[str, Any]],
    *,
    videos_dir: Path,
    pipeline_config: PipelineConfig | None = None,
) -> list[dict[str, Any]]:
    model_client = create_async_model_client()
    analysis_prompt = load_prompt(DEFAULT_ANALYSIS_PROMPT)

    return await run_workflow_tasks(
        tasks,
        model_client=model_client,
        analysis_prompt=analysis_prompt,
        videos_dir=videos_dir,
        styles_resolver=_task_styles,
        config=pipeline_config,
        on_error=_handle_task_error,
    )


def main() -> None:
    configure_logging()

    project_root = _project_root()
    input_path = _input_path(project_root)
    videos_dir = project_root / DEFAULT_VIDEOS_DIR
    output_path = _output_path(project_root)

    tasks = load_input(input_path)
    results = asyncio.run(_run_workflow(tasks, videos_dir=videos_dir))

    _write_results(results, output_path)
    print(f"Wrote {len(results)} workflow results to {output_path}")


if __name__ == "__main__":
    main()
