import asyncio
import json
from pathlib import Path
from typing import Any

import fire
import structlog

from file_io.api import configure_logging, load_input
from model_client.api import create_async_model_client
from model_client.caption_generation import list_caption_styles
from model_client.prompts import load_prompt
from workflow.async_pipeline import run_workflow_tasks
from workflow.config import AppConfig, load_pipeline_config

log = structlog.get_logger(__name__)

CONTAINER_INPUT_PATH = Path("/input/tasks.json")
CONTAINER_OUTPUT_PATH = Path("/output/results.json")


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _resolve_tasks_path(cfg: AppConfig, project_root: Path) -> Path:
    if CONTAINER_INPUT_PATH.is_file():
        return CONTAINER_INPUT_PATH
    return cfg.input.tasks if cfg.input.tasks.is_absolute() else (project_root / cfg.input.tasks).resolve()


def _resolve_output_path(cfg: AppConfig, project_root: Path) -> Path:
    if CONTAINER_OUTPUT_PATH.parent.is_dir():
        return CONTAINER_OUTPUT_PATH
    return cfg.output.path if cfg.output.path.is_absolute() else (project_root / cfg.output.path).resolve()


def _resolve_videos_dir(cfg: AppConfig, project_root: Path) -> Path:
    if cfg.input.videos_dir.is_absolute():
        return cfg.input.videos_dir
    return (project_root / cfg.input.videos_dir).resolve()


def _make_styles_resolver(default_styles: list[str] | None):
    def _resolve_styles(task: dict[str, Any]) -> list[str]:
        styles = task.get("styles")
        if styles is not None:
            if not isinstance(styles, list) or not all(isinstance(style, str) for style in styles):
                raise ValueError("task styles must be a list of strings")
            return styles
        if default_styles is not None:
            return default_styles
        return list_caption_styles()

    return _resolve_styles


def _empty_captions(task: dict[str, Any], default_styles: list[str] | None) -> dict[str, str]:
    resolver = _make_styles_resolver(default_styles)
    try:
        styles = resolver(task)
    except ValueError:
        styles = default_styles or list_caption_styles()
    return {style: "" for style in styles}


def _write_results(results: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def _caption_params(cfg: AppConfig) -> dict[str, Any]:
    return {
        "temperature": cfg.captions.model.temperature,
        "max_tokens": cfg.captions.model.max_tokens,
        "timeout_seconds": cfg.captions.model.timeout_seconds,
    }


def _create_stage_client(model_cfg) -> Any:
    return create_async_model_client(
        provider=model_cfg.provider,
        model=model_cfg.model,
        temperature=model_cfg.temperature,
        max_tokens=model_cfg.max_tokens,
        timeout_seconds=model_cfg.timeout_seconds,
    )


async def _run_workflow(cfg: AppConfig, tasks: list[dict[str, Any]], project_root: Path) -> list[dict[str, Any]]:
    vlm_client = _create_stage_client(cfg.vlm.model)
    caption_client = _create_stage_client(cfg.captions.model)
    analysis_prompt = load_prompt(cfg.vlm.prompt)
    default_styles = cfg.captions.styles

    def on_error(task: dict[str, Any], error: Exception) -> dict[str, Any]:
        task_id = task.get("task_id", "unknown")
        log.error("workflow_task_failed", task_id=task_id, error=str(error))
        return {
            "task_id": task_id,
            "captions": _empty_captions(task, default_styles),
        }

    return await run_workflow_tasks(
        tasks,
        vlm_client=vlm_client,
        caption_client=caption_client,
        analysis_prompt=analysis_prompt,
        videos_dir=_resolve_videos_dir(cfg, project_root),
        styles_resolver=_make_styles_resolver(default_styles),
        config=cfg.pipeline.pipeline_config,
        preprocess_kwargs=cfg.pipeline.preprocess_kwargs,
        skip_download=cfg.input.skip_download,
        caption_params=_caption_params(cfg),
        on_error=on_error,
    )


def main(config: str = "config/pipeline.yaml") -> None:
    """Run the video caption workflow using settings from a YAML config file."""
    configure_logging()

    project_root = _project_root()
    cfg = load_pipeline_config(Path(config), project_root=project_root)
    tasks_path = _resolve_tasks_path(cfg, project_root)
    output_path = _resolve_output_path(cfg, project_root)

    tasks = load_input(tasks_path)
    results = asyncio.run(_run_workflow(cfg, tasks, project_root))

    _write_results(results, output_path)
    print(f"Wrote {len(results)} workflow results to {output_path}")


def cli() -> None:
    fire.Fire(main)


if __name__ == "__main__":
    cli()
