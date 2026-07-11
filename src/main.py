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
from workflow.config import AppConfig, ModelStageConfig, load_pipeline_config

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


def _make_styles_resolver(cfg: AppConfig):
    default_styles = cfg.captions.configured_styles()

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


def _empty_captions(task: dict[str, Any], cfg: AppConfig) -> dict[str, str]:
    resolver = _make_styles_resolver(cfg)
    try:
        styles = resolver(task)
    except ValueError:
        styles = cfg.captions.configured_styles() or list_caption_styles()
    return {style: "" for style in styles}


def _write_results(results: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def _caption_params_for_style(cfg: AppConfig, style: str) -> dict[str, Any]:
    model_cfg = cfg.captions.model_for_style(style)
    return {
        "temperature": model_cfg.temperature,
        "max_tokens": model_cfg.max_tokens,
        "timeout_seconds": model_cfg.timeout_seconds,
    }


def _model_client_key(model_cfg: ModelStageConfig) -> tuple[Any, ...]:
    return (
        model_cfg.provider,
        model_cfg.model,
        model_cfg.temperature,
        model_cfg.max_tokens,
        model_cfg.timeout_seconds,
    )


def _build_caption_clients(cfg: AppConfig) -> dict[str, Any]:
    client_cache: dict[tuple[Any, ...], Any] = {}
    clients: dict[str, Any] = {}

    known_styles = set(list_caption_styles())
    configured = cfg.captions.configured_styles()
    style_names = set(configured or known_styles)
    style_names.update(known_styles)

    for style in style_names:
        model_cfg = cfg.captions.model_for_style(style)
        key = _model_client_key(model_cfg)
        if key not in client_cache:
            client_cache[key] = _create_stage_client(model_cfg)
        clients[style] = client_cache[key]

    return clients


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
    caption_clients = _build_caption_clients(cfg)
    analysis_prompt = load_prompt(cfg.vlm.prompt)

    def on_error(task: dict[str, Any], error: Exception) -> dict[str, Any]:
        task_id = task.get("task_id", "unknown")
        log.error("workflow_task_failed", task_id=task_id, error=str(error))
        return {
            "task_id": task_id,
            "captions": _empty_captions(task, cfg),
        }

    return await run_workflow_tasks(
        tasks,
        vlm_client=vlm_client,
        caption_clients=caption_clients,
        caption_params_for_style=lambda style: _caption_params_for_style(cfg, style),
        analysis_prompt=analysis_prompt,
        videos_dir=_resolve_videos_dir(cfg, project_root),
        styles_resolver=_make_styles_resolver(cfg),
        config=cfg.pipeline.pipeline_config,
        preprocess_kwargs=cfg.pipeline.preprocess_kwargs,
        skip_download=cfg.input.skip_download,
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

    for task in tasks:
        task_id = task["task_id"]
        video_url = task["video_url"]
        styles = task.get("styles", [])
        
        log.info("processing_task", task_id=task_id)

        # 1. Download Video
        video_path = download_for_task(task_id, video_url, videos_dir=videos_dir)

        # 2. Extract Grids using Pipeline 2
        extraction_res = extract_smart_grids(video_path)
        grids_b64 = extraction_res["grids_b64"]
        
        task_result = {
            "task_id": task_id,
            "video_url": video_url,
            "generations": []
        }

        # 3. Prompt VLM for each style
        for style in styles:
            log.info("generating_style", task_id=task_id, style=style)
            prompt = load_prompt(style)
            
            response = generate_from_images_base64(
                images_base64=grids_b64,
                system_prompt=prompt.system,
                user_prompt=prompt.user
            )
            
            task_result["generations"].append({
                "style": style,
                "response": response
            })
            
        results.append(task_result)

    output_path = save_output(results, output_dir)
    print(f"Wrote {len(results)} task results to {output_path}")

def cli() -> None:
    fire.Fire(main)


if __name__ == "__main__":
    cli()
