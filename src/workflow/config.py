from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from model_client.caption_generation import list_caption_styles
from model_client.config import (
    DEFAULT_MAX_TOKENS,
    DEFAULT_TEMPERATURE,
    DEFAULT_TIMEOUT_SECONDS,
)
from model_client.prompts import load_prompt, list_prompt_names
from preprocessing.preprocessing import (
    DEFAULT_GRID_COLS,
    DEFAULT_GRID_ROWS,
    DEFAULT_MAX_DIM,
    DEFAULT_MAX_FRAMES,
    DEFAULT_PRUNE_THRESHOLD,
    DEFAULT_TARGET_FPS,
)
from workflow.async_pipeline import PipelineConfig

CONTAINER_CONFIG_PATH = Path("/config/pipeline.yaml")

DEFAULT_CONFIG: dict[str, Any] = {
    "input": {
        "tasks": "input/tasks.json",
        "videos_dir": "videos",
        "skip_download": False,
    },
    "output": {
        "path": "output/results.json",
    },
    "pipeline": {
        "fps": DEFAULT_TARGET_FPS,
        "max_dim": DEFAULT_MAX_DIM,
        "prune_threshold": DEFAULT_PRUNE_THRESHOLD,
        "grid_cols": DEFAULT_GRID_COLS,
        "grid_rows": DEFAULT_GRID_ROWS,
        "max_frames": DEFAULT_MAX_FRAMES,
        "concurrency": {
            "max_download_workers": 2,
            "max_preprocess_workers": None,
            "max_inference_workers": 3,
        },
    },
    "vlm": {
        "provider": None,
        "model": None,
        "temperature": 0.0,
        "max_tokens": 1024,
        "timeout_seconds": DEFAULT_TIMEOUT_SECONDS,
        "prompt": "detailed_chronological",
    },
    "captions": {
        "provider": None,
        "model": None,
        "temperature": None,
        "max_tokens": DEFAULT_MAX_TOKENS,
        "timeout_seconds": DEFAULT_TIMEOUT_SECONDS,
        "styles": None,
    },
}


class ConfigError(ValueError):
    """Raised when pipeline YAML configuration is invalid."""


@dataclass(frozen=True)
class InputConfig:
    tasks: Path
    videos_dir: Path
    skip_download: bool


@dataclass(frozen=True)
class OutputConfig:
    path: Path


@dataclass(frozen=True)
class ModelStageConfig:
    provider: str | None
    model: str | None
    temperature: float | None
    max_tokens: int | None
    timeout_seconds: float | None


@dataclass(frozen=True)
class PipelineStageConfig:
    fps: float
    max_dim: int
    prune_threshold: float
    grid_cols: int
    grid_rows: int
    max_frames: int | None
    concurrency: PipelineConfig

    @property
    def preprocess_kwargs(self) -> dict[str, Any]:
        return {
            "fps": self.fps,
            "max_dim": self.max_dim,
            "prune_threshold": self.prune_threshold,
            "grid_cols": self.grid_cols,
            "grid_rows": self.grid_rows,
            "max_frames": self.max_frames,
        }

    @property
    def pipeline_config(self) -> PipelineConfig:
        return self.concurrency


@dataclass(frozen=True)
class VlmConfig:
    model: ModelStageConfig
    prompt: str


@dataclass(frozen=True)
class CaptionConfig:
    model: ModelStageConfig
    styles: list[str] | None


@dataclass(frozen=True)
class AppConfig:
    input: InputConfig
    output: OutputConfig
    pipeline: PipelineStageConfig
    vlm: VlmConfig
    captions: CaptionConfig


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _require_mapping(data: Any, section: str) -> dict[str, Any]:
    if not isinstance(data, dict):
        raise ConfigError(f"Config section '{section}' must be a mapping")
    return data


def _read_optional_str(value: Any, field: str, section: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ConfigError(f"{section}.{field} must be a string or null")
    stripped = value.strip()
    return stripped or None


def _read_float(value: Any, field: str, section: str) -> float:
    if not isinstance(value, (int, float)):
        raise ConfigError(f"{section}.{field} must be a number")
    return float(value)


def _read_optional_float(value: Any, field: str, section: str) -> float | None:
    if value is None:
        return None
    return _read_float(value, field, section)


def _read_int(value: Any, field: str, section: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ConfigError(f"{section}.{field} must be an integer")
    return value


def _read_optional_int(value: Any, field: str, section: str) -> int | None:
    if value is None:
        return None
    return _read_int(value, field, section)


def _read_bool(value: Any, field: str, section: str) -> bool:
    if not isinstance(value, bool):
        raise ConfigError(f"{section}.{field} must be a boolean")
    return value


def _read_path(value: Any, field: str, section: str, base_dir: Path) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{section}.{field} must be a non-empty string path")
    path = Path(value.strip())
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


def _parse_model_stage(data: dict[str, Any], section: str) -> ModelStageConfig:
    return ModelStageConfig(
        provider=_read_optional_str(data.get("provider"), "provider", section),
        model=_read_optional_str(data.get("model"), "model", section),
        temperature=_read_optional_float(data.get("temperature"), "temperature", section),
        max_tokens=_read_optional_int(data.get("max_tokens"), "max_tokens", section),
        timeout_seconds=_read_optional_float(
            data.get("timeout_seconds"), "timeout_seconds", section
        ),
    )


def _parse_styles(value: Any) -> list[str] | None:
    if value is None:
        return None
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ConfigError("captions.styles must be a list of strings or null")
    return [item.strip() for item in value if item.strip()]


def _validate_prompt(prompt_name: str) -> None:
    try:
        load_prompt(prompt_name)
    except FileNotFoundError as error:
        available = ", ".join(list_prompt_names()) or "(none)"
        raise ConfigError(
            f"Unknown VLM prompt '{prompt_name}'. Available: {available}"
        ) from error


def _validate_styles(styles: list[str] | None) -> None:
    if styles is None:
        return
    available = set(list_caption_styles())
    unknown = [style for style in styles if style not in available]
    if unknown:
        available_list = ", ".join(sorted(available)) or "(none)"
        raise ConfigError(
            f"Unknown caption style(s): {', '.join(unknown)}. Available: {available_list}"
        )


def resolve_config_path(path: Path, project_root: Path) -> Path:
    candidates = [
        path,
        project_root / path,
        CONTAINER_CONFIG_PATH,
        project_root / "config" / "pipeline.yaml",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise ConfigError(f"Pipeline config not found: {path}")


def load_pipeline_config(path: Path, *, project_root: Path | None = None) -> AppConfig:
    """Load and validate pipeline configuration from YAML."""
    root = project_root or Path.cwd()
    config_path = resolve_config_path(path, root)

    with config_path.open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)

    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ConfigError("Pipeline config root must be a mapping")

    allowed_keys = set(DEFAULT_CONFIG)
    unknown_keys = sorted(set(raw) - allowed_keys)
    if unknown_keys:
        raise ConfigError(f"Unknown top-level config keys: {', '.join(unknown_keys)}")

    merged = _deep_merge(DEFAULT_CONFIG, raw)
    base_dir = root.resolve()

    input_data = _require_mapping(merged.get("input"), "input")
    output_data = _require_mapping(merged.get("output"), "output")
    pipeline_data = _require_mapping(merged.get("pipeline"), "pipeline")
    concurrency_data = _require_mapping(pipeline_data.get("concurrency"), "pipeline.concurrency")
    vlm_data = _require_mapping(merged.get("vlm"), "vlm")
    captions_data = _require_mapping(merged.get("captions"), "captions")

    vlm_prompt = _read_optional_str(vlm_data.get("prompt"), "prompt", "vlm")
    if not vlm_prompt:
        raise ConfigError("vlm.prompt must be a non-empty string")
    _validate_prompt(vlm_prompt)

    caption_styles = _parse_styles(captions_data.get("styles"))
    _validate_styles(caption_styles)

    return AppConfig(
        input=InputConfig(
            tasks=_read_path(input_data.get("tasks"), "tasks", "input", base_dir),
            videos_dir=_read_path(
                input_data.get("videos_dir"), "videos_dir", "input", base_dir
            ),
            skip_download=_read_bool(
                input_data.get("skip_download"), "skip_download", "input"
            ),
        ),
        output=OutputConfig(
            path=_read_path(output_data.get("path"), "path", "output", base_dir),
        ),
        pipeline=PipelineStageConfig(
            fps=_read_float(pipeline_data.get("fps"), "fps", "pipeline"),
            max_dim=_read_int(pipeline_data.get("max_dim"), "max_dim", "pipeline"),
            prune_threshold=_read_float(
                pipeline_data.get("prune_threshold"), "prune_threshold", "pipeline"
            ),
            grid_cols=_read_int(pipeline_data.get("grid_cols"), "grid_cols", "pipeline"),
            grid_rows=_read_int(pipeline_data.get("grid_rows"), "grid_rows", "pipeline"),
            max_frames=_read_optional_int(
                pipeline_data.get("max_frames"), "max_frames", "pipeline"
            ),
            concurrency=PipelineConfig(
                max_download_workers=_read_int(
                    concurrency_data.get("max_download_workers"),
                    "max_download_workers",
                    "pipeline.concurrency",
                ),
                max_preprocess_workers=_read_optional_int(
                    concurrency_data.get("max_preprocess_workers"),
                    "max_preprocess_workers",
                    "pipeline.concurrency",
                ),
                max_inference_workers=_read_int(
                    concurrency_data.get("max_inference_workers"),
                    "max_inference_workers",
                    "pipeline.concurrency",
                ),
            ),
        ),
        vlm=VlmConfig(
            model=_parse_model_stage(vlm_data, "vlm"),
            prompt=vlm_prompt,
        ),
        captions=CaptionConfig(
            model=_parse_model_stage(captions_data, "captions"),
            styles=caption_styles,
        ),
    )
