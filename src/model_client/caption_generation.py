from __future__ import annotations

from pathlib import Path
from typing import Any

from model_client.client import ModelClient
from model_client.response_parsing import format_json_for_prompt

CAPTION_PROMPTS_DIR = Path(__file__).parent / "prompts" / "caption_generation"
VIDEO_JSON_PLACEHOLDER = "[INSERT_VIDEO_JSON_HERE]"
DEFAULT_CAPTION_SYSTEM_PROMPT = "You generate styled video captions from factual JSON input."
DEFAULT_CAPTION_TEMPERATURE = 0.7
CREATIVE_CAPTION_TEMPERATURE = 1.0
CREATIVE_CAPTION_STYLES = frozenset(
    {
        "humorous_non_tech",
        "humorous_tech",
        "sarcastic",
    }
)


def list_caption_styles() -> list[str]:
    """List available caption-generation prompt styles."""
    return sorted(path.stem for path in CAPTION_PROMPTS_DIR.glob("*.md"))


def load_caption_template(style: str) -> str:
    """Load a caption-generation prompt template by style name."""
    path = CAPTION_PROMPTS_DIR / f"{style}.md"
    if not path.is_file():
        available = ", ".join(list_caption_styles()) or "(none)"
        raise FileNotFoundError(f"Unknown caption style '{style}'. Available: {available}")
    return path.read_text(encoding="utf-8")


def caption_temperature_for_style(style: str) -> float:
    """Return the default generation temperature for a caption style."""
    if style in CREATIVE_CAPTION_STYLES:
        return CREATIVE_CAPTION_TEMPERATURE
    return DEFAULT_CAPTION_TEMPERATURE


def build_caption_prompt(style: str, video_analysis: Any) -> str:
    """Insert parsed VLM JSON into a caption-generation prompt template."""
    template = load_caption_template(style)
    if VIDEO_JSON_PLACEHOLDER not in template:
        raise ValueError(
            f"Caption prompt '{style}' must contain {VIDEO_JSON_PLACEHOLDER}"
        )
    return template.replace(
        VIDEO_JSON_PLACEHOLDER,
        format_json_for_prompt(video_analysis),
    )


def generate_caption(
    model_client: ModelClient,
    video_analysis: Any,
    style: str,
    *,
    temperature: float | None = None,
    max_tokens: int | None = None,
    timeout_seconds: float | None = None,
) -> str:
    """Generate a styled caption from parsed VLM JSON."""
    return model_client.generate_text(
        DEFAULT_CAPTION_SYSTEM_PROMPT,
        build_caption_prompt(style, video_analysis),
        temperature=temperature
        if temperature is not None
        else caption_temperature_for_style(style),
        max_tokens=max_tokens,
        timeout_seconds=timeout_seconds,
    )
