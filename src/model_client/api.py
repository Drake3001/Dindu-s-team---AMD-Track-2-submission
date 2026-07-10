from typing import Any

from model_client.client import ModelClient
from model_client.config import load_model_config
from model_client.messages import DEFAULT_IMAGE_MIME_TYPE


def create_model_client(
    provider: str | None = None,
    api_key: str | None = None,
    model: str | None = None,
    base_url: str | None = None,
    timeout_seconds: float | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
) -> ModelClient:
    """Create a model client from explicit overrides and environment config."""
    config = load_model_config(
        provider=provider,
        api_key=api_key,
        model=model,
        base_url=base_url,
        timeout_seconds=timeout_seconds,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return ModelClient(config)


def chat(
    messages: list[dict[str, Any]],
    *,
    provider: str | None = None,
    api_key: str | None = None,
    model: str | None = None,
    base_url: str | None = None,
    timeout_seconds: float | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
) -> str:
    """Send OpenAI-compatible chat messages to the configured provider."""
    client = create_model_client(
        provider=provider,
        api_key=api_key,
        model=model,
        base_url=base_url,
        timeout_seconds=timeout_seconds,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return client.chat(messages)


def generate_text(
    system_prompt: str,
    user_prompt: str,
    *,
    provider: str | None = None,
    api_key: str | None = None,
    model: str | None = None,
    base_url: str | None = None,
    timeout_seconds: float | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
) -> str:
    """Generate text from a system prompt and a user prompt."""
    client = create_model_client(
        provider=provider,
        api_key=api_key,
        model=model,
        base_url=base_url,
        timeout_seconds=timeout_seconds,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return client.generate_text(system_prompt, user_prompt)


def generate_from_image_base64(
    image_base64: str,
    system_prompt: str,
    user_prompt: str,
    *,
    image_mime_type: str = DEFAULT_IMAGE_MIME_TYPE,
    provider: str | None = None,
    api_key: str | None = None,
    model: str | None = None,
    base_url: str | None = None,
    timeout_seconds: float | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
) -> str:
    """Generate text from a prompt and one base64-encoded image."""
    client = create_model_client(
        provider=provider,
        api_key=api_key,
        model=model,
        base_url=base_url,
        timeout_seconds=timeout_seconds,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return client.generate_from_image_base64(
        image_base64,
        system_prompt,
        user_prompt,
        image_mime_type=image_mime_type,
    )


def generate_from_images_base64(
    images_base64: list[str],
    system_prompt: str,
    user_prompt: str,
    *,
    image_mime_type: str = DEFAULT_IMAGE_MIME_TYPE,
    provider: str | None = None,
    api_key: str | None = None,
    model: str | None = None,
    base_url: str | None = None,
    timeout_seconds: float | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
) -> str:
    """Generate text from a prompt and one or more base64-encoded images."""
    client = create_model_client(
        provider=provider,
        api_key=api_key,
        model=model,
        base_url=base_url,
        timeout_seconds=timeout_seconds,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return client.generate_from_images_base64(
        images_base64,
        system_prompt,
        user_prompt,
        image_mime_type=image_mime_type,
    )


def generate_from_frame_grid_base64(
    frame_grid_base64: str,
    system_prompt: str,
    user_prompt: str,
    *,
    frame_count: int,
    cols: int,
    rows: int,
    empty_cells: int,
    width_px: int,
    height_px: int,
    image_mime_type: str = DEFAULT_IMAGE_MIME_TYPE,
    provider: str | None = None,
    api_key: str | None = None,
    model: str | None = None,
    base_url: str | None = None,
    timeout_seconds: float | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
) -> str:
    """Generate text from one base64 image containing a grid of video frames."""
    client = create_model_client(
        provider=provider,
        api_key=api_key,
        model=model,
        base_url=base_url,
        timeout_seconds=timeout_seconds,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return client.generate_from_frame_grid_base64(
        frame_grid_base64,
        system_prompt,
        user_prompt,
        frame_count=frame_count,
        cols=cols,
        rows=rows,
        empty_cells=empty_cells,
        width_px=width_px,
        height_px=height_px,
        image_mime_type=image_mime_type,
    )
