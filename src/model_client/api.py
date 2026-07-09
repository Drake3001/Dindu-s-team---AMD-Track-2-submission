from typing import Any

from model_client.client import ModelClient
from model_client.config import load_model_config


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
