from typing import Any

from openai import OpenAI, OpenAIError
import structlog

from model_client.config import ModelConfig, load_model_config

log = structlog.get_logger(__name__)


class ModelRequestError(RuntimeError):
    """Raised when a model API request fails."""


class ModelResponseError(RuntimeError):
    """Raised when a model API response has an unexpected shape."""


class ModelClient:
    """Small wrapper around the OpenAI SDK for configured model providers."""

    def __init__(self, config: ModelConfig) -> None:
        self.config = config
        self._client = OpenAI(
            api_key=config.api_key,
            base_url=config.base_url,
            timeout=config.timeout_seconds,
        )

    @property
    def chat_completions_url(self) -> str:
        return f"{self.config.base_url}/chat/completions"

    def chat(
        self,
        messages: list[dict[str, Any]],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        timeout_seconds: float | None = None,
    ) -> str:
        """Send chat messages to the configured model and return the assistant text."""
        if not messages:
            raise ModelRequestError("messages must contain at least one message")

        resolved_temperature = self.config.temperature if temperature is None else temperature
        resolved_max_tokens = self.config.max_tokens if max_tokens is None else max_tokens
        timeout = self.config.timeout_seconds if timeout_seconds is None else timeout_seconds

        log.info(
            "model_request_started",
            provider=self.config.provider,
            model=self.config.model,
            url=self.chat_completions_url,
        )

        try:
            response = self._client.with_options(timeout=timeout).chat.completions.create(
                model=self.config.model,
                messages=messages,
                temperature=resolved_temperature,
                max_tokens=resolved_max_tokens,
                stream=False,
            )
        except OpenAIError as error:
            raise ModelRequestError(f"Model API request failed: {error}") from error

        content = _extract_message_content(response)
        log.info(
            "model_request_completed",
            provider=self.config.provider,
            model=self.config.model,
        )
        return content

    def generate_text(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        timeout_seconds: float | None = None,
    ) -> str:
        """Generate text from a system prompt and a user prompt."""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        return self.chat(
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout_seconds=timeout_seconds,
        )


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


def _extract_message_content(response: Any) -> str:
    try:
        content = response.choices[0].message.content
    except (AttributeError, IndexError, TypeError) as error:
        raise ModelResponseError(
            "Model API response is missing choices[0].message.content"
        ) from error

    if not isinstance(content, str):
        raise ModelResponseError("Model API response content must be a string")

    return content
