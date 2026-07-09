from __future__ import annotations

from typing import Any

import requests
import structlog

from model_client.config import ModelConfig

log = structlog.get_logger(__name__)


class ModelRequestError(RuntimeError):
    """Raised when a model API request fails."""


class ModelResponseError(RuntimeError):
    """Raised when a model API response has an unexpected shape."""


class ModelClient:
    """Client for OpenAI-compatible chat completions APIs."""

    def __init__(self, config: ModelConfig) -> None:
        self.config = config

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

        payload = {
            "model": self.config.model,
            "messages": messages,
            "temperature": self.config.temperature if temperature is None else temperature,
            "max_tokens": self.config.max_tokens if max_tokens is None else max_tokens,
            "stream": False,
        }
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
        }
        timeout = self.config.timeout_seconds if timeout_seconds is None else timeout_seconds

        log.info(
            "model_request_started",
            provider=self.config.provider,
            model=self.config.model,
            url=self.chat_completions_url,
        )

        try:
            response = requests.post(
                self.chat_completions_url,
                headers=headers,
                json=payload,
                timeout=timeout,
            )
        except requests.RequestException as error:
            raise ModelRequestError(f"Model API request failed: {error}") from error

        if response.status_code >= 400:
            detail = response.text[:500]
            raise ModelRequestError(
                f"Model API returned HTTP {response.status_code}: {detail}"
            )

        try:
            data = response.json()
        except ValueError as error:
            raise ModelResponseError("Model API response is not valid JSON") from error

        content = _extract_message_content(data)
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


def _extract_message_content(data: dict[str, Any]) -> str:
    try:
        content = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as error:
        raise ModelResponseError(
            "Model API response is missing choices[0].message.content"
        ) from error

    if not isinstance(content, str):
        raise ModelResponseError("Model API response content must be a string")

    return content
