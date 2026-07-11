from typing import Any

from openai import OpenAI, OpenAIError
import structlog

from model_client.config import ModelConfig
from model_client.messages import (
    DEFAULT_IMAGE_MIME_TYPE,
    build_frame_grids_messages,
    build_image_messages,
    build_text_messages,
)
from model_client.types import ModelRequestError, ModelResponseError

log = structlog.get_logger(__name__)


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
        return self.chat(
            build_text_messages(system_prompt, user_prompt),
            temperature=temperature,
            max_tokens=max_tokens,
            timeout_seconds=timeout_seconds,
        )

    def generate_from_image_base64(
        self,
        image_base64: str,
        system_prompt: str,
        user_prompt: str,
        *,
        image_mime_type: str = DEFAULT_IMAGE_MIME_TYPE,
        temperature: float | None = None,
        max_tokens: int | None = None,
        timeout_seconds: float | None = None,
    ) -> str:
        """Generate text from a prompt and one base64-encoded image."""
        return self.generate_from_images_base64(
            [image_base64],
            system_prompt,
            user_prompt,
            image_mime_type=image_mime_type,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout_seconds=timeout_seconds,
        )

    def generate_from_images_base64(
        self,
        images_base64: list[str],
        system_prompt: str,
        user_prompt: str,
        *,
        image_mime_type: str = DEFAULT_IMAGE_MIME_TYPE,
        temperature: float | None = None,
        max_tokens: int | None = None,
        timeout_seconds: float | None = None,
    ) -> str:
        """Generate text from a prompt and one or more base64-encoded images."""
        return self.chat(
            build_image_messages(
                images_base64,
                system_prompt,
                user_prompt,
                image_mime_type=image_mime_type,
            ),
            temperature=temperature,
            max_tokens=max_tokens,
            timeout_seconds=timeout_seconds,
        )

    def generate_from_frame_grids(
        self,
        grids_base64: list[str],
        system_prompt: str,
        user_prompt: str,
        *,
        grids_meta: list[dict[str, int]],
        image_mime_type: str = DEFAULT_IMAGE_MIME_TYPE,
        temperature: float | None = None,
        max_tokens: int | None = None,
        timeout_seconds: float | None = None,
    ) -> str:
        """Generate text from one or more base64 grid images in a single message."""
        return self.chat(
            build_frame_grids_messages(
                grids_base64,
                system_prompt,
                user_prompt,
                grids_meta=grids_meta,
                image_mime_type=image_mime_type,
            ),
            temperature=temperature,
            max_tokens=max_tokens,
            timeout_seconds=timeout_seconds,
        )


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
