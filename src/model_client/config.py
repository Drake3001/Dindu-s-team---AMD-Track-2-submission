import os
from dataclasses import dataclass

from dotenv import find_dotenv, load_dotenv

DEFAULT_PROVIDER = "openrouter"
DEFAULT_TIMEOUT_SECONDS = 60.0
DEFAULT_TEMPERATURE = 0.7
DEFAULT_MAX_TOKENS = 512

PROVIDER_DEFAULTS = {
    "openrouter": {
        "base_url": "https://openrouter.ai/api/v1",
        "api_key_env": "OPENROUTER_API_KEY",
        "model_env": "OPENROUTER_MODEL",
    },
    "fireworks": {
        "base_url": "https://api.fireworks.ai/inference/v1",
        "api_key_env": "FIREWORKS_API_KEY",
        "model_env": "FIREWORKS_MODEL",
    },
}

_ENV_LOADED = False


class ModelConfigError(ValueError):
    """Raised when model client configuration is missing or invalid."""


@dataclass(frozen=True)
class ModelConfig:
    provider: str
    api_key: str
    model: str
    base_url: str
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS
    temperature: float = DEFAULT_TEMPERATURE
    max_tokens: int = DEFAULT_MAX_TOKENS


def _ensure_env_loaded() -> None:
    global _ENV_LOADED
    if not _ENV_LOADED:
        load_dotenv(find_dotenv(filename="credentials/.env"))
        _ENV_LOADED = True


def _read_float_env(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None or value == "":
        return default

    try:
        return float(value)
    except ValueError as error:
        raise ModelConfigError(f"{name} must be a number") from error


def _read_int_env(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or value == "":
        return default

    try:
        return int(value)
    except ValueError as error:
        raise ModelConfigError(f"{name} must be an integer") from error


def load_model_config(
    provider: str | None = None,
    api_key: str | None = None,
    model: str | None = None,
    base_url: str | None = None,
    timeout_seconds: float | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
) -> ModelConfig:
    """Load model client settings from overrides and `credentials/.env`."""
    _ensure_env_loaded()

    provider_name = (provider or os.getenv("MODEL_PROVIDER") or DEFAULT_PROVIDER).strip().lower()
    provider_defaults = PROVIDER_DEFAULTS.get(provider_name)
    if provider_defaults is None:
        supported = ", ".join(sorted(PROVIDER_DEFAULTS))
        raise ModelConfigError(
            f"Unsupported MODEL_PROVIDER '{provider_name}'. Supported providers: {supported}"
        )

    api_key_env = provider_defaults["api_key_env"]
    model_env = provider_defaults["model_env"]

    resolved_api_key = api_key or os.getenv(api_key_env)
    if not resolved_api_key:
        raise ModelConfigError(f"Missing API key. Set {api_key_env} in credentials/.env")

    resolved_model = model or os.getenv(model_env)
    if not resolved_model:
        raise ModelConfigError(f"Missing model name. Set {model_env} in credentials/.env")

    resolved_base_url = (
        base_url
        or os.getenv("MODEL_BASE_URL")
        or provider_defaults["base_url"]
    )

    return ModelConfig(
        provider=provider_name,
        api_key=resolved_api_key,
        model=resolved_model,
        base_url=resolved_base_url.rstrip("/"),
        timeout_seconds=timeout_seconds
        if timeout_seconds is not None
        else _read_float_env("MODEL_TIMEOUT_SECONDS", DEFAULT_TIMEOUT_SECONDS),
        temperature=temperature
        if temperature is not None
        else _read_float_env("MODEL_TEMPERATURE", DEFAULT_TEMPERATURE),
        max_tokens=max_tokens
        if max_tokens is not None
        else _read_int_env("MODEL_MAX_TOKENS", DEFAULT_MAX_TOKENS),
    )
