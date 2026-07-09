import os
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from model_client import generate_text
from model_client.api import create_model_client
from model_client.config import ModelConfigError, load_model_config


class ModelClientConfigTests(unittest.TestCase):
    def test_missing_api_key_raises_clear_error(self) -> None:
        with patch.dict(os.environ, {"MODEL_PROVIDER": "openrouter", "OPENROUTER_MODEL": "test"}, clear=True):
            with self.assertRaisesRegex(ModelConfigError, "OPENROUTER_API_KEY"):
                load_model_config()

    def test_unknown_provider_raises_clear_error(self) -> None:
        with patch.dict(os.environ, {"MODEL_PROVIDER": "unknown"}, clear=True):
            with self.assertRaisesRegex(ModelConfigError, "Unsupported MODEL_PROVIDER"):
                load_model_config()


class ModelClientRequestTests(unittest.TestCase):
    @patch("model_client.api.OpenAI")
    def test_openrouter_chat_request(self, openai_class: Mock) -> None:
        sdk_client = openai_class.return_value
        sdk_client.with_options.return_value = sdk_client
        sdk_client.chat.completions.create.return_value = _completion("hello")

        client = create_model_client(
            provider="openrouter",
            api_key="secret",
            model="openai/test",
            temperature=0.2,
            max_tokens=123,
        )

        result = client.chat([{"role": "user", "content": "Hi"}])

        self.assertEqual(result, "hello")
        openai_class.assert_called_once_with(
            api_key="secret",
            base_url="https://openrouter.ai/api/v1",
            timeout=60.0,
        )
        sdk_client.with_options.assert_called_once_with(timeout=60.0)
        sdk_client.chat.completions.create.assert_called_once_with(
            model="openai/test",
            messages=[{"role": "user", "content": "Hi"}],
            temperature=0.2,
            max_tokens=123,
            stream=False,
        )

    @patch("model_client.api.OpenAI")
    def test_fireworks_generate_text_request(self, openai_class: Mock) -> None:
        sdk_client = openai_class.return_value
        sdk_client.with_options.return_value = sdk_client
        sdk_client.chat.completions.create.return_value = _completion("done")

        result = generate_text(
            "System",
            "User",
            provider="fireworks",
            api_key="secret",
            model="accounts/fireworks/models/test",
        )

        self.assertEqual(result, "done")
        openai_class.assert_called_once_with(
            api_key="secret",
            base_url="https://api.fireworks.ai/inference/v1",
            timeout=60.0,
        )
        sdk_client.chat.completions.create.assert_called_once_with(
            model="accounts/fireworks/models/test",
            messages=[
                {"role": "system", "content": "System"},
                {"role": "user", "content": "User"},
            ],
            temperature=0.7,
            max_tokens=512,
            stream=False,
        )


def _completion(content: str) -> SimpleNamespace:
    return SimpleNamespace(
        choices=[
            SimpleNamespace(message=SimpleNamespace(content=content)),
        ],
    )


if __name__ == "__main__":
    unittest.main()
