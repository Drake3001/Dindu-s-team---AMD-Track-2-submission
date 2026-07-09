import os
import unittest
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
    @patch("model_client.client.requests.post")
    def test_openrouter_chat_request(self, post: Mock) -> None:
        post.return_value.status_code = 200
        post.return_value.json.return_value = {
            "choices": [{"message": {"content": "hello"}}]
        }

        client = create_model_client(
            provider="openrouter",
            api_key="secret",
            model="openai/test",
            temperature=0.2,
            max_tokens=123,
        )

        result = client.chat([{"role": "user", "content": "Hi"}])

        self.assertEqual(result, "hello")
        post.assert_called_once()
        _, kwargs = post.call_args
        self.assertEqual(
            post.call_args.args[0],
            "https://openrouter.ai/api/v1/chat/completions",
        )
        self.assertEqual(kwargs["headers"]["Authorization"], "Bearer secret")
        self.assertEqual(kwargs["json"]["model"], "openai/test")
        self.assertEqual(kwargs["json"]["temperature"], 0.2)
        self.assertEqual(kwargs["json"]["max_tokens"], 123)

    @patch("model_client.client.requests.post")
    def test_fireworks_generate_text_request(self, post: Mock) -> None:
        post.return_value.status_code = 200
        post.return_value.json.return_value = {
            "choices": [{"message": {"content": "done"}}]
        }

        result = generate_text(
            "System",
            "User",
            provider="fireworks",
            api_key="secret",
            model="accounts/fireworks/models/test",
        )

        self.assertEqual(result, "done")
        self.assertEqual(
            post.call_args.args[0],
            "https://api.fireworks.ai/inference/v1/chat/completions",
        )
        self.assertEqual(
            post.call_args.kwargs["json"]["messages"],
            [
                {"role": "system", "content": "System"},
                {"role": "user", "content": "User"},
            ],
        )


if __name__ == "__main__":
    unittest.main()
