import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

from model_client.api import create_async_model_client
from model_client.client import AsyncModelClient


class AsyncModelClientTests(unittest.IsolatedAsyncioTestCase):
    @patch("model_client.client.AsyncOpenAI")
    async def test_async_chat_request(self, async_openai_class: Mock) -> None:
        sdk_client = async_openai_class.return_value
        sdk_client.with_options.return_value = sdk_client
        sdk_client.chat.completions.create = AsyncMock(
            return_value=_completion("async hello")
        )

        client = create_async_model_client(
            provider="openrouter",
            api_key="secret",
            model="openai/test",
            temperature=0.2,
            max_tokens=123,
        )

        result = await client.chat([{"role": "user", "content": "Hi"}])

        self.assertEqual(result, "async hello")
        async_openai_class.assert_called_once_with(
            api_key="secret",
            base_url="https://openrouter.ai/api/v1",
            timeout=60.0,
            max_retries=2,
        )
        sdk_client.with_options.assert_called_once_with(timeout=60.0)
        sdk_client.chat.completions.create.assert_awaited_once_with(
            model="openai/test",
            messages=[{"role": "user", "content": "Hi"}],
            temperature=0.2,
            max_tokens=123,
            stream=False,
        )

    @patch("model_client.client.AsyncOpenAI")
    async def test_generate_from_frame_grids(self, async_openai_class: Mock) -> None:
        sdk_client = async_openai_class.return_value
        sdk_client.with_options.return_value = sdk_client
        sdk_client.chat.completions.create = AsyncMock(
            return_value=_completion("grid response")
        )

        client = AsyncModelClient(
            create_async_model_client(
                provider="openrouter",
                api_key="secret",
                model="vision/test",
            ).config
        )

        result = await client.generate_from_frame_grids(
            ["grid1"],
            "System",
            "Describe.",
            grids_meta=[
                {
                    "frame_count": 8,
                    "cols": 4,
                    "rows": 4,
                    "empty_cells": 8,
                    "width_px": 512,
                    "height_px": 512,
                }
            ],
        )

        self.assertEqual(result, "grid response")
        messages = sdk_client.chat.completions.create.await_args.kwargs["messages"]
        self.assertEqual(messages[0]["content"], "System")
        self.assertEqual(messages[1]["content"][1]["image_url"]["url"], "data:image/jpeg;base64,grid1")


def _completion(content: str) -> SimpleNamespace:
    return SimpleNamespace(
        choices=[
            SimpleNamespace(message=SimpleNamespace(content=content)),
        ],
    )


if __name__ == "__main__":
    unittest.main()
