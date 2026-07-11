import os
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from model_client import generate_from_frame_grids, generate_text
from model_client.api import create_model_client
from model_client.config import ModelConfigError, load_model_config
from model_client.messages import (
    build_frame_grids_context,
    build_frame_grids_messages,
    build_image_messages,
)
from model_client.types import ModelRequestError

SAMPLE_GRIDS_META = [
    {
        "frame_count": 10,
        "cols": 4,
        "rows": 4,
        "empty_cells": 6,
        "width_px": 512,
        "height_px": 512,
    },
    {
        "frame_count": 4,
        "cols": 4,
        "rows": 4,
        "empty_cells": 12,
        "width_px": 512,
        "height_px": 384,
    },
]


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
    @patch("model_client.client.OpenAI")
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

    @patch("model_client.client.OpenAI")
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

    @patch("model_client.client.OpenAI")
    def test_generate_from_images_base64_builds_multimodal_messages(
        self,
        openai_class: Mock,
    ) -> None:
        sdk_client = openai_class.return_value
        sdk_client.with_options.return_value = sdk_client
        sdk_client.chat.completions.create.return_value = _completion("vision")

        client = create_model_client(
            provider="openrouter",
            api_key="secret",
            model="vision/test",
        )

        result = client.generate_from_images_base64(
            ["abc123", "data:image/png;base64,xyz789"],
            "System",
            "Describe these frames.",
            image_mime_type="image/png",
        )

        self.assertEqual(result, "vision")
        sdk_client.chat.completions.create.assert_called_once_with(
            model="vision/test",
            messages=[
                {"role": "system", "content": "System"},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Describe these frames."},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": "data:image/png;base64,abc123",
                            },
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": "data:image/png;base64,xyz789",
                            },
                        },
                    ],
                },
            ],
            temperature=0.7,
            max_tokens=512,
            stream=False,
        )

    @patch("model_client.client.OpenAI")
    def test_generate_from_frame_grids_adds_grid_instruction(
        self,
        openai_class: Mock,
    ) -> None:
        sdk_client = openai_class.return_value
        sdk_client.with_options.return_value = sdk_client
        sdk_client.chat.completions.create.return_value = _completion("grid")

        result = generate_from_frame_grids(
            ["grid1", "grid2"],
            "System",
            "Describe the video.",
            grids_meta=SAMPLE_GRIDS_META,
            provider="openrouter",
            api_key="secret",
            model="vision/test",
        )

        self.assertEqual(result, "grid")
        messages = sdk_client.chat.completions.create.call_args.kwargs["messages"]
        content = messages[1]["content"]
        self.assertIn("2 base64-encoded JPEG image(s)", content[0]["text"])
        self.assertIn("Image 1:", content[0]["text"])
        self.assertIn("Image 2:", content[0]["text"])
        self.assertIn("solid black", content[0]["text"])
        self.assertIn("Describe the video.", content[0]["text"])
        self.assertEqual(
            content[1]["image_url"]["url"],
            "data:image/jpeg;base64,grid1",
        )
        self.assertEqual(
            content[2]["image_url"]["url"],
            "data:image/jpeg;base64,grid2",
        )


class ModelClientMessageTests(unittest.TestCase):
    def test_build_image_messages_normalizes_raw_base64(self) -> None:
        messages = build_image_messages(
            ["abc123", "data:image/png;base64,xyz789"],
            "System",
            "Describe.",
            image_mime_type="image/png",
        )

        content = messages[1]["content"]
        self.assertEqual(content[1]["image_url"]["url"], "data:image/png;base64,abc123")
        self.assertEqual(content[2]["image_url"]["url"], "data:image/png;base64,xyz789")

    def test_build_image_messages_rejects_empty_images(self) -> None:
        with self.assertRaisesRegex(ModelRequestError, "images_base64"):
            build_image_messages([], "System", "Describe.")

    def test_build_frame_grids_context_includes_each_grid(self) -> None:
        context = build_frame_grids_context(SAMPLE_GRIDS_META, image_count=2)

        self.assertIn("2 base64-encoded JPEG image(s)", context)
        self.assertIn("Image 1: 10 frame(s) in a 4x4 grid (512x512 px)", context)
        self.assertIn("Image 2: 4 frame(s) in a 4x4 grid (512x384 px)", context)
        self.assertIn("solid black", context)

    def test_build_frame_grids_messages_adds_all_images(self) -> None:
        messages = build_frame_grids_messages(
            ["grid1", "grid2"],
            "System",
            "Describe.",
            grids_meta=SAMPLE_GRIDS_META,
        )
        content = messages[1]["content"]

        self.assertIn("Describe.", content[0]["text"])
        self.assertEqual(len([part for part in content if part["type"] == "image_url"]), 2)


def _completion(content: str) -> SimpleNamespace:
    return SimpleNamespace(
        choices=[
            SimpleNamespace(message=SimpleNamespace(content=content)),
        ],
    )


if __name__ == "__main__":
    unittest.main()
