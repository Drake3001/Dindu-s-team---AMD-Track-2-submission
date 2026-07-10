import unittest
from unittest.mock import Mock

from model_client.caption_generation import (
    build_caption_prompt,
    caption_temperature_for_style,
    generate_caption,
    list_caption_styles,
)


class CaptionGenerationTests(unittest.TestCase):
    def test_list_caption_styles_includes_expected_styles(self) -> None:
        styles = list_caption_styles()

        self.assertIn("formal", styles)
        self.assertIn("humorous_non_tech", styles)
        self.assertIn("humorous_tech", styles)
        self.assertIn("sarcastic", styles)

    def test_build_caption_prompt_inserts_pretty_json(self) -> None:
        prompt = build_caption_prompt(
            "formal",
            {"events": [{"order": 1, "description": "A person paddles."}]},
        )

        self.assertIn('"events": [', prompt)
        self.assertIn('"description": "A person paddles."', prompt)
        self.assertNotIn("[INSERT_VIDEO_JSON_HERE]", prompt)

    def test_humorous_styles_use_higher_temperature(self) -> None:
        self.assertEqual(caption_temperature_for_style("formal"), 0.7)
        self.assertEqual(caption_temperature_for_style("humorous_non_tech"), 1.0)
        self.assertEqual(caption_temperature_for_style("humorous_tech"), 1.0)
        self.assertEqual(caption_temperature_for_style("sarcastic"), 1.0)

    def test_generate_caption_uses_style_temperature_by_default(self) -> None:
        model_client = Mock()
        model_client.generate_text.return_value = "caption"

        result = generate_caption(
            model_client,
            {"setting": "ocean"},
            "humorous_tech",
        )

        self.assertEqual(result, "caption")
        kwargs = model_client.generate_text.call_args.kwargs
        self.assertEqual(kwargs["temperature"], 1.0)

    def test_generate_caption_allows_temperature_override(self) -> None:
        model_client = Mock()
        model_client.generate_text.return_value = "caption"

        generate_caption(
            model_client,
            {"setting": "ocean"},
            "humorous_tech",
            temperature=0.3,
        )

        kwargs = model_client.generate_text.call_args.kwargs
        self.assertEqual(kwargs["temperature"], 0.3)


if __name__ == "__main__":
    unittest.main()
