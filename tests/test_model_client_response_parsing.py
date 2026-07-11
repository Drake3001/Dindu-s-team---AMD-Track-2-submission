import unittest

from model_client.response_parsing import (
    format_json_for_prompt,
    parse_json_from_model_response,
)
from model_client.types import ModelResponseError


class ModelResponseParsingTests(unittest.TestCase):
    def test_parse_clean_json(self) -> None:
        parsed = parse_json_from_model_response('{"events": [{"order": 1}]}')

        self.assertEqual(parsed["events"][0]["order"], 1)

    def test_parse_fenced_json(self) -> None:
        parsed = parse_json_from_model_response(
            '```json\n{"events": [{"description": "Ocean"}]}\n```'
        )

        self.assertEqual(parsed["events"][0]["description"], "Ocean")

    def test_parse_json_with_surrounding_text(self) -> None:
        parsed = parse_json_from_model_response(
            'Here is the analysis:\n{"setting": "beach"}\nDone.'
        )

        self.assertEqual(parsed["setting"], "beach")

    def test_parse_json_array_with_surrounding_text(self) -> None:
        parsed = parse_json_from_model_response(
            'Events:\n[{"order": 1, "description": "Start"}]'
        )

        self.assertEqual(parsed[0]["description"], "Start")

    def test_invalid_response_raises_model_response_error(self) -> None:
        with self.assertRaises(ModelResponseError):
            parse_json_from_model_response("no json here")

    def test_format_json_for_prompt_pretty_prints(self) -> None:
        formatted = format_json_for_prompt({"setting": "ocean"})

        self.assertEqual(formatted, '{\n  "setting": "ocean"\n}')


if __name__ == "__main__":
    unittest.main()
