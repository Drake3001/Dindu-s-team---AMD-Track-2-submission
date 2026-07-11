import unittest

from model_client.prompts import list_prompt_names, load_prompt, load_prompts


class PromptLoaderTests(unittest.TestCase):
    def test_list_prompt_names_includes_video_analysis_approaches(self) -> None:
        names = list_prompt_names()
        self.assertIn("detailed_chronological", names)
        self.assertIn("structured_breakdown", names)
        self.assertIn("concise_factual", names)

    def test_load_prompt_returns_system_and_user(self) -> None:
        prompt = load_prompt("concise_factual")
        self.assertEqual(prompt.name, "concise_factual")
        self.assertIn("video analyst", prompt.system.lower())
        self.assertIn("json", prompt.system.lower())
        self.assertIn("factual", prompt.user.lower())
        self.assertIn("setting", prompt.user.lower())

    def test_load_prompts_defaults_to_all(self) -> None:
        prompts = load_prompts()
        self.assertEqual(len(prompts), len(list_prompt_names()))

    def test_load_prompts_can_select_one(self) -> None:
        prompts = load_prompts(["structured_breakdown"])
        self.assertEqual(len(prompts), 1)
        self.assertEqual(prompts[0].name, "structured_breakdown")

    def test_load_prompt_unknown_raises(self) -> None:
        with self.assertRaises(FileNotFoundError):
            load_prompt("does_not_exist")


if __name__ == "__main__":
    unittest.main()
