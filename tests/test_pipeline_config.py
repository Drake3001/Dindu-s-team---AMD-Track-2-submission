import tempfile
import unittest
from pathlib import Path

import yaml

from workflow.config import ConfigError, load_pipeline_config


class PipelineConfigTests(unittest.TestCase):
    def test_loads_full_yaml(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "pipeline.yaml"
            config_path.write_text(
                """
input:
  tasks: input/custom_tasks.json
  videos_dir: custom_videos
  skip_download: true
output:
  path: output/custom_results.json
pipeline:
  fps: 2.0
  max_dim: 384
  prune_threshold: 8.0
  grid_cols: 2
  grid_rows: 2
  max_frames: 16
  concurrency:
    max_download_workers: 1
    max_preprocess_workers: 2
    max_inference_workers: 4
vlm:
  provider: fireworks
  model: accounts/fireworks/models/test-vlm
  temperature: 0.1
  max_tokens: 900
  timeout_seconds: 45
  prompt: concise_factual
captions:
  provider: openrouter
  model: test-caption-model
  temperature: 0.5
  max_tokens: 300
  timeout_seconds: 30
  styles:
    - formal
    - sarcastic
""",
                encoding="utf-8",
            )

            cfg = load_pipeline_config(config_path, project_root=Path(tmpdir))

        self.assertTrue(cfg.input.skip_download)
        self.assertEqual(cfg.pipeline.fps, 2.0)
        self.assertEqual(cfg.pipeline.max_dim, 384)
        self.assertEqual(cfg.pipeline.concurrency.max_inference_workers, 4)
        self.assertEqual(cfg.vlm.prompt, "concise_factual")
        self.assertEqual(cfg.vlm.model.provider, "fireworks")
        self.assertEqual(cfg.vlm.model.model, "accounts/fireworks/models/test-vlm")
        self.assertEqual(cfg.captions.model.model, "test-caption-model")
        self.assertEqual(cfg.captions.styles, ["formal", "sarcastic"])

    def test_partial_yaml_merges_defaults(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "pipeline.yaml"
            config_path.write_text(
                """
pipeline:
  fps: 0.5
vlm:
  prompt: structured_breakdown
""",
                encoding="utf-8",
            )

            cfg = load_pipeline_config(config_path, project_root=Path(tmpdir))

        self.assertEqual(cfg.pipeline.fps, 0.5)
        self.assertEqual(cfg.pipeline.max_dim, 512)
        self.assertEqual(cfg.vlm.prompt, "structured_breakdown")
        self.assertIsNone(cfg.vlm.model.provider)
        self.assertIsNone(cfg.captions.styles)

    def test_unknown_prompt_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "pipeline.yaml"
            config_path.write_text(
                """
vlm:
  prompt: does_not_exist
""",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ConfigError, "Unknown VLM prompt"):
                load_pipeline_config(config_path, project_root=Path(tmpdir))

    def test_unknown_style_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "pipeline.yaml"
            config_path.write_text(
                """
captions:
  styles:
    - formal
    - not_a_style
""",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ConfigError, "Unknown caption style"):
                load_pipeline_config(config_path, project_root=Path(tmpdir))

    def test_null_model_and_provider_stay_none(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "pipeline.yaml"
            config_path.write_text(
                yaml.safe_dump(
                    {
                        "vlm": {"prompt": "detailed_chronological", "provider": None, "model": None},
                        "captions": {"provider": None, "model": None},
                    }
                ),
                encoding="utf-8",
            )

            cfg = load_pipeline_config(config_path, project_root=Path(tmpdir))

        self.assertIsNone(cfg.vlm.model.provider)
        self.assertIsNone(cfg.vlm.model.model)
        self.assertIsNone(cfg.captions.model.provider)
        self.assertIsNone(cfg.captions.model.model)


if __name__ == "__main__":
    unittest.main()
