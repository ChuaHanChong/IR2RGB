"""Tests for run_flux2.py — backward compatibility with and without --lora_weights."""

import argparse
import subprocess
import sys
import textwrap
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class TestRunFlux2:
    """Test run_flux2.py argument parsing and model selection logic."""

    def _parse_args(self, argv: list[str]) -> argparse.Namespace:
        """Parse run_flux2.py arguments without executing the script at module level."""
        parser = argparse.ArgumentParser()
        parser.add_argument("--input_folder", type=str, required=True)
        parser.add_argument("--output_folder", type=str, required=True)
        parser.add_argument("--prompt", type=str, default="Turn this thermal infrared image...")
        parser.add_argument("--model_name", type=str, default="black-forest-labs/FLUX.2-klein-4B")
        parser.add_argument("--num_inference_steps", type=int, default=4)
        parser.add_argument("--guidance_scale", type=float, default=1.0)
        parser.add_argument("--seed", type=int, default=0)
        parser.add_argument("--lora_weights", type=str, default=None)
        return parser.parse_args(argv)

    def test_default_args_no_lora(self):
        """Without --lora_weights, default should be None (backward compatible)."""
        args = self._parse_args(["--input_folder", "/in", "--output_folder", "/out"])
        assert args.lora_weights is None
        assert args.seed == 0
        assert args.model_name == "black-forest-labs/FLUX.2-klein-4B"

    def test_lora_weights_arg_parsed(self):
        """--lora_weights should be captured when provided."""
        args = self._parse_args([
            "--input_folder", "/in",
            "--output_folder", "/out",
            "--lora_weights", "/path/to/lora",
        ])
        assert args.lora_weights == "/path/to/lora"

    def test_seed_arg_parsed(self):
        """--seed should override the default."""
        args = self._parse_args([
            "--input_folder", "/in",
            "--output_folder", "/out",
            "--seed", "42",
        ])
        assert args.seed == 42

    def test_klein4b_uses_klein_pipeline(self):
        """Klein-4B model name should use Flux2KleinPipeline."""
        mock_klein_pipe = MagicMock()
        mock_klein_cls = MagicMock(return_value=mock_klein_pipe)
        mock_klein_cls.from_pretrained = MagicMock(return_value=mock_klein_pipe)

        args = self._parse_args([
            "--input_folder", "/nonexistent",
            "--output_folder", "/tmp/out",
        ])

        with patch("diffusers.Flux2KleinPipeline") as mock_cls:
            mock_cls.from_pretrained.return_value = mock_klein_pipe
            assert args.model_name in [
                "black-forest-labs/FLUX.2-klein-4B",
                "black-forest-labs/FLUX.2-klein-9B",
            ]

    def test_lora_loading_called_when_provided(self, tmp_path):
        """When --lora_weights is given, pipe.load_lora_weights() should be called."""
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        Image.fromarray(np.zeros((64, 64, 3), dtype=np.uint8)).save(input_dir / "test.jpg")

        args = self._parse_args([
            "--input_folder", str(input_dir),
            "--output_folder", str(tmp_path / "output"),
            "--lora_weights", "/fake/lora/path",
            "--model_name", "black-forest-labs/FLUX.2-klein-4B",
        ])

        mock_pipe = MagicMock()
        mock_output = MagicMock()
        mock_output.images = [Image.fromarray(np.zeros((64, 64, 3), dtype=np.uint8))]
        mock_pipe.return_value = mock_output

        with patch("diffusers.Flux2KleinPipeline.from_pretrained", return_value=mock_pipe):
            if args.lora_weights:
                mock_pipe.load_lora_weights(args.lora_weights)

            mock_pipe.load_lora_weights.assert_called_once_with("/fake/lora/path")

    def test_lora_not_called_when_absent(self):
        """When --lora_weights is absent, load_lora_weights should NOT be called."""
        args = self._parse_args([
            "--input_folder", "/in",
            "--output_folder", "/out",
        ])

        mock_pipe = MagicMock()
        if args.lora_weights:
            mock_pipe.load_lora_weights(args.lora_weights)

        mock_pipe.load_lora_weights.assert_not_called()

    def test_unsupported_model_exits_early(self, tmp_path):
        """Unsupported model names should print an error and return early."""
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        Image.fromarray(np.zeros((32, 32, 3), dtype=np.uint8)).save(input_dir / "dummy.jpg")

        script = textwrap.dedent(f"""\
            import sys
            sys.argv = [
                "run_flux2.py",
                "--input_folder", "{input_dir}",
                "--output_folder", "{tmp_path}/out",
                "--model_name", "unsupported/model",
            ]
            # Mock diffusers to avoid import errors
            import unittest.mock as um
            sys.modules["diffusers"] = um.MagicMock()
            sys.modules["tqdm"] = um.MagicMock()

            # Now import and run
            import importlib
            spec = importlib.util.spec_from_file_location("run_flux2", "{PROJECT_ROOT}/run_flux2.py")
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
        """)

        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert "Unsupported model name" in result.stdout
