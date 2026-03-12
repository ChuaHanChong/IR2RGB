"""Tests for DreamBooth LoRA training scripts — argument parsing validation.

Full training requires GPU + hours of compute, so we validate argument parsing
and that the scripts are syntactically valid Python.
"""

import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class TestTrainingScriptArgs:
    """Validate that the DreamBooth LoRA training scripts parse arguments correctly."""

    @pytest.mark.parametrize("script_name", [
        "train_dreambooth_lora_flux2_img2img.py",
        "train_dreambooth_lora_flux2_klein_img2img.py",
    ])
    def test_script_compiles(self, script_name):
        """Training scripts should be valid Python (compile without syntax errors)."""
        script_path = PROJECT_ROOT / script_name
        with open(script_path) as f:
            source = f.read()
        compile(source, script_path, "exec")

    @pytest.mark.parametrize("script_name", [
        "train_dreambooth_lora_flux2_img2img.py",
        "train_dreambooth_lora_flux2_klein_img2img.py",
    ])
    def test_help_flag_works(self, script_name):
        """Training scripts should respond to --help without errors."""
        result = subprocess.run(
            [sys.executable, str(PROJECT_ROOT / script_name), "--help"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0
        assert "--pretrained_model_name_or_path" in result.stdout
        assert "--instance_data_dir" in result.stdout or "--dataset_name" in result.stdout

    @pytest.mark.parametrize("script_name", [
        "train_dreambooth_lora_flux2_img2img.py",
        "train_dreambooth_lora_flux2_klein_img2img.py",
    ])
    def test_required_args_error_when_missing(self, script_name):
        """Training scripts should fail clearly when required args are missing."""
        result = subprocess.run(
            [sys.executable, str(PROJECT_ROOT / script_name)],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode != 0
        assert "required" in result.stderr.lower() or "error" in result.stderr.lower()
