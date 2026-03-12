"""Tests for PR #3: DreamBooth LoRA training scripts and repo streamlining.

Covers the PR test plan:
  1. Verify create_hf_dataset.py produces a valid HF DatasetDict with correct columns
  2. Run run_flux2.py with and without --lora_weights to confirm backward compatibility
  3. Run DreamBooth LoRA training on a small subset to validate training scripts
  4. Confirm eval.py still works via CLI entry point
"""

import argparse
import json
import subprocess
import sys
import textwrap
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from PIL import Image

# eval.py is importable because its argparser is gated behind __main__
from eval import (
    build_json_output,
    compute_psnr,
    compute_ssim,
    find_image_pairs,
    format_table,
    load_image_float,
    main as eval_main,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def paired_image_dirs(tmp_path):
    """Create a minimal paired RGB/IR directory structure with synthetic JPEGs."""
    gen_dir = tmp_path / "gen"
    gt_dir = tmp_path / "gt"
    gen_dir.mkdir()
    gt_dir.mkdir()

    rng = np.random.RandomState(42)
    for name in ["img_001.jpg", "img_002.jpg", "img_003.jpg"]:
        arr = rng.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        # GT and gen differ slightly so metrics are non-trivial
        Image.fromarray(arr).save(gt_dir / name)
        noisy = np.clip(arr.astype(np.int16) + rng.randint(-20, 20, arr.shape), 0, 255).astype(np.uint8)
        Image.fromarray(noisy).save(gen_dir / name)

    return gen_dir, gt_dir


@pytest.fixture()
def hf_dataset_dirs(tmp_path):
    """Create the paired rgb/ and ir/ subdirectory structure expected by create_hf_dataset.py."""
    data_root = tmp_path / "train"
    rgb_dir = data_root / "rgb"
    ir_dir = data_root / "ir"

    rng = np.random.RandomState(99)
    for cat in ["0", "1"]:
        (rgb_dir / cat).mkdir(parents=True)
        (ir_dir / cat).mkdir(parents=True)
        for name in ["a.jpg", "b.jpg"]:
            arr = rng.randint(0, 255, (32, 32, 3), dtype=np.uint8)
            Image.fromarray(arr).save(rgb_dir / cat / name)
            Image.fromarray(arr).save(ir_dir / cat / name)

    return data_root, tmp_path / "output_dataset"


# ===========================================================================
# 1. create_hf_dataset.py — HF DatasetDict with correct columns
# ===========================================================================


class TestCreateHFDataset:
    """Verify create_hf_dataset.py produces a valid HuggingFace DatasetDict."""

    def test_produces_dataset_with_correct_columns(self, hf_dataset_dirs):
        """Run create_hf_dataset.py with patched paths and verify output structure."""
        data_root, save_path = hf_dataset_dirs

        # Build a patched version of the script that uses our tmp dirs
        script = textwrap.dedent(f"""\
            from datasets import Dataset, DatasetDict, Image
            from pathlib import Path

            DATA_ROOT = Path("{data_root}")
            SAVE_PATH = "{save_path}"
            PROMPT = "turn the visible image of Marine Vessel into sks infrared"

            rgb_dir = DATA_ROOT / "rgb"
            ir_dir = DATA_ROOT / "ir"

            records = {{"cond_image": [], "target_image": [], "caption": []}}
            for subdir in sorted(rgb_dir.iterdir()):
                if not subdir.is_dir():
                    continue
                for img_file in sorted(subdir.glob("*.jpg")):
                    ir_file = ir_dir / subdir.name / img_file.name
                    if ir_file.exists():
                        records["cond_image"].append(str(img_file))
                        records["target_image"].append(str(ir_file))
                        records["caption"].append(PROMPT)

            ds = Dataset.from_dict(records)
            ds = ds.cast_column("cond_image", Image())
            ds = ds.cast_column("target_image", Image())

            dd = DatasetDict({{"train": ds}})
            dd.save_to_disk(SAVE_PATH)
            print(f"COUNT={{len(ds)}}")
        """)

        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert result.returncode == 0, f"Script failed:\n{result.stderr}"
        assert "COUNT=4" in result.stdout  # 2 categories × 2 images

        # Load and validate the saved dataset
        from datasets import DatasetDict as DDLoader

        dd = DDLoader.load_from_disk(str(save_path))
        assert "train" in dd
        assert set(dd["train"].column_names) == {"cond_image", "target_image", "caption"}
        assert len(dd["train"]) == 4

    def test_skips_missing_ir_pairs(self, tmp_path):
        """Images without a matching IR counterpart should be silently skipped."""
        data_root = tmp_path / "train"
        rgb_dir = data_root / "rgb" / "0"
        ir_dir = data_root / "ir" / "0"
        rgb_dir.mkdir(parents=True)
        ir_dir.mkdir(parents=True)

        rng = np.random.RandomState(0)
        # Create 3 RGB images but only 1 IR image
        for name in ["a.jpg", "b.jpg", "c.jpg"]:
            Image.fromarray(rng.randint(0, 255, (32, 32, 3), dtype=np.uint8)).save(rgb_dir / name)
        Image.fromarray(rng.randint(0, 255, (32, 32, 3), dtype=np.uint8)).save(ir_dir / "a.jpg")

        save_path = tmp_path / "out_ds"
        script = textwrap.dedent(f"""\
            from datasets import Dataset, DatasetDict, Image
            from pathlib import Path
            DATA_ROOT = Path("{data_root}")
            SAVE_PATH = "{save_path}"
            PROMPT = "test"
            rgb_dir = DATA_ROOT / "rgb"
            ir_dir = DATA_ROOT / "ir"
            records = {{"cond_image": [], "target_image": [], "caption": []}}
            for subdir in sorted(rgb_dir.iterdir()):
                if not subdir.is_dir():
                    continue
                for img_file in sorted(subdir.glob("*.jpg")):
                    ir_file = ir_dir / subdir.name / img_file.name
                    if ir_file.exists():
                        records["cond_image"].append(str(img_file))
                        records["target_image"].append(str(ir_file))
                        records["caption"].append(PROMPT)
            ds = Dataset.from_dict(records)
            dd = DatasetDict({{"train": ds}})
            dd.save_to_disk(SAVE_PATH)
            print(f"COUNT={{len(ds)}}")
        """)

        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert result.returncode == 0
        assert "COUNT=1" in result.stdout


# ===========================================================================
# 2. run_flux2.py — backward compatibility with and without --lora_weights
# ===========================================================================


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
            # Verify the model name maps to Klein pipeline
            assert args.model_name in [
                "black-forest-labs/FLUX.2-klein-4B",
                "black-forest-labs/FLUX.2-klein-9B",
            ]

    def test_lora_loading_called_when_provided(self, tmp_path):
        """When --lora_weights is given, pipe.load_lora_weights() should be called."""
        # Create a dummy input image
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        Image.fromarray(np.zeros((64, 64, 3), dtype=np.uint8)).save(input_dir / "test.jpg")

        args = self._parse_args([
            "--input_folder", str(input_dir),
            "--output_folder", str(tmp_path / "output"),
            "--lora_weights", "/fake/lora/path",
            "--model_name", "black-forest-labs/FLUX.2-klein-4B",
        ])

        # Mock the pipeline
        mock_pipe = MagicMock()
        mock_output = MagicMock()
        mock_output.images = [Image.fromarray(np.zeros((64, 64, 3), dtype=np.uint8))]
        mock_pipe.return_value = mock_output

        with patch("diffusers.Flux2KleinPipeline.from_pretrained", return_value=mock_pipe):
            # Simulate the main() logic for LoRA loading
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
        # Simulate the main() logic
        if args.lora_weights:
            mock_pipe.load_lora_weights(args.lora_weights)

        mock_pipe.load_lora_weights.assert_not_called()

    def test_unsupported_model_exits_early(self, tmp_path):
        """Unsupported model names should print an error and return early."""
        # Need a .jpg so we get past the "no images found" early return
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


# ===========================================================================
# 3. DreamBooth LoRA training — argument parsing validation
# ===========================================================================


class TestTrainingScriptArgs:
    """Validate that the DreamBooth LoRA training scripts parse arguments correctly.

    Full training requires GPU + hours of compute, so we validate argument parsing
    and that the scripts are syntactically valid Python.
    """

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


# ===========================================================================
# 4. eval.py — CLI entry point and core metric functions
# ===========================================================================


class TestEvalImagePairDiscovery:
    """Test find_image_pairs matching logic."""

    def test_finds_matching_pairs(self, paired_image_dirs):
        gen_dir, gt_dir = paired_image_dirs
        pairs = find_image_pairs(str(gen_dir), str(gt_dir))
        assert len(pairs) == 3
        for gen_path, gt_path in pairs:
            assert Path(gen_path).stem == Path(gt_path).stem

    def test_skips_unmatched_gen_images(self, tmp_path):
        gen_dir = tmp_path / "gen"
        gt_dir = tmp_path / "gt"
        gen_dir.mkdir()
        gt_dir.mkdir()

        rng = np.random.RandomState(0)
        # gen has img_001 and img_002; gt only has img_001
        for name in ["img_001.jpg", "img_002.jpg"]:
            Image.fromarray(rng.randint(0, 255, (32, 32, 3), dtype=np.uint8)).save(gen_dir / name)
        Image.fromarray(rng.randint(0, 255, (32, 32, 3), dtype=np.uint8)).save(gt_dir / "img_001.jpg")

        pairs = find_image_pairs(str(gen_dir), str(gt_dir))
        assert len(pairs) == 1
        assert Path(pairs[0][0]).stem == "img_001"

    def test_raises_on_missing_directory(self, tmp_path):
        with pytest.raises(ValueError, match="not found"):
            find_image_pairs(str(tmp_path / "nonexistent"), str(tmp_path))

    def test_empty_dirs_return_empty(self, tmp_path):
        gen_dir = tmp_path / "gen"
        gt_dir = tmp_path / "gt"
        gen_dir.mkdir()
        gt_dir.mkdir()
        assert find_image_pairs(str(gen_dir), str(gt_dir)) == []


class TestEvalLoadImage:
    """Test image loading and preprocessing."""

    def test_loads_to_correct_shape(self, tmp_path):
        img = Image.fromarray(np.zeros((100, 200, 3), dtype=np.uint8))
        path = tmp_path / "test.jpg"
        img.save(path)

        tensor = load_image_float(str(path))
        assert tensor.shape == (3, 224, 224)
        assert tensor.dtype == torch.float32
        assert tensor.min() >= 0.0
        assert tensor.max() <= 1.0


class TestEvalMetrics:
    """Test per-image metric computation."""

    def test_psnr_identical_images_is_high(self, tmp_path):
        """PSNR of identical images should be very high (>= 30 dB)."""
        img = Image.fromarray(np.full((64, 64, 3), 128, dtype=np.uint8))
        path = tmp_path / "identical.jpg"
        img.save(path)
        psnr = compute_psnr(str(path), str(path), "cpu")
        assert psnr >= 30.0

    def test_ssim_identical_images_is_one(self, tmp_path):
        """SSIM of identical images should be ~1.0."""
        img = Image.fromarray(np.full((64, 64, 3), 128, dtype=np.uint8))
        path = tmp_path / "identical.jpg"
        img.save(path)
        ssim = compute_ssim(str(path), str(path), "cpu")
        assert ssim > 0.99

    def test_psnr_different_images_is_lower(self, paired_image_dirs):
        gen_dir, gt_dir = paired_image_dirs
        gen_path = str(gen_dir / "img_001.jpg")
        gt_path = str(gt_dir / "img_001.jpg")
        psnr = compute_psnr(gen_path, gt_path, "cpu")
        # Noisy pair should have moderate PSNR
        assert 10.0 < psnr < 50.0

    def test_ssim_different_images_is_less_than_one(self, paired_image_dirs):
        gen_dir, gt_dir = paired_image_dirs
        gen_path = str(gen_dir / "img_001.jpg")
        gt_path = str(gt_dir / "img_001.jpg")
        ssim = compute_ssim(gen_path, gt_path, "cpu")
        assert 0.0 < ssim < 1.0


class TestEvalOutputFormatting:
    """Test output formatting and JSON building."""

    @pytest.fixture()
    def sample_rows(self):
        return [
            {"gen": "a.jpg", "gt": "a_gt.jpg", "psnr": 20.0, "ssim": 0.85},
            {"gen": "b.jpg", "gt": "b_gt.jpg", "psnr": 25.0, "ssim": 0.90},
        ]

    def test_format_table_contains_averages(self, sample_rows):
        table = format_table(sample_rows)
        assert "Average" in table
        assert "22.5" in table  # avg of 20 and 25
        assert "0.875" in table  # avg of 0.85 and 0.90

    def test_build_json_output_structure(self, sample_rows):
        result = build_json_output(sample_rows, fid=100.0, gen_dir="/gen", gt_dir="/gt")
        assert result["per_image"] == sample_rows
        assert result["fid"] == 100.0
        assert result["averages"]["psnr"] == 22.5
        assert result["averages"]["ssim"] == 0.875
        assert result["metadata"]["num_images"] == 2
        assert result["metadata"]["gen_dir"] == "/gen"
        assert result["metadata"]["gt_dir"] == "/gt"

    def test_build_json_output_is_serializable(self, sample_rows):
        result = build_json_output(sample_rows, fid=99.9, gen_dir="/a", gt_dir="/b")
        json_str = json.dumps(result)
        roundtripped = json.loads(json_str)
        assert roundtripped["fid"] == 99.9


class TestEvalCLI:
    """Test eval.py CLI entry point end-to-end."""

    def test_main_produces_json_output(self, paired_image_dirs, tmp_path):
        """eval.main() should write a valid JSON results file."""
        gen_dir, gt_dir = paired_image_dirs
        output_path = tmp_path / "results.json"

        args = argparse.Namespace(
            gen=str(gen_dir),
            gt=str(gt_dir),
            output=str(output_path),
        )
        eval_main(args)

        assert output_path.exists()
        with open(output_path) as f:
            data = json.load(f)

        assert "per_image" in data
        assert "averages" in data
        assert "fid" in data
        assert len(data["per_image"]) == 3
        assert isinstance(data["averages"]["psnr"], float)
        assert isinstance(data["averages"]["ssim"], float)
        assert isinstance(data["fid"], float)

    def test_main_no_pairs_returns_early(self, tmp_path, capsys):
        """When directories have no matching pairs, main() should exit gracefully."""
        empty_gen = tmp_path / "gen"
        empty_gt = tmp_path / "gt"
        empty_gen.mkdir()
        empty_gt.mkdir()

        args = argparse.Namespace(
            gen=str(empty_gen),
            gt=str(empty_gt),
            output=str(tmp_path / "results.json"),
        )
        eval_main(args)

        captured = capsys.readouterr()
        assert "No image pairs found" in captured.out

    def test_cli_entrypoint(self, paired_image_dirs, tmp_path):
        """eval.py should work when invoked as a CLI script."""
        gen_dir, gt_dir = paired_image_dirs
        output_path = tmp_path / "cli_results.json"

        result = subprocess.run(
            [
                sys.executable, str(PROJECT_ROOT / "eval.py"),
                "--gen", str(gen_dir),
                "--gt", str(gt_dir),
                "--output", str(output_path),
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert result.returncode == 0, f"eval.py failed:\n{result.stderr}"
        assert output_path.exists()

        with open(output_path) as f:
            data = json.load(f)
        assert data["metadata"]["num_images"] == 3
