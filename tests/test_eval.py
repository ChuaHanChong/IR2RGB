"""Tests for eval.py — CLI entry point and core metric functions."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

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
        Image.fromarray(arr).save(gt_dir / name)
        noisy = np.clip(arr.astype(np.int16) + rng.randint(-20, 20, arr.shape), 0, 255).astype(np.uint8)
        Image.fromarray(noisy).save(gen_dir / name)

    return gen_dir, gt_dir


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
