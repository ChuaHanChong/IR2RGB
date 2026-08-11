"""Tests for IRDegrader.

Run unit tests:
    pytest degradation/test_ir_degrade.py -v

Visual smoke (saves 8 augmented samples to /tmp/ir_aug_smoke):
    python degradation/test_ir_degrade.py --smoke
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from degradation import IRDegrader


def _make_input(size: int = 256) -> Image.Image:
    rng = np.random.default_rng(0)
    arr = rng.integers(64, 192, size=(size, size), dtype=np.uint8)
    return Image.fromarray(arr, mode="L")


def test_shape_and_mode_preserved() -> None:
    aug = IRDegrader(seed=0)
    out = aug(_make_input(256))
    assert out.mode == "L"
    assert out.size == (256, 256)


def test_determinism_with_seed() -> None:
    img = _make_input(128)
    out_a = np.asarray(IRDegrader(seed=42)(img))
    out_b = np.asarray(IRDegrader(seed=42)(img))
    assert np.array_equal(out_a, out_b)


def test_histogram_in_envelope() -> None:
    aug = IRDegrader(seed=0)
    means = []
    for _ in range(50):
        out = aug(_make_input(128))
        means.append(np.asarray(out).mean())
    assert 16.0 < float(np.mean(means)) < 240.0


def smoke(out_dir: Path = Path("/tmp/ir_aug_smoke"), n: int = 8) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    aug = IRDegrader()
    src = _make_input(256)
    src.save(out_dir / "input.png")
    for i in range(n):
        aug.rng = np.random.default_rng(i)
        aug(src).save(out_dir / f"aug_{i:02d}.png")
    print(f"Wrote {n + 1} samples to {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--out", default="/tmp/ir_aug_smoke")
    args = parser.parse_args()
    if args.smoke:
        smoke(Path(args.out))
    else:
        pytest.main([__file__, "-v"])
