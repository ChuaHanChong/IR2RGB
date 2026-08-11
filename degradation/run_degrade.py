"""Batch IRDegrader over a folder, preserving subdirectory structure.

Usage:
    python -m degradation.degrade_folder \\
        --input  /path/to/clean_synthetic_ir \\
        --output /path/to/degraded_synthetic_ir \\
        [--seed 0] [--p_per_op 0.7]
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

from degradation import IRDegrader


def _file_seed(base_seed: int, relpath: str) -> int:
    h = hashlib.sha256(f"{base_seed}:{relpath}".encode()).digest()
    return int.from_bytes(h[:4], "big")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--p_per_op", type=float, default=0.7)
    parser.add_argument("--exts", nargs="+", default=[".png", ".jpg", ".jpeg"])
    parser.add_argument(
        "--enable_ops",
        nargs="+",
        default=None,
        help=f"Subset of operators to enable (default: all). Valid: {IRDegrader.OP_NAMES}",
    )
    args = parser.parse_args()

    in_root = Path(args.input)
    out_root = Path(args.output)
    paths = sorted(p for p in in_root.rglob("*") if p.suffix.lower() in args.exts)
    if not paths:
        raise SystemExit(f"No images under {in_root}")

    enabled = set(args.enable_ops) if args.enable_ops else None
    aug = IRDegrader(p_per_op=args.p_per_op, enabled_ops=enabled)

    for src in tqdm(paths, desc="Degrading"):
        rel = src.relative_to(in_root)
        dst = out_root / rel
        if dst.exists():
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        aug.rng = np.random.default_rng(_file_seed(args.seed, str(rel)))
        img = Image.open(src).convert("L")
        aug(img).save(dst)

    print(f"Wrote {len(paths)} images to {out_root}")


if __name__ == "__main__":
    main()
