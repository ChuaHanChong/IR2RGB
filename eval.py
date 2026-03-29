#!/usr/bin/env python3
"""Evaluate IR image reconstruction quality against ground-truth IR images."""

import argparse
import json
import warnings
from pathlib import Path

import torch
from PIL import Image
from tabulate import tabulate
from torchmetrics.image import (
    FrechetInceptionDistance,
    PeakSignalNoiseRatio,
    StructuralSimilarityIndexMeasure,
)
from torchvision import transforms

# Shared resize + to-tensor pipeline: PIL → float32 [0,1] [C,H,W] at 256×256
_to_float = transforms.Compose(
    [
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ]
)


# ---------------------------------------------------------------------------
# Image pair discovery
# ---------------------------------------------------------------------------


def find_image_pairs(gen_dir: str, gt_dir: str) -> list[tuple[str, str]]:
    """Return sorted list of (gen_path, gt_path) pairs matched by filename stem."""
    gen_path = Path(gen_dir)
    gt_path = Path(gt_dir)

    if not gen_path.is_dir():
        raise ValueError(f"Generated image directory not found: {gen_dir}")
    if not gt_path.is_dir():
        raise ValueError(f"Ground-truth directory not found: {gt_dir}")

    gen_files = {f.stem: f for f in sorted(gen_path.rglob("*.jpg"))}
    gt_files = {f.stem: f for f in sorted(gt_path.rglob("*.jpg"))}

    pairs = []
    for stem, gen_file in sorted(gen_files.items()):
        if stem in gt_files:
            pairs.append((str(gen_file), str(gt_files[stem])))
        else:
            warnings.warn(f"No GT match for {gen_file.name} — skipping.")

    for stem in sorted(gt_files.keys() - gen_files.keys()):
        warnings.warn(f"No generated image for GT {gt_files[stem].name} — skipping.")

    return pairs


# ---------------------------------------------------------------------------
# Image loading
# ---------------------------------------------------------------------------


def load_image_float(path: str) -> torch.Tensor:
    """Load image as float32 tensor [3, 256, 256] with values in [0, 1]."""
    img = Image.open(path).convert("RGB")
    return _to_float(img)


# ---------------------------------------------------------------------------
# Per-image metrics
# ---------------------------------------------------------------------------


def compute_psnr(gen_path: str, gt_path: str, device: str) -> float:
    """Compute PSNR between a generated image and its ground truth."""
    gen_b = load_image_float(gen_path).unsqueeze(0).to(device)
    gt_b = load_image_float(gt_path).unsqueeze(0).to(device)
    return PeakSignalNoiseRatio(data_range=1.0).to(device)(gen_b, gt_b).item()


def compute_ssim(gen_path: str, gt_path: str, device: str) -> float:
    """Compute SSIM between a generated image and its ground truth."""
    gen_b = load_image_float(gen_path).unsqueeze(0).to(device)
    gt_b = load_image_float(gt_path).unsqueeze(0).to(device)
    return StructuralSimilarityIndexMeasure(data_range=1.0).to(device)(gen_b, gt_b).item()


# ---------------------------------------------------------------------------
# Dataset-level metric
# ---------------------------------------------------------------------------


def compute_fid(pairs: list[tuple[str, str]], device: str) -> float:
    """Compute FID between the full set of generated and ground-truth images."""
    fid_metric = FrechetInceptionDistance(feature=2048, normalize=True).to(device)

    for gen_path, gt_path in pairs:
        gen_t = load_image_float(gen_path).unsqueeze(0).to(device)
        gt_t = load_image_float(gt_path).unsqueeze(0).to(device)
        fid_metric.update(gt_t, real=True)
        fid_metric.update(gen_t, real=False)

    return fid_metric.compute().item()


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------


def format_table(rows: list[dict]) -> str:
    """Format per-image metrics as a human-readable table string with an Average row."""
    metric_keys = ["psnr", "ssim"]
    averages = {}
    for key in metric_keys:
        values = [r[key] for r in rows if r.get(key) is not None]
        averages[key] = sum(values) / len(values) if values else float("nan")

    headers = ["Generated Image", "Ground-Truth Image", "PSNR (dB)", "SSIM"]
    table_rows = []
    for r in rows:
        table_rows.append(
            [
                r["gen"],
                r["gt"],
                f"{r['psnr']:.4f}",
                f"{r['ssim']:.4f}",
            ]
        )

    table_rows.append(
        [
            "Average",
            "",
            f"{averages['psnr']:.4f}",
            f"{averages['ssim']:.4f}",
        ]
    )

    return tabulate(table_rows, headers=headers, tablefmt="github")


def build_json_output(
    rows: list[dict],
    fid: float,
    gen_dir: str,
    gt_dir: str,
) -> dict:
    """Build the full JSON-serialisable output dictionary."""
    metric_keys = ["psnr", "ssim"]
    averages = {}
    for key in metric_keys:
        values = [r[key] for r in rows if r.get(key) is not None]
        averages[key] = sum(values) / len(values) if values else None

    return {
        "per_image": rows,
        "averages": averages,
        "fid": fid,
        "metadata": {
            "gen_dir": gen_dir,
            "gt_dir": gt_dir,
            "num_images": len(rows),
        },
    }


# ---------------------------------------------------------------------------
# Main execution
# ---------------------------------------------------------------------------


def main(args: argparse.Namespace) -> None:
    device = "cuda"
    print(f"Using device: {device}")

    pairs = find_image_pairs(args.gen, args.gt)
    if not pairs:
        print("No image pairs found. Check --gen and --gt directories.")
        return

    print(f"Found {len(pairs)} image pairs. Computing metrics...\n")

    rows = []
    for gen_path, gt_path in pairs:
        print(f"  Evaluating {Path(gen_path).name} ...", flush=True)

        psnr_val = compute_psnr(gen_path, gt_path, device)
        ssim_val = compute_ssim(gen_path, gt_path, device)

        rows.append(
            {
                "gen": gen_path,
                "gt": gt_path,
                "psnr": psnr_val,
                "ssim": ssim_val,
            }
        )

    fid_val = compute_fid(pairs, device)

    print("\nPer-Image Results")
    print("=" * 80)
    print(format_table(rows))

    print("\nDataset-Level Metric")
    print("=" * 80)
    print(f"FID: {fid_val:.4f}")

    output = build_json_output(rows, fid_val, args.gen, args.gt)
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate IR image reconstruction quality using TorchMetrics.")
    parser.add_argument("--gen", required=True, help="Folder of generated IR images.")
    parser.add_argument("--gt", required=True, help="Folder of ground-truth IR images.")
    parser.add_argument(
        "--output",
        default="eval_results.json",
        help="Path to save JSON results (default: eval_results.json).",
    )

    args = parser.parse_args()
    main(args)
