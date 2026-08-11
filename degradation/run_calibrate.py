"""Calibrate IR degradation parameter ranges from a real IR dataset.

Usage:
    python -m degradation.calibrate \\
        --input /data/.../红外船舶数据库_Selected/val/ir \\
        --output degradation/ir_stats.json
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm


def measure_one(path: Path) -> dict[str, float] | None:  # noqa: D401
    """Estimate per-image stats. Returns None on read failure."""
    try:
        with Image.open(path) as im:
            im_l = im.convert("L")
        x = np.asarray(im_l, dtype=np.float32) / 255.0
    except Exception:
        return None

    blur1 = cv2.GaussianBlur(x, (0, 0), sigmaX=1.0, sigmaY=1.0)
    noise_sigma = float((x - blur1).std() * 255.0)

    f = np.fft.fftshift(np.abs(np.fft.fft2(x - x.mean())))
    h, w = f.shape
    cy, cx = h // 2, w // 2
    yy, xx = np.mgrid[0:h, 0:w]
    r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2).astype(np.int32)
    rmax = min(cy, cx)
    radial = np.array([f[r == k].mean() for k in range(1, rmax)])
    radial = np.maximum(radial, 1e-8)
    log_radial = np.log(radial)
    k_axis = np.arange(1, rmax, dtype=np.float32)
    # Slope of log-power vs k^2 gives sigma in pixel space (Gaussian -> Gaussian in FFT).
    slope, _ = np.polyfit(k_axis**2, log_radial, 1)
    blur_sigma = float(np.sqrt(max(1e-6, -1.0 / (2.0 * slope * np.pi**2 / max(h, w) ** 2)))) if slope < 0 else 0.0
    blur_sigma = float(np.clip(blur_sigma, 0.1, 5.0))

    blur_lo = cv2.GaussianBlur(x, (0, 0), sigmaX=20.0, sigmaY=20.0)
    nuc_sigma = float(blur_lo.std() / max(1e-6, blur_lo.mean()))

    row_mean = x.mean(axis=1, keepdims=True)
    fpn_sigma = float((x - row_mean).mean(axis=0).std() * 255.0)

    p2, p98 = np.percentile(x * 255.0, (2, 98))

    return {
        "noise_sigma_8bit": noise_sigma,
        "blur_sigma_px": blur_sigma,
        "nuc_sigma": nuc_sigma,
        "fpn_sigma_8bit": fpn_sigma,
        "p2": float(p2),
        "p98": float(p98),
    }


def percentile_pair(values: list[float], lo: float = 10.0, hi: float = 90.0) -> tuple[float, float]:
    arr = np.asarray([v for v in values if not np.isnan(v)], dtype=np.float32)
    if len(arr) == 0:
        return (0.0, 0.0)
    return (float(np.percentile(arr, lo)), float(np.percentile(arr, hi)))


# Safety caps: estimators conflate scene content (ship structure) with sensor artifacts on
# busy maritime imagery. Caps prevent the calibrated range from blowing past plausible
# IR-sensor physics. (lo_floor, lo_ceil), (hi_floor, hi_ceil).
SAFETY_CAPS: dict[str, tuple[tuple[float, float], tuple[float, float]]] = {
    "noise_sigma_8bit": ((1.5, 4.0), (3.0, 8.0)),
    "blur_sigma_px":    ((0.3, 0.8), (1.0, 2.0)),
    "nuc_sigma":        ((0.02, 0.08), (0.05, 0.15)),
    "fpn_sigma_8bit":   ((0.5, 2.0), (2.0, 6.0)),
}


def apply_caps(key: str, lo: float, hi: float) -> tuple[float, float]:
    if key not in SAFETY_CAPS:
        return (lo, hi)
    (lo_floor, lo_ceil), (hi_floor, hi_ceil) = SAFETY_CAPS[key]
    capped_lo = float(np.clip(lo, lo_floor, lo_ceil))
    capped_hi = float(np.clip(hi, hi_floor, hi_ceil))
    if capped_hi < capped_lo:
        capped_hi = capped_lo
    return (capped_lo, capped_hi)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Folder of real IR images (JPEG).")
    parser.add_argument("--output", required=True, help="Path to write ir_stats.json.")
    parser.add_argument("--n", type=int, default=200, help="Max samples to scan.")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    paths = sorted(Path(args.input).rglob("*.jpg")) + sorted(Path(args.input).rglob("*.jpeg"))
    if not paths:
        raise SystemExit(f"No .jpg/.jpeg files found under {args.input}")
    random.Random(args.seed).shuffle(paths)
    paths = paths[: args.n]

    samples: list[dict[str, float]] = []
    for p in tqdm(paths, desc="Calibrating"):
        m = measure_one(p)
        if m is not None:
            samples.append(m)

    if not samples:
        raise SystemExit("No usable samples.")

    def collect(key: str) -> list[float]:
        return [s[key] for s in samples]

    raw = {
        "noise_sigma_8bit": percentile_pair(collect("noise_sigma_8bit")),
        "blur_sigma_px":    percentile_pair(collect("blur_sigma_px")),
        "nuc_sigma":        percentile_pair(collect("nuc_sigma")),
        "fpn_sigma_8bit":   percentile_pair(collect("fpn_sigma_8bit")),
    }
    stats: dict[str, object] = {key: list(apply_caps(key, *value)) for key, value in raw.items()}
    stats["histogram_p2_p98"] = [float(np.mean(collect("p2"))), float(np.mean(collect("p98")))]
    stats["n_samples"] = len(samples)
    stats["raw_uncapped"] = {key: list(value) for key, value in raw.items()}

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(stats, indent=2))
    print(f"Wrote {out_path}")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
