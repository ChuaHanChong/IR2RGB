"""Validate that degraded synthetic IR matches real IR via FID + visual grid.

Computes:
  - fid_clean_overall, fid_degraded_overall (700 imgs each)
  - per_category_fids (×7)
  - fid_baseline (real-vs-real, random 50/50 split)

Saves match_grid.png with 14 rows (2/cat) × 5 cols [RGB | gray | clean | degraded | real].
Writes fid_report.json.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from torchmetrics.image import FrechetInceptionDistance
from torchvision import transforms

from degradation import IRDegrader


def _contrast_stretch(img: Image.Image, p_lo: float = 2.0, p_hi: float = 98.0) -> Image.Image:
    """Per-image p2/p98 percentile stretch to fill [0, 255]."""
    arr = np.asarray(img.convert("L"), dtype=np.float32)
    lo, hi = np.percentile(arr, (p_lo, p_hi))
    if hi <= lo:
        return img
    arr = np.clip((arr - lo) / (hi - lo) * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(arr, mode="L").convert("RGB")

_to_float = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])


def _load_to_3ch(path: Path) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    return _to_float(img)


def _fid(pairs_a: list[Path], pairs_b: list[Path], device: str) -> float:
    fid = FrechetInceptionDistance(feature=2048, normalize=True).to(device)
    for p in pairs_a:
        fid.update(_load_to_3ch(p).unsqueeze(0).to(device), real=True)
    for p in pairs_b:
        fid.update(_load_to_3ch(p).unsqueeze(0).to(device), real=False)
    return float(fid.compute().item())


def _list_imgs(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*.jpg"))


def _list_imgs_in_cat(root: Path, cat: str) -> list[Path]:
    return sorted((root / cat).rglob("*.jpg"))


def _load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    ]
    for path in candidates:
        if Path(path).exists():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def _build_grid(
    rgb_root: Path,
    clean_root: Path,
    degraded_root: Path,  # kept for back-compat, ignored when n_variants > 0
    real_root: Path,
    out_path: Path,
    n_per_cat: int = 1,
    cell: int = 160,
    n_variants: int = 6,
    aug_ops: list[str] | None = None,
    aug_p: float = 0.5,
    p_sweep: list[float] | None = None,
    seeds: list[int] | None = None,
) -> None:
    """Grid showing augmentation variety per source.

    Columns: [Source RGB | Clean LoRA | Real IR (stretched) | Variant 1 .. Variant N]
    Variation mode: if `p_sweep` is provided, columns vary p_per_op (fixed seed=0).
    Otherwise columns vary the seed (fixed p_per_op=aug_p).
    """
    enabled = set(aug_ops) if aug_ops else None
    if p_sweep:
        ps = list(p_sweep)
        seed_list = [0] * len(ps)
        headers_v = [f"p={p:.2f}" for p in ps]
        cfg_label = f"ops={aug_ops or 'all'}  p sweep across columns (seed=0)"
    else:
        seed_list = list(seeds) if seeds else list(range(n_variants))
        ps = [aug_p] * len(seed_list)
        headers_v = [f"seed={s}" for s in seed_list]
        cfg_label = f"ops={aug_ops or 'all'} p={aug_p}  seed sweep across columns"
    headers = ["Source RGB", "Clean LoRA", "Real IR (stretch)"] + headers_v

    cats = sorted(d.name for d in real_root.iterdir() if d.is_dir())
    rng = random.Random(0)
    rows: list[tuple[str, list[Image.Image]]] = []
    for cat in cats:
        files = sorted(p.stem for p in (real_root / cat).glob("*.jpg"))
        picked = rng.sample(files, min(n_per_cat, len(files)))
        for stem in picked:
            rgb = Image.open(rgb_root / cat / f"{stem}.jpg").convert("RGB").resize((cell, cell))
            clean_pil = Image.open(clean_root / cat / f"{stem}.jpg").convert("L")
            clean = clean_pil.resize((cell, cell)).convert("RGB")
            real_raw = Image.open(real_root / cat / f"{stem}.jpg")
            real = _contrast_stretch(real_raw).resize((cell, cell))

            variants: list[Image.Image] = []
            for col, (p, s) in enumerate(zip(ps, seed_list)):
                aug = IRDegrader(p_per_op=p, enabled_ops=enabled)
                aug.rng = np.random.default_rng(s)
                v = aug(clean_pil).resize((cell, cell)).convert("RGB")
                variants.append(v)
            rows.append((f"cat {cat} / {stem}", [rgb, clean, real] + variants))

    n_rows, n_cols = len(rows), len(headers)
    pad = 3
    header_h = 36
    title_h = 28
    label_w = 132
    grid_w = label_w + n_cols * cell + (n_cols + 1) * pad
    grid_h = title_h + header_h + n_rows * cell + (n_rows + 1) * pad
    canvas = Image.new("RGB", (grid_w, grid_h), color=(255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    font_t = _load_font(15)
    font_h = _load_font(14)
    font_l = _load_font(12)

    draw.text((10, 6), f"IRDegrader augmentation variety  |  {cfg_label}", fill=(0, 0, 0), font=font_t)
    for c, text in enumerate(headers):
        x = label_w + pad + c * (cell + pad)
        draw.text((x + 4, title_h + 8), text, fill=(0, 0, 0), font=font_h)

    for r, (label, row) in enumerate(rows):
        y = title_h + header_h + pad + r * (cell + pad)
        draw.text((6, y + cell // 2 - 8), label, fill=(60, 60, 60), font=font_l)
        for c, im in enumerate(row):
            x = label_w + pad + c * (cell + pad)
            canvas.paste(im, (x, y))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)
    print(f"Wrote {out_path}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--rgb_root", required=True)
    p.add_argument("--clean_root", required=True)
    p.add_argument("--degraded_root", required=True)
    p.add_argument("--real_root", required=True)
    p.add_argument("--report", required=True)
    p.add_argument("--grid", required=True)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--grid_only", action="store_true", help="Skip FID, just rebuild the visual grid.")
    p.add_argument("--n_variants", type=int, default=6, help="Augmentation variants per source in grid.")
    p.add_argument("--aug_ops", nargs="+", default=None, help="Operators to enable for variety grid (default: all).")
    p.add_argument("--aug_p", type=float, default=0.5, help="p_per_op for variety grid (when not using --p_sweep).")
    p.add_argument("--p_sweep", nargs="+", type=float, default=None, help="If set, vary p across columns instead of seed.")
    p.add_argument("--seeds", nargs="+", type=int, default=None, help="Explicit seed list for seed sweep (default: 0..n_variants-1).")
    args = p.parse_args()

    clean_root = Path(args.clean_root)
    degraded_root = Path(args.degraded_root)
    real_root = Path(args.real_root)
    rgb_root = Path(args.rgb_root)

    if args.grid_only:
        _build_grid(
            rgb_root, clean_root, degraded_root, real_root, Path(args.grid),
            n_variants=args.n_variants, aug_ops=args.aug_ops, aug_p=args.aug_p,
            p_sweep=args.p_sweep, seeds=args.seeds,
        )
        return

    real_files = _list_imgs(real_root)
    clean_files = _list_imgs(clean_root)
    degraded_files = _list_imgs(degraded_root)
    print(f"real={len(real_files)} clean={len(clean_files)} degraded={len(degraded_files)}")

    print("Computing fid_clean_overall...")
    fid_clean = _fid(real_files, clean_files, args.device)
    print(f"  -> {fid_clean:.3f}")

    print("Computing fid_degraded_overall...")
    fid_degraded = _fid(real_files, degraded_files, args.device)
    print(f"  -> {fid_degraded:.3f}")

    print("Computing fid_baseline (real vs real, random 50/50)...")
    rng = random.Random(0)
    shuffled = list(real_files)
    rng.shuffle(shuffled)
    half = len(shuffled) // 2
    fid_baseline = _fid(shuffled[:half], shuffled[half:], args.device)
    print(f"  -> {fid_baseline:.3f}")

    cats = sorted(d.name for d in real_root.iterdir() if d.is_dir())
    per_cat: dict[str, dict[str, float]] = {}
    for cat in cats:
        rf = _list_imgs_in_cat(real_root, cat)
        cf = _list_imgs_in_cat(clean_root, cat)
        df = _list_imgs_in_cat(degraded_root, cat)
        if not (rf and cf and df):
            continue
        print(f"Computing per-category FID cat={cat}...")
        per_cat[cat] = {
            "fid_clean": _fid(rf, cf, args.device),
            "fid_degraded": _fid(rf, df, args.device),
            "n": len(rf),
        }

    report = {
        "fid_clean_overall": fid_clean,
        "fid_degraded_overall": fid_degraded,
        "fid_baseline_real_vs_real": fid_baseline,
        "per_category": per_cat,
        "improvement_overall": fid_clean - fid_degraded,
    }
    Path(args.report).write_text(json.dumps(report, indent=2))
    print(f"Wrote {args.report}")
    print(json.dumps(report, indent=2))

    _build_grid(
        rgb_root, clean_root, degraded_root, real_root, Path(args.grid),
        n_variants=args.n_variants, aug_ops=args.aug_ops, aug_p=args.aug_p,
        p_sweep=args.p_sweep, seeds=args.seeds,
    )


if __name__ == "__main__":
    main()
