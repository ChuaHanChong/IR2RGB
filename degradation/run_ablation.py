"""Sweep degradation configs and report FID against real IR."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchmetrics.image import FrechetInceptionDistance
from torchvision import transforms
from tqdm import tqdm

from degradation import IRDegrader

_to_float = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])

CONFIGS: list[dict] = [
    {"name": "contrast_match_only", "p": 1.0, "ops": ["contrast_match"]},
    {"name": "contrast_match_noise_p0.5", "p": 0.5, "ops": ["contrast_match", "noise"]},
    {"name": "contrast_match_noise_blur_p0.5", "p": 0.5, "ops": ["contrast_match", "noise", "blur"]},
    {"name": "noise_only_p0.5", "p": 0.5, "ops": ["noise"]},
    {"name": "all_8_ops_p0.5", "p": 0.5, "ops": None},
    {"name": "all_8_ops_p0.7", "p": 0.7, "ops": None},
]


def _file_seed(base: int, rel: str) -> int:
    return int.from_bytes(hashlib.sha256(f"{base}:{rel}".encode()).digest()[:4], "big")


def degrade_dir(in_root: Path, out_root: Path, p: float, ops: list[str] | None, base_seed: int = 0) -> None:
    if out_root.exists():
        shutil.rmtree(out_root)
    aug = IRDegrader(p_per_op=p, enabled_ops=set(ops) if ops else None)
    paths = sorted(p_ for p_ in in_root.rglob("*.jpg"))
    for src in tqdm(paths, desc=out_root.name, leave=False):
        rel = src.relative_to(in_root)
        dst = out_root / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        aug.rng = np.random.default_rng(_file_seed(base_seed, str(rel)))
        aug(Image.open(src).convert("L")).save(dst)


def fid(a_root: Path, b_root: Path, device: str) -> float:
    fid_m = FrechetInceptionDistance(feature=2048, normalize=True).to(device)
    for p_ in sorted(a_root.rglob("*.jpg")):
        fid_m.update(_to_float(Image.open(p_).convert("RGB")).unsqueeze(0).to(device), real=True)
    for p_ in sorted(b_root.rglob("*.jpg")):
        fid_m.update(_to_float(Image.open(p_).convert("RGB")).unsqueeze(0).to(device), real=False)
    return float(fid_m.compute().item())


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--clean_root", required=True)
    p.add_argument("--real_root", required=True)
    p.add_argument("--work_root", required=True, help="Where ablation outputs are written.")
    p.add_argument("--report", required=True)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    clean_root = Path(args.clean_root)
    real_root = Path(args.real_root)
    work_root = Path(args.work_root)
    work_root.mkdir(parents=True, exist_ok=True)

    print("Computing baselines...")
    fid_clean = fid(real_root, clean_root, args.device)
    print(f"  fid_clean (no degradation) = {fid_clean:.3f}")

    rows = [{"name": "clean (no degradation)", "fid": fid_clean, "p": None, "ops": None}]
    for cfg in CONFIGS:
        out = work_root / cfg["name"]
        print(f"=== {cfg['name']} (p={cfg['p']}, ops={cfg['ops']}) ===")
        degrade_dir(clean_root, out, cfg["p"], cfg["ops"])
        f = fid(real_root, out, args.device)
        print(f"  fid = {f:.3f} (delta vs clean: {f - fid_clean:+.3f})")
        rows.append({"name": cfg["name"], "p": cfg["p"], "ops": cfg["ops"] or "all", "fid": f, "delta_vs_clean": f - fid_clean})

    Path(args.report).write_text(json.dumps({"results": rows}, indent=2))
    print("\n=== Summary (sorted by FID, lower=better) ===")
    for r in sorted(rows, key=lambda x: x["fid"]):
        delta = r.get("delta_vs_clean", 0.0)
        flag = " ★" if r["name"] == "clean (no degradation)" else ""
        print(f"  {r['fid']:7.3f}  {r['name']:35s} (Δ vs clean {delta:+7.3f}){flag}")


if __name__ == "__main__":
    main()
