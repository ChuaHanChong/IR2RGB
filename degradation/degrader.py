"""Physics-faithful IR degradation augmentation for DINOv2 SSL.

Operators (in physics order): contrast_match -> blur -> vignette -> NUC ->
highlight_saturation -> FPN -> dead/hot pixels -> sensor noise.
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image

DEFAULT_RANGES: dict[str, tuple[float, float]] = {
    "blur_sigma_px": (0.3, 1.5),
    "vignette_strength": (0.0, 0.25),
    "nuc_sigma": (0.02, 0.08),
    "highlight_pivot_pct": (0.80, 0.95),  # percentile above which highlights compress
    "highlight_strength": (0.3, 0.8),  # 0=no compression, 1=hard clip to 1.0
    "fpn_sigma_8bit": (1.0, 4.0),
    "dead_pixel_rate": (1e-5, 1e-4),
    "noise_sigma_8bit": (2.0, 6.0),
    "histogram_p2_p98": (0, 255),
}


class IRDegrader(torch.nn.Module):
    """Stochastic IR sensor degradation as a torchvision transform."""

    OP_NAMES = (
        "contrast_match",
        "blur",
        "vignette",
        "nuc",
        "highlight",
        "fpn",
        "dead_pixels",
        "noise",
    )

    def __init__(
        self,
        stats_path: str | Path | None = None,
        p_per_op: float = 0.7,
        seed: int | None = None,
        enabled_ops: set[str] | None = None,
    ) -> None:
        super().__init__()
        self.p_per_op = p_per_op
        self.rng = np.random.default_rng(seed)
        self.ranges = self._load_ranges(stats_path)
        self.enabled_ops = set(self.OP_NAMES) if enabled_ops is None else set(enabled_ops)
        unknown = self.enabled_ops - set(self.OP_NAMES)
        if unknown:
            raise ValueError(f"Unknown ops: {unknown}. Valid: {self.OP_NAMES}")
        self._rgb_warning_emitted = False

    @staticmethod
    def _load_ranges(stats_path: str | Path | None) -> dict[str, tuple[float, float]]:
        ranges = dict(DEFAULT_RANGES)
        path = Path(stats_path) if stats_path is not None else Path(__file__).parent / "ir_stats.json"
        if not path.exists():
            warnings.warn(
                f"IRDegrader: stats file {path} not found; using built-in defaults.",
                stacklevel=2,
            )
            return ranges
        loaded: dict[str, Any] = json.loads(path.read_text())
        for key, value in loaded.items():
            if key in ranges and isinstance(value, (list, tuple)) and len(value) == 2:
                ranges[key] = (float(value[0]), float(value[1]))
        return ranges

    def _u(self, key: str) -> float:
        lo, hi = self.ranges[key]
        return float(self.rng.uniform(lo, hi))

    def _coin(self) -> bool:
        return bool(self.rng.random() < self.p_per_op)

    def forward(self, img: Image.Image) -> Image.Image:
        if img.mode != "L":
            if not self._rgb_warning_emitted:
                warnings.warn(
                    f"IRDegrader: input mode is {img.mode!r}; auto-converting to 'L'.",
                    stacklevel=2,
                )
                self._rgb_warning_emitted = True
            img = img.convert("L")

        x = np.asarray(img, dtype=np.float32) / 255.0
        h, w = x.shape

        if "contrast_match" in self.enabled_ops and self._coin():
            x = self._contrast_match(x)
        if "blur" in self.enabled_ops and self._coin():
            x = self._optical_blur(x)
        if "vignette" in self.enabled_ops and self._coin():
            x = self._vignetting(x, h, w)
        if "nuc" in self.enabled_ops and self._coin():
            x = self._nuc_residual(x, h, w)
        if "highlight" in self.enabled_ops and self._coin():
            x = self._highlight_saturation(x)
        if "fpn" in self.enabled_ops and self._coin():
            x = self._fpn(x, h, w)
        if "dead_pixels" in self.enabled_ops and self._coin():
            x = self._dead_hot_pixels(x)
        if "noise" in self.enabled_ops and self._coin():
            x = self._sensor_noise(x)

        x = np.clip(x, 0.0, 1.0)
        return Image.fromarray((x * 255.0 + 0.5).astype(np.uint8), mode="L")

    def _contrast_match(self, x: np.ndarray) -> np.ndarray:
        # Linearly map x's [p2, p98] to the calibrated real-IR [p2, p98].
        # Compresses dynamic range to match real distribution. Deterministic per image.
        tgt_p2, tgt_p98 = self.ranges["histogram_p2_p98"]
        tgt_p2 /= 255.0
        tgt_p98 /= 255.0
        src_p2, src_p98 = np.percentile(x, [2, 98])
        if src_p98 - src_p2 < 1e-3:
            return x
        return tgt_p2 + (x - src_p2) / (src_p98 - src_p2) * (tgt_p98 - tgt_p2)

    def _optical_blur(self, x: np.ndarray) -> np.ndarray:
        sigma = self._u("blur_sigma_px")
        ksize = max(3, int(2 * round(3 * sigma) + 1))
        return cv2.GaussianBlur(x, (ksize, ksize), sigmaX=sigma, sigmaY=sigma)

    def _vignetting(self, x: np.ndarray, h: int, w: int) -> np.ndarray:
        strength = self._u("vignette_strength")
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
        r = np.sqrt(((yy - cy) / cy) ** 2 + ((xx - cx) / cx) ** 2)
        r = np.clip(r, 0.0, 1.0)
        mask = 1.0 - strength * (1.0 - np.cos(r * np.pi / 2.0))
        return x * mask.astype(np.float32)

    def _nuc_residual(self, x: np.ndarray, h: int, w: int) -> np.ndarray:
        sigma = self._u("nuc_sigma")
        gain = np.zeros((h, w), dtype=np.float32)
        for grid, amp in [(4, 1.0), (8, 0.5), (16, 0.25)]:
            small = self.rng.normal(0.0, 1.0, size=(grid, grid)).astype(np.float32)
            gain += amp * cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)
        gain *= sigma / (gain.std() + 1e-8)
        return x * (1.0 + gain)

    def _highlight_saturation(self, x: np.ndarray) -> np.ndarray:
        # Mimics IR camera AGC: warm objects (high pixel values) lose internal detail
        # because the 8-bit dynamic range is too narrow for the full thermal contrast.
        pivot_pct = self._u("highlight_pivot_pct")
        strength = self._u("highlight_strength")
        pivot = float(np.quantile(x, pivot_pct))
        if pivot >= 1.0 - 1e-3:
            return x
        out = x.copy()
        above = x > pivot
        if not above.any():
            return out
        # Compress [pivot, 1.0] -> [pivot + (1-pivot)*strength, 1.0].
        new_lo = pivot + (1.0 - pivot) * strength
        out[above] = new_lo + (1.0 - new_lo) * (x[above] - pivot) / (1.0 - pivot)
        return out

    def _fpn(self, x: np.ndarray, h: int, w: int) -> np.ndarray:
        sigma = self._u("fpn_sigma_8bit") / 255.0
        col = self.rng.normal(0.0, sigma, size=(1, w)).astype(np.float32)
        row = self.rng.normal(0.0, sigma * 0.5, size=(h, 1)).astype(np.float32)
        return x + col + row

    def _dead_hot_pixels(self, x: np.ndarray) -> np.ndarray:
        rate = self._u("dead_pixel_rate")
        mask = self.rng.random(x.shape) < rate
        if not mask.any():
            return x
        out = x.copy()
        # half dead (0), half hot (1)
        hot = mask & (self.rng.random(x.shape) < 0.5)
        dead = mask & ~hot
        out[hot] = 1.0
        out[dead] = 0.0
        return out

    def _sensor_noise(self, x: np.ndarray) -> np.ndarray:
        sigma = self._u("noise_sigma_8bit") / 255.0
        return x + self.rng.normal(0.0, sigma, size=x.shape).astype(np.float32)
