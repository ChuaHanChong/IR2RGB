# IR Degradation Methods

8 operators modeling the uncooled-microbolometer IR sensor stack.
Apply in physics order; each gated by `p_per_op` and selectable via `enabled_ops`.

```python
from degradation import IRDegrader
aug = IRDegrader(p_per_op=0.5, enabled_ops={"contrast_match", "noise"})
out = aug(pil_l_image)  # PIL "L" -> PIL "L"
```

## Operators (in pipeline order)

Operators are **stacked sequentially** — each modifies `x` in place; the next
sees the updated tensor. Each is independently coin-flipped at `p_per_op`, so
with `p_per_op=0.5` and all 8 enabled, ~4 ops fire per call. Order matters:
`contrast_match` first sets dynamic range, multiplicative artifacts (`nuc`,
`vignette`) precede additive ones (`fpn`, `noise`).

Each operator models a **distinct physical phenomenon** — none are
functionally redundant. Grouped by sensor-stack stage:

| # | Group | Operator | Models | Key param | Stoch | Removable when |
|---|-------|----------|--------|-----------|:---:|----------------|
| 1 | **Dynamic range / ISP** | `contrast_match` | Global p2/p98 alignment to target histogram | `histogram_p2_p98` | no | never (cheapest, only FID-improving op) |
| 5 |  | `highlight_saturation` | Top-percentile soft clip (AGC ceiling on warm objects) | `highlight_pivot_pct`, `highlight_strength` | yes | target shows no AGC clipping |
| 2 | **Spatial blur (MTF)** | `blur` | Lens PSF — continuous Gaussian convolution | `blur_sigma_px` | yes | LoRA already matches sharpness |
| 4 | **Multiplicative gain** | `nuc_residual` | Random blotchy bolometer gain (multi-octave) | `nuc_sigma` | yes | cooled InSb / non-bolometer sensors |
| 3 |  | `vignette` | Radial darkening (cold-shield aperture) | `vignette_strength` | yes | cropped/center-square datasets |
| 6 | **Additive noise** | `fpn` | Column/row readout offsets — *structured* | `fpn_sigma_8bit` | yes | camera does on-board column correction |
| 8 |  | `sensor_noise` | Pixel Gaussian — *unstructured* | `noise_sigma_8bit` | yes | rarely (universal sensor effect) |
| 7 |  | `dead_hot_pixels` | Sparse stuck-low/high pixels | `dead_pixel_rate` | yes | factory-calibrated sensors |

Internally `np.float32` in `[0, 1]`; clipped + cast to uint8 at the end.

## Calibration

```bash
python -m degradation.run_calibrate --input <real-ir-dir> --output degradation/ir_stats.json --n 700
```

Writes capped ranges to `ir_stats.json`. Re-run when the target dataset changes.
Estimators are noisy on busy scenes (vessels), so the calibrator caps each
parameter to safe bounds; raw uncapped values are kept under `raw_uncapped`.

## Validated recipes (`exp-015_lora` vs `红外船舶数据库/val`, 700 imgs, FID-2048)

| Goal | Config | FID | Δ vs clean |
|------|--------|-----|------------|
| **Best match** | `enabled_ops={"contrast_match"}, p=1.0` | **59.65** | **−2.13** |
| **SSL aug (best)** | `enabled_ops={"contrast_match","noise","blur"}, p=0.5` | **61.08** | **−0.70** |
| SSL aug (FID-neutral) | `enabled_ops={"contrast_match","noise"}, p=0.5` | 61.94 | +0.15 |
| clean (no degradation) | — | 61.78 | 0 |
| Real-vs-real floor | (50/50 split) | 63.24 | — |
| Light noise only | `enabled_ops={"noise"}, p=0.5` | 62.78 | +1.00 |
| All 8 ops @ p=0.5 | `enabled_ops=None, p=0.5` | 107.69 | +45.91 |
| All 8 ops @ p=0.7 | `enabled_ops=None, p=0.7` | 137.31 | +75.53 |

Recipe is **specific to this pipeline**. Re-run `run_ablation.py` for any
different LoRA, raw FLUX output, or different real-IR target.

Operators that hurt FID most here: `fpn`, `dead_pixels` — they add structural
artifacts the LoRA correctly avoids. (`jpeg`, `resolution_loss`, `quantize`
were removed: input is already JPEG/256×256/8-bit, so they were no-ops or
counterproductive double-passes.)

## Files

| File | Purpose |
|------|---------|
| `degrader.py` | `IRDegrader` class |
| `run_calibrate.py` | Calibrate stats from real IR |
| `run_degrade.py` | Batch degrade a folder |
| `run_validate.py` | FID + visual grid (`--n_variants` shows variety) |
| `run_ablation.py` | Sweep configs, rank by FID |
| `test_degrader.py` | pytest (shape, determinism, histogram) |
| `ir_stats.json` | Calibrated parameter ranges |
