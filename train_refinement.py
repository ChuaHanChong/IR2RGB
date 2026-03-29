"""Train a lightweight U-Net refinement network for two-stage pixel refinement.

# [ml-opt] two-stage-pixel-refinement
# Takes FLUX LoRA output (grayscale 256x256) paired with grayscale condition
# and refines it toward ground-truth IR.
# Loss: L1 + SSIM + optional LPIPS (perceptual).
# Supports hard-category oversampling and focal per-sample weighting.
#
# Architecture: 4-level U-Net with skip connections, GroupNorm, SiLU.
# Uses residual learning: output = flux_input + predicted_residual.
# Dual-input: concatenates FLUX output + grayscale condition (2ch -> 1ch).
# Concept adapted from DiffBIR (ECCV 2024): restoration module + diffusion prior.

Usage:
    python train_refinement.py \\
        --flux_dir /path/to/flux_outputs \\
        --gt_dir /path/to/gt_ir \\
        --output_dir /path/to/refinement_model \\
        --epochs 100 \\
        --batch_size 16 \\
        --lr 1e-4

    # With grayscale condition input and hard-category oversampling:
    python train_refinement.py \\
        --flux_dir /path/to/flux_outputs \\
        --gt_dir /path/to/gt_ir \\
        --cond_dir /path/to/grayscale_inputs \\
        --output_dir /path/to/refinement_model \\
        --oversample_map '{"1":3,"5":3,"4":2}' \\
        --use_lpips \\
        --focal_gamma 1.0
"""

import argparse
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler


# ---------------------------------------------------------------------------
# SSIM Loss (differentiable, custom implementation)
# ---------------------------------------------------------------------------
def _gaussian_window(size, sigma):
    """Create 2D Gaussian window for SSIM computation."""
    coords = torch.arange(size, dtype=torch.float32) - size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    return g.unsqueeze(0) * g.unsqueeze(1)


class SSIMLoss(nn.Module):
    """Differentiable SSIM loss (1 - SSIM).

    Supports multi-channel input by applying per-channel SSIM and averaging.
    """

    def __init__(self, window_size=11, sigma=1.5):
        super().__init__()
        self.window_size = window_size
        window = _gaussian_window(window_size, sigma)
        # [1, 1, window_size, window_size]
        self.register_buffer("window", window.unsqueeze(0).unsqueeze(0))

    def forward(self, pred, target):
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2
        channels = pred.shape[1]
        # Expand window to match channel count
        window = self.window.expand(channels, -1, -1, -1)

        pad = self.window_size // 2
        mu_pred = F.conv2d(pred, window, padding=pad, groups=channels)
        mu_target = F.conv2d(target, window, padding=pad, groups=channels)

        mu_pred_sq = mu_pred ** 2
        mu_target_sq = mu_target ** 2
        mu_cross = mu_pred * mu_target

        sigma_pred_sq = F.conv2d(pred ** 2, window, padding=pad, groups=channels) - mu_pred_sq
        sigma_target_sq = F.conv2d(target ** 2, window, padding=pad, groups=channels) - mu_target_sq
        sigma_cross = F.conv2d(pred * target, window, padding=pad, groups=channels) - mu_cross

        ssim_map = ((2 * mu_cross + C1) * (2 * sigma_cross + C2)) / \
                   ((mu_pred_sq + mu_target_sq + C1) * (sigma_pred_sq + sigma_target_sq + C2))

        return 1.0 - ssim_map.mean()


# ---------------------------------------------------------------------------
# Lightweight U-Net Architecture
# ---------------------------------------------------------------------------
class ConvBlock(nn.Module):
    """Double convolution block with GroupNorm and SiLU activation."""
    # [ml-opt] two-stage-pixel-refinement: SiLU activation (research recommends SiLU over GELU)

    def __init__(self, in_ch, out_ch, dropout=0.0):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.GroupNorm(min(8, out_ch), out_ch),
            nn.SiLU(inplace=True),
            nn.Dropout2d(dropout) if dropout > 0 else nn.Identity(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.GroupNorm(min(8, out_ch), out_ch),
            nn.SiLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class RefinementUNet(nn.Module):
    """Lightweight U-Net for image refinement.

    # [ml-opt] two-stage-pixel-refinement
    Architecture: 4 encoder + 4 decoder blocks with skip connections.
    Input: 1 or 2 channels (FLUX output, optionally + grayscale condition).
    Output: 1 channel (refined image).
    Uses residual learning: output = flux_input + predicted_residual.
    Concept from DiffBIR (ECCV 2024): separate pixel-level refinement after diffusion.

    Args:
        in_channels: Number of input channels (1 = FLUX only, 2 = FLUX + condition).
        base_ch: Base channel width. Total params scale quadratically with this.
        dropout: Dropout rate for regularization.
    """

    def __init__(self, in_channels=1, base_ch=32, dropout=0.0):
        super().__init__()
        self.in_channels = in_channels

        # Encoder
        self.enc1 = ConvBlock(in_channels, base_ch, dropout)       # 256
        self.enc2 = ConvBlock(base_ch, base_ch * 2, dropout)       # 128
        self.enc3 = ConvBlock(base_ch * 2, base_ch * 4, dropout)   # 64
        self.enc4 = ConvBlock(base_ch * 4, base_ch * 8, dropout)   # 32

        self.pool = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = ConvBlock(base_ch * 8, base_ch * 8, dropout)  # 16

        # Decoder (upsample + skip concatenation)
        self.up4 = nn.ConvTranspose2d(base_ch * 8, base_ch * 8, 2, stride=2)
        self.dec4 = ConvBlock(base_ch * 16, base_ch * 4, dropout)
        self.up3 = nn.ConvTranspose2d(base_ch * 4, base_ch * 4, 2, stride=2)
        self.dec3 = ConvBlock(base_ch * 8, base_ch * 2, dropout)
        self.up2 = nn.ConvTranspose2d(base_ch * 2, base_ch * 2, 2, stride=2)
        self.dec2 = ConvBlock(base_ch * 4, base_ch, dropout)
        self.up1 = nn.ConvTranspose2d(base_ch, base_ch, 2, stride=2)
        self.dec1 = ConvBlock(base_ch * 2, base_ch, dropout)

        # Output head (predicts residual)
        self.head = nn.Conv2d(base_ch, 1, 1)

    def forward(self, x):
        """Forward pass.

        Args:
            x: Input tensor of shape [B, in_channels, H, W].
               Channel 0 is always the FLUX output (used for residual connection).
               Channel 1 (optional) is the grayscale condition.

        Returns:
            Refined image of shape [B, 1, H, W], clamped to [0, 1].
        """
        # Save FLUX output for residual connection (always channel 0)
        flux_input = x[:, :1, :, :]

        # Encoder
        e1 = self.enc1(x)                  # base_ch, 256
        e2 = self.enc2(self.pool(e1))      # base_ch*2, 128
        e3 = self.enc3(self.pool(e2))      # base_ch*4, 64
        e4 = self.enc4(self.pool(e3))      # base_ch*8, 32

        # Bottleneck
        b = self.bottleneck(self.pool(e4))  # base_ch*8, 16

        # Decoder with skip connections
        d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1))   # base_ch*4, 32
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))  # base_ch*2, 64
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))  # base_ch, 128
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))  # base_ch, 256

        # Residual learning: output = flux_input + residual
        residual = self.head(d1)
        return torch.clamp(flux_input + residual, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class RefinementDataset(Dataset):
    """Paired dataset: FLUX output (+ optional grayscale condition) -> GT IR.

    # [ml-opt] two-stage-pixel-refinement
    Supports subdirectory layout matching the 7-category vessel dataset.
    Supports optional grayscale condition input for dual-channel mode.
    Supports per-category oversampling for hard categories.
    Supports random horizontal flip and random crop augmentation.
    """

    def __init__(
        self,
        flux_dir,
        gt_dir,
        cond_dir=None,
        oversample_map=None,
        augment=False,
        crop_size=None,
        extensions=(".jpg", ".png"),
    ):
        self.flux_dir = Path(flux_dir)
        self.gt_dir = Path(gt_dir)
        self.cond_dir = Path(cond_dir) if cond_dir else None
        self.augment = augment
        self.crop_size = crop_size
        self.extensions = extensions

        if oversample_map is None:
            oversample_map = {}

        # Collect matched pairs grouped by category
        cat_pairs = {}
        flux_images = sorted(
            p for ext in extensions for p in self.flux_dir.rglob(f"*{ext}")
        )

        for flux_path in flux_images:
            rel = flux_path.relative_to(self.flux_dir)
            gt_path = self.gt_dir / rel
            # Try alternative extension if not found
            if not gt_path.exists():
                for alt_ext in extensions:
                    alt_path = gt_path.with_suffix(alt_ext)
                    if alt_path.exists():
                        gt_path = alt_path
                        break
            if not gt_path.exists():
                continue

            cond_path = None
            if self.cond_dir:
                cond_path = self.cond_dir / rel
                if not cond_path.exists():
                    for alt_ext in extensions:
                        alt_path = cond_path.with_suffix(alt_ext)
                        if alt_path.exists():
                            cond_path = alt_path
                            break
                if not cond_path.exists():
                    cond_path = None  # Skip condition if not found

            cat = rel.parts[0] if len(rel.parts) > 1 else "flat"
            if cat not in cat_pairs:
                cat_pairs[cat] = []
            cat_pairs[cat].append((flux_path, gt_path, cond_path))

        # Build final pair list with oversampling
        self.pairs = []
        self.categories = []  # Track category for each sample (for focal weighting)
        for cat, pairs in sorted(cat_pairs.items()):
            repeat = int(oversample_map.get(cat, 1))
            for _ in range(repeat):
                self.pairs.extend(pairs)
                self.categories.extend([cat] * len(pairs))
            if repeat > 1:
                print(f"  Cat {cat}: {len(pairs)} pairs x {repeat}")
            else:
                print(f"  Cat {cat}: {len(pairs)} pairs")

        print(f"RefinementDataset: {len(self.pairs)} total pairs (augment={augment})")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        flux_path, gt_path, cond_path = self.pairs[idx]

        flux_img = np.array(Image.open(flux_path).convert("L"), dtype=np.float32) / 255.0
        gt_img = np.array(Image.open(gt_path).convert("L"), dtype=np.float32) / 255.0

        cond_img = None
        if cond_path is not None:
            cond_img = np.array(Image.open(cond_path).convert("L"), dtype=np.float32) / 255.0

        # Augmentation
        if self.augment:
            # Random horizontal flip
            if np.random.rand() > 0.5:
                flux_img = np.fliplr(flux_img).copy()
                gt_img = np.fliplr(gt_img).copy()
                if cond_img is not None:
                    cond_img = np.fliplr(cond_img).copy()

            # [ml-opt] two-stage-pixel-refinement: random crop augmentation
            if self.crop_size and self.crop_size < flux_img.shape[0]:
                h, w = flux_img.shape
                cs = self.crop_size
                top = np.random.randint(0, h - cs + 1)
                left = np.random.randint(0, w - cs + 1)
                flux_img = flux_img[top:top + cs, left:left + cs]
                gt_img = gt_img[top:top + cs, left:left + cs]
                if cond_img is not None:
                    cond_img = cond_img[top:top + cs, left:left + cs]

        # Build input tensor
        flux_tensor = torch.from_numpy(flux_img).unsqueeze(0)  # [1, H, W]
        gt_tensor = torch.from_numpy(gt_img).unsqueeze(0)      # [1, H, W]

        if cond_img is not None:
            cond_tensor = torch.from_numpy(cond_img).unsqueeze(0)  # [1, H, W]
            input_tensor = torch.cat([flux_tensor, cond_tensor], dim=0)  # [2, H, W]
        else:
            input_tensor = flux_tensor  # [1, H, W]

        return input_tensor, gt_tensor, idx


# ---------------------------------------------------------------------------
# Loss computation
# ---------------------------------------------------------------------------
class CompositeLoss(nn.Module):
    """Composite loss: L1 + SSIM + optional LPIPS.

    # [ml-opt] two-stage-pixel-refinement
    # [ml-opt] focal-loss-hard-categories: per-sample focal weighting

    Args:
        l1_weight: Weight for L1 loss.
        ssim_weight: Weight for SSIM loss (1 - SSIM).
        lpips_weight: Weight for LPIPS perceptual loss (0 = disabled).
        focal_gamma: Focal loss gamma. 0 = no focal weighting.
                     Higher values upweight harder samples more.
    """

    def __init__(self, l1_weight=1.0, ssim_weight=0.1, lpips_weight=0.0, focal_gamma=0.0):
        super().__init__()
        self.l1_weight = l1_weight
        self.ssim_weight = ssim_weight
        self.lpips_weight = lpips_weight
        self.focal_gamma = focal_gamma

        self.ssim_loss = SSIMLoss()
        self.lpips_fn = None

        if lpips_weight > 0:
            try:
                import lpips
                # LPIPS expects 3-channel input; we replicate grayscale
                self.lpips_fn = lpips.LPIPS(net="vgg", verbose=False)
                # Freeze LPIPS parameters
                for p in self.lpips_fn.parameters():
                    p.requires_grad = False
                print("[CompositeLoss] LPIPS perceptual loss enabled (VGG)")
            except ImportError:
                print("[CompositeLoss] WARNING: lpips not installed, disabling perceptual loss")
                self.lpips_weight = 0.0

    def forward(self, pred, target):
        """Compute composite loss with optional focal weighting.

        Args:
            pred: Predicted images [B, 1, H, W].
            target: Ground truth images [B, 1, H, W].

        Returns:
            Tuple of (total_loss, loss_dict) where loss_dict has per-component values.
        """
        batch_size = pred.shape[0]
        loss_dict = {}

        # --- Per-sample L1 loss ---
        per_sample_l1 = F.l1_loss(pred, target, reduction="none").reshape(batch_size, -1).mean(dim=1)
        loss_dict["l1"] = per_sample_l1.mean().item()

        # --- SSIM loss (batch-level, since windowed) ---
        ssim_val = self.ssim_loss(pred, target)
        loss_dict["ssim"] = ssim_val.item()

        # --- LPIPS loss ---
        lpips_val = torch.tensor(0.0, device=pred.device)
        if self.lpips_weight > 0 and self.lpips_fn is not None:
            # LPIPS expects 3-channel [-1, 1] input
            pred_3ch = pred.expand(-1, 3, -1, -1) * 2.0 - 1.0
            target_3ch = target.expand(-1, 3, -1, -1) * 2.0 - 1.0
            lpips_val = self.lpips_fn(pred_3ch, target_3ch).mean()
            loss_dict["lpips"] = lpips_val.item()

        # --- Focal weighting (Proposal 5) ---
        # [ml-opt] focal-loss-hard-categories
        if self.focal_gamma > 0:
            # Normalize per-sample loss relative to batch mean
            with torch.no_grad():
                difficulty = per_sample_l1 / (per_sample_l1.mean() + 1e-8)
                focal_weight = difficulty.pow(self.focal_gamma)
                # Normalize so mean weight = 1 (preserves loss scale)
                focal_weight = focal_weight / (focal_weight.mean() + 1e-8)
            weighted_l1 = (per_sample_l1 * focal_weight).mean()
            loss_dict["focal_max_weight"] = focal_weight.max().item()
        else:
            weighted_l1 = per_sample_l1.mean()

        # --- Composite loss ---
        total = self.l1_weight * weighted_l1 + self.ssim_weight * ssim_val + self.lpips_weight * lpips_val
        loss_dict["total"] = total.item()

        return total, loss_dict


# ---------------------------------------------------------------------------
# Cosine annealing with linear warmup
# ---------------------------------------------------------------------------
class CosineWarmupScheduler(torch.optim.lr_scheduler._LRScheduler):
    """Cosine annealing with linear warmup."""

    def __init__(self, optimizer, warmup_steps, total_steps, min_lr=0.0, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            # Linear warmup
            factor = self.last_epoch / max(1, self.warmup_steps)
        else:
            # Cosine annealing
            progress = (self.last_epoch - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
            factor = 0.5 * (1.0 + math.cos(math.pi * progress))
        return [self.min_lr + (base_lr - self.min_lr) * factor for base_lr in self.base_lrs]


# ---------------------------------------------------------------------------
# Validation utilities
# ---------------------------------------------------------------------------
def compute_psnr(pred, target):
    """Compute PSNR in dB. pred and target are [B, 1, H, W] in [0, 1]."""
    mse = F.mse_loss(pred, target, reduction="none").reshape(pred.shape[0], -1).mean(dim=1)
    psnr = 10.0 * torch.log10(1.0 / (mse + 1e-10))
    return psnr  # [B]


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")

    # --- Dataset ---
    oversample_map = {}
    if args.oversample_map:
        oversample_map = json.loads(args.oversample_map)
        oversample_map = {str(k): int(v) for k, v in oversample_map.items()}

    in_channels = 2 if args.cond_dir else 1

    train_ds = RefinementDataset(
        flux_dir=args.flux_dir,
        gt_dir=args.gt_dir,
        cond_dir=args.cond_dir,
        oversample_map=oversample_map,
        augment=True,
        crop_size=args.crop_size,
    )

    val_ds = None
    if args.val_flux_dir and args.val_gt_dir:
        val_ds = RefinementDataset(
            flux_dir=args.val_flux_dir,
            gt_dir=args.val_gt_dir,
            cond_dir=args.val_cond_dir,
            augment=False,
        )

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = None
    if val_ds and len(val_ds) > 0:
        val_loader = DataLoader(
            val_ds, batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers, pin_memory=True,
        )

    # --- Model ---
    model = RefinementUNet(
        in_channels=in_channels,
        base_ch=args.base_channels,
        dropout=args.dropout,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"RefinementUNet: {n_params:,} parameters (in_ch={in_channels}, base_ch={args.base_channels})")

    # --- Loss ---
    criterion = CompositeLoss(
        l1_weight=args.l1_weight,
        ssim_weight=args.ssim_weight,
        lpips_weight=args.lpips_weight,
        focal_gamma=args.focal_gamma,
    ).to(device)

    # --- Optimizer ---
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    # --- Scheduler ---
    total_steps = args.epochs * len(train_loader)
    warmup_steps = args.warmup_epochs * len(train_loader)
    scheduler = CosineWarmupScheduler(
        optimizer, warmup_steps=warmup_steps, total_steps=total_steps,
        min_lr=args.lr * 0.01,
    )

    # --- Output ---
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    config = vars(args)
    config["n_params"] = n_params
    config["in_channels"] = in_channels
    config["total_train_samples"] = len(train_ds)
    config["total_val_samples"] = len(val_ds) if val_ds else 0
    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    best_val_psnr = 0.0
    best_epoch = 0
    history = []
    global_step = 0

    print(f"\nStarting training for {args.epochs} epochs...")
    print(f"  Train samples: {len(train_ds)}, Val samples: {len(val_ds) if val_ds else 0}")
    print(f"  Batch size: {args.batch_size}, Total steps: {total_steps}")
    print(f"  LR: {args.lr}, Warmup epochs: {args.warmup_epochs}")
    print(f"  Loss weights: L1={args.l1_weight}, SSIM={args.ssim_weight}, "
          f"LPIPS={args.lpips_weight}, Focal_gamma={args.focal_gamma}")
    print()

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        model.train()
        total_loss = 0.0
        total_l1 = 0.0
        total_ssim_loss = 0.0
        total_lpips_loss = 0.0
        n_batches = 0

        for input_batch, gt_batch, _ in train_loader:
            input_batch = input_batch.to(device)
            gt_batch = gt_batch.to(device)

            pred = model(input_batch)

            loss, loss_dict = criterion(pred, gt_batch)

            optimizer.zero_grad()
            loss.backward()
            # [ml-opt] two-stage-pixel-refinement: gradient clipping for stability
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            scheduler.step()
            global_step += 1

            total_loss += loss_dict["total"]
            total_l1 += loss_dict["l1"]
            total_ssim_loss += loss_dict["ssim"]
            total_lpips_loss += loss_dict.get("lpips", 0.0)
            n_batches += 1

        avg_loss = total_loss / n_batches
        avg_l1 = total_l1 / n_batches
        avg_ssim_loss = total_ssim_loss / n_batches
        avg_lpips = total_lpips_loss / n_batches
        cur_lr = optimizer.param_groups[0]["lr"]
        epoch_time = time.time() - epoch_start

        log_msg = (
            f"Epoch {epoch:3d}/{args.epochs} "
            f"[{epoch_time:.1f}s] "
            f"loss={avg_loss:.6f} "
            f"(L1={avg_l1:.6f}, SSIM={avg_ssim_loss:.6f}"
        )
        if args.lpips_weight > 0:
            log_msg += f", LPIPS={avg_lpips:.6f}"
        log_msg += f") lr={cur_lr:.2e}"

        epoch_record = {
            "epoch": epoch,
            "train_loss": avg_loss,
            "train_l1": avg_l1,
            "train_ssim_loss": avg_ssim_loss,
            "lr": cur_lr,
        }

        # --- Validation ---
        if val_loader and (epoch % args.val_every == 0 or epoch == args.epochs):
            model.eval()
            val_psnr_sum = 0.0
            val_ssim_sum = 0.0
            val_loss_sum = 0.0
            val_count = 0

            with torch.no_grad():
                for input_batch, gt_batch, _ in val_loader:
                    input_batch = input_batch.to(device)
                    gt_batch = gt_batch.to(device)
                    pred = model(input_batch)

                    val_loss, _ = criterion(pred, gt_batch)
                    val_loss_sum += val_loss.item() * pred.shape[0]

                    # Per-image PSNR
                    psnr_vals = compute_psnr(pred, gt_batch)
                    val_psnr_sum += psnr_vals.sum().item()

                    # Per-image SSIM (via 1 - SSIMLoss per image)
                    ssim_fn = criterion.ssim_loss
                    for i in range(pred.shape[0]):
                        ssim_val = 1.0 - ssim_fn(pred[i:i + 1], gt_batch[i:i + 1])
                        val_ssim_sum += ssim_val.item()
                    val_count += pred.shape[0]

            val_psnr = val_psnr_sum / max(val_count, 1)
            val_ssim = val_ssim_sum / max(val_count, 1)
            val_loss_avg = val_loss_sum / max(val_count, 1)

            log_msg += f" | val_PSNR={val_psnr:.4f} val_SSIM={val_ssim:.4f} val_loss={val_loss_avg:.6f}"

            if val_psnr > best_val_psnr:
                best_val_psnr = val_psnr
                best_epoch = epoch
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_psnr": val_psnr,
                    "val_ssim": val_ssim,
                    "config": config,
                }, output_dir / "best_model.pt")
                log_msg += " [BEST]"

            epoch_record.update({
                "val_psnr": val_psnr,
                "val_ssim": val_ssim,
                "val_loss": val_loss_avg,
            })

        history.append(epoch_record)
        print(log_msg)

        # Save checkpoint every N epochs
        if epoch % args.save_every == 0:
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "config": config,
            }, output_dir / f"model_epoch{epoch}.pt")

    # Save final model
    torch.save({
        "epoch": args.epochs,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": config,
    }, output_dir / "final_model.pt")

    # Save training history
    with open(output_dir / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"\nTraining complete.")
    print(f"  Best val PSNR: {best_val_psnr:.4f} (epoch {best_epoch})")
    print(f"  Models saved to {output_dir}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train refinement U-Net (two-stage pixel refinement)"
    )

    # Data paths
    parser.add_argument("--flux_dir", required=True,
                        help="Directory of FLUX LoRA outputs (training)")
    parser.add_argument("--gt_dir", required=True,
                        help="Directory of ground truth IR images (training)")
    parser.add_argument("--cond_dir", default=None,
                        help="Directory of grayscale condition images (optional, enables 2ch input)")
    parser.add_argument("--val_flux_dir", default=None,
                        help="Directory of FLUX outputs (validation)")
    parser.add_argument("--val_gt_dir", default=None,
                        help="Directory of GT IR images (validation)")
    parser.add_argument("--val_cond_dir", default=None,
                        help="Directory of grayscale condition images (validation)")
    parser.add_argument("--output_dir", required=True,
                        help="Output directory for model checkpoints")

    # Training
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--warmup_epochs", type=int, default=5,
                        help="Number of warmup epochs for LR scheduler")
    parser.add_argument("--grad_clip", type=float, default=1.0,
                        help="Gradient clipping norm (0 = disabled)")
    parser.add_argument("--num_workers", type=int, default=4)

    # Architecture
    parser.add_argument("--base_channels", type=int, default=32,
                        help="Base channel width for U-Net")
    parser.add_argument("--dropout", type=float, default=0.0,
                        help="Dropout rate for regularization")

    # Loss weights
    parser.add_argument("--l1_weight", type=float, default=1.0,
                        help="Weight for L1 loss")
    parser.add_argument("--ssim_weight", type=float, default=0.1,
                        help="Weight for SSIM loss (1 - SSIM)")
    parser.add_argument("--lpips_weight", type=float, default=0.0,
                        help="Weight for LPIPS perceptual loss (0 = disabled)")
    # [ml-opt] focal-loss-hard-categories
    parser.add_argument("--focal_gamma", type=float, default=0.0,
                        help="Focal loss gamma for hard sample mining (0 = disabled, try 0.5-2.0)")

    # Data augmentation
    parser.add_argument("--crop_size", type=int, default=None,
                        help="Random crop size for augmentation (None = disabled)")
    parser.add_argument("--oversample_map", default=None,
                        help='JSON dict mapping category to repeat count, '
                             'e.g. \'{"1":3,"4":2,"5":3}\'')

    # Checkpointing
    parser.add_argument("--val_every", type=int, default=5,
                        help="Validate every N epochs")
    parser.add_argument("--save_every", type=int, default=25,
                        help="Save checkpoint every N epochs")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    # Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    train(args)
