"""Apply refinement U-Net to FLUX outputs (inference).

# [ml-opt] two-stage-pixel-refinement
# Two-stage pipeline: FLUX output -> RefinementUNet -> refined output
# Supports optional grayscale condition input (dual-channel mode).

Usage:
    # Single-channel mode (FLUX output only):
    python refine.py \\
        --model_path /path/to/best_model.pt \\
        --input_dir /path/to/flux_outputs \\
        --output_dir /path/to/refined_outputs

    # Dual-channel mode (FLUX output + grayscale condition):
    python refine.py \\
        --model_path /path/to/best_model.pt \\
        --input_dir /path/to/flux_outputs \\
        --cond_dir /path/to/grayscale_inputs \\
        --output_dir /path/to/refined_outputs
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from train_refinement import RefinementUNet


def load_model(model_path, device):
    """Load refinement model from checkpoint.

    Supports both old format (raw state_dict) and new format (dict with config).
    """
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        # New checkpoint format
        config = checkpoint.get("config", {})
        in_channels = config.get("in_channels", 1)
        base_ch = config.get("base_channels", 32)
        dropout = config.get("dropout", 0.0)
        state_dict = checkpoint["model_state_dict"]
        epoch = checkpoint.get("epoch", "?")
        val_psnr = checkpoint.get("val_psnr", None)
        print(f"Loaded checkpoint from epoch {epoch}" +
              (f" (val_PSNR={val_psnr:.4f})" if val_psnr else ""))
    else:
        # Legacy format: raw state_dict
        state_dict = checkpoint
        # Infer in_channels from first conv weight shape
        first_key = [k for k in state_dict.keys() if "enc1" in k and "weight" in k][0]
        in_channels = state_dict[first_key].shape[1]
        base_ch = state_dict[first_key].shape[0]
        dropout = 0.0
        print(f"Loaded legacy checkpoint (inferred in_channels={in_channels}, base_ch={base_ch})")

    model = RefinementUNet(in_channels=in_channels, base_ch=base_ch, dropout=dropout).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    return model, in_channels


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model
    model, in_channels = load_model(args.model_path, device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"RefinementUNet: {n_params:,} parameters (in_channels={in_channels})")

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cond_dir = Path(args.cond_dir) if args.cond_dir else None
    if in_channels == 2 and cond_dir is None:
        print("WARNING: Model expects 2-channel input but --cond_dir not provided. "
              "Will replicate FLUX input as condition.")
    if in_channels == 1 and cond_dir is not None:
        print("WARNING: Model expects 1-channel input but --cond_dir provided. "
              "Condition input will be ignored.")
        cond_dir = None

    # Find input images
    extensions = (".jpg", ".png")
    input_paths = sorted(
        p for ext in extensions for p in input_dir.rglob(f"*{ext}")
    )
    print(f"Processing {len(input_paths)} images...")

    with torch.no_grad():
        for img_path in tqdm(input_paths, desc="Refining"):
            # Load FLUX output
            img = np.array(Image.open(img_path).convert("L"), dtype=np.float32) / 255.0
            flux_tensor = torch.from_numpy(img).unsqueeze(0).unsqueeze(0).to(device)  # [1, 1, H, W]

            # Build input
            if in_channels == 2:
                if cond_dir is not None:
                    rel_path = img_path.relative_to(input_dir)
                    cond_path = cond_dir / rel_path
                    # Try alternative extension
                    if not cond_path.exists():
                        for ext in extensions:
                            alt = cond_path.with_suffix(ext)
                            if alt.exists():
                                cond_path = alt
                                break
                    if cond_path.exists():
                        cond_img = np.array(Image.open(cond_path).convert("L"), dtype=np.float32) / 255.0
                        cond_tensor = torch.from_numpy(cond_img).unsqueeze(0).unsqueeze(0).to(device)
                    else:
                        # Fallback: replicate FLUX input
                        cond_tensor = flux_tensor
                else:
                    cond_tensor = flux_tensor

                input_tensor = torch.cat([flux_tensor, cond_tensor], dim=1)  # [1, 2, H, W]
            else:
                input_tensor = flux_tensor  # [1, 1, H, W]

            # Forward pass
            refined = model(input_tensor)
            refined_np = (refined.squeeze().cpu().numpy() * 255).clip(0, 255).astype(np.uint8)

            # Save with same relative path structure
            rel_path = img_path.relative_to(input_dir)
            out_path = output_dir / rel_path
            out_path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(refined_np, mode="L").save(out_path)

    print(f"Refined images saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Apply refinement U-Net to FLUX outputs"
    )
    parser.add_argument("--model_path", required=True,
                        help="Path to refinement model checkpoint (.pt)")
    parser.add_argument("--input_dir", required=True,
                        help="Directory of FLUX outputs to refine")
    parser.add_argument("--cond_dir", default=None,
                        help="Directory of grayscale condition images (for 2-channel models)")
    parser.add_argument("--output_dir", required=True,
                        help="Output directory for refined images")
    # Kept for backward compat but now auto-detected from checkpoint
    parser.add_argument("--base_channels", type=int, default=None,
                        help="(Deprecated) Auto-detected from checkpoint")
    args = parser.parse_args()
    main(args)
