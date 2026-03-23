#!/usr/bin/env python3
"""Apply contrast stretching to IR images using p2/p98 percentile rescaling.

Reference: https://scikit-image.org/docs/stable/auto_examples/color_exposure/plot_equalize.html

Usage:
    python contrast_stretch.py --input_dir /path/to/ir --output_dir /path/to/ir_enhanced
"""
import argparse
import os

import numpy as np
from PIL import Image
from skimage import exposure
from tqdm import tqdm


def contrast_stretch(img_array):
    """Rescale intensity from [p2, p98] to [0, 255]."""
    p2, p98 = np.percentile(img_array, (2, 98))
    return exposure.rescale_intensity(img_array, in_range=(p2, p98))


def main(args):
    input_dir = args.input_dir
    output_dir = args.output_dir

    # Collect all jpg images preserving subdirectory structure
    image_paths = []
    for root, _, files in os.walk(input_dir):
        for f in sorted(files):
            if f.lower().endswith((".jpg", ".jpeg", ".png")):
                image_paths.append(os.path.join(root, f))

    print(f"Found {len(image_paths)} images in {input_dir}")

    for img_path in tqdm(image_paths, desc="Contrast stretching"):
        img = np.array(Image.open(img_path).convert("L"))
        stretched = contrast_stretch(img)
        stretched_pil = Image.fromarray(stretched)

        # Preserve subdirectory structure
        rel_path = os.path.relpath(img_path, input_dir)
        out_path = os.path.join(output_dir, rel_path)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        stretched_pil.save(out_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Contrast stretching for IR images")
    parser.add_argument("--input_dir", required=True, help="Input IR images directory")
    parser.add_argument("--output_dir", required=True, help="Output directory for enhanced images")
    args = parser.parse_args()
    main(args)
