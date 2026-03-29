"""Create a HuggingFace DatasetDict from paired image directory structure.

Supports both flat and subdirectory layouts:
  Flat:   cond_dir/*.jpg + target_dir/*.jpg
  Subdir: cond_dir/0/*.jpg + target_dir/0/*.jpg

Usage:
    python create_hf_dataset.py \
        --cond_dir /path/to/grayscale \
        --target_dir /path/to/ir \
        --output_dir /path/to/HFDataset \
        --prompt "turn the visible image of Marine Vessel into sks infrared"
"""

import argparse
from datasets import Dataset, DatasetDict, Image
from pathlib import Path


def collect_pairs(cond_dir, target_dir):
    """Collect matched image pairs from cond and target directories."""
    cond_path = Path(cond_dir)
    target_path = Path(target_dir)

    pairs = []

    # Check if subdirectory layout
    subdirs = sorted([d for d in cond_path.iterdir() if d.is_dir()])

    if subdirs:
        for subdir in subdirs:
            for img_file in sorted(subdir.glob("*.jpg")):
                target_file = target_path / subdir.name / img_file.name
                if target_file.exists():
                    pairs.append((str(img_file), str(target_file)))
    else:
        # Flat layout
        for img_file in sorted(cond_path.glob("*.jpg")):
            target_file = target_path / img_file.name
            if target_file.exists():
                pairs.append((str(img_file), str(target_file)))

    return pairs


def main(args):
    pairs = collect_pairs(args.cond_dir, args.target_dir)

    if not pairs:
        print(f"No matching pairs found between {args.cond_dir} and {args.target_dir}")
        return

    records = {
        "cond_image": [p[0] for p in pairs],
        "target_image": [p[1] for p in pairs],
        "caption": [args.prompt] * len(pairs),
    }

    ds = Dataset.from_dict(records)
    ds = ds.cast_column("cond_image", Image())
    ds = ds.cast_column("target_image", Image())

    dd = DatasetDict({"train": ds})
    dd.save_to_disk(args.output_dir)
    print(f"Saved {len(ds)} examples to {args.output_dir}")
    print(f"Columns: {ds.column_names}")
    print(f"Sample: cond={pairs[0][0]}, target={pairs[0][1]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create HF Dataset from paired image directories")
    parser.add_argument("--cond_dir", required=True, help="Directory of conditioning images (e.g., grayscale/rgb)")
    parser.add_argument("--target_dir", required=True, help="Directory of target images (e.g., ir)")
    parser.add_argument("--output_dir", required=True, help="Output path for HF Dataset")
    parser.add_argument("--prompt", default="turn the visible image of Marine Vessel into sks infrared",
                        help="Caption/prompt for all images")
    args = parser.parse_args()
    main(args)
