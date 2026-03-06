import argparse
import os
import sys
from pathlib import Path

import cv2
import pandas as pd
from tqdm import tqdm


def laplacian_variance(image_path: Path) -> float:
    """Return the Laplacian variance of the grayscale image."""
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    return cv2.Laplacian(img, cv2.CV_64F).var()


def main(args):
    paths = sorted(Path(args.input_folder).rglob("*.jpg"))
    if not paths:
        print(f"No .jpg images found under {args.input_folder}", file=sys.stderr)
        sys.exit(1)

    records = []
    for path in tqdm(paths, desc="Computing Laplacian variance"):
        records.append({"image_path": str(path), "laplacian_variance": laplacian_variance(path)})

    df = pd.DataFrame(records, columns=["image_path", "laplacian_variance"])
    os.makedirs(os.path.dirname(os.path.abspath(args.output_file)), exist_ok=True)
    df.to_csv(args.output_file, index=False)
    print(f"Saved {len(df)} rows to {args.output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute Laplacian variance for IR images to detect blur.")
    parser.add_argument("--input_folder", required=True, help="Root folder of the dataset.")
    parser.add_argument("--output_file", required=True, help="Output CSV file path.")
    args = parser.parse_args()

    main(args)
