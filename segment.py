from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import Sam3Model, Sam3Processor
from tqdm import tqdm


def refine_mask(mask: np.ndarray, convex_hull: bool = False) -> np.ndarray:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_contour = max(contours, key=cv2.contourArea)

    if convex_hull:
        largest_contour = cv2.convexHull(largest_contour)

    refined_mask = np.zeros_like(mask)
    cv2.drawContours(
        refined_mask,
        [largest_contour],
        contourIdx=-1,
        color=1,
        thickness=-1,
    )

    return refined_mask


def main(args):
    input_folder = Path(args.input_folder)
    input_paths = sorted(input_folder.rglob("*.jpg"))
    if not input_paths:
        print(f"No .jpg images found under {args.input_folder}")
        return

    device = "cuda"
    model = Sam3Model.from_pretrained("facebook/sam3").to(device)
    processor = Sam3Processor.from_pretrained("facebook/sam3")

    output_folder = Path(args.output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    for input_path in tqdm(input_paths, desc="Segmenting images"):
        relative_path = input_path.relative_to(input_folder)
        output_path = output_folder / relative_path.with_suffix(".png")
        if output_path.exists():
            continue

        try:
            image = Image.open(input_path).convert("RGB")
            inputs = processor(images=image, text=args.text, return_tensors="pt").to(device)

            with torch.no_grad():
                outputs = model(**inputs)

            results = processor.post_process_instance_segmentation(
                outputs,
                threshold=args.threshold,
                mask_threshold=args.mask_threshold,
                target_sizes=inputs.get("original_sizes").tolist()
            )[0]

            masks = results["masks"].cpu().numpy()
            scores = results["scores"].cpu().numpy()

            if len(masks) == 0:
                print(f"No masks found, skipping: {input_path}")
                continue

            if args.merge_masks:
                merged = np.any(masks, axis=0).astype(np.uint8)
                final_mask = refine_mask(merged)
            else:
                sorted_ind = np.argsort(scores)[::-1]
                top_mask = masks[sorted_ind[0]].astype(np.uint8)
                final_mask = refine_mask(top_mask)

            output_path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(final_mask * 255).save(output_path)
        
        except Exception as e:
            print(f"Error processing {input_path.name}: {e}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Segment objects in images using SAM3.")
    parser.add_argument("--input_folder", required=True, help="Folder containing input images.")
    parser.add_argument("--output_folder", required=True, help="Folder to save output masks.")
    parser.add_argument("--text", default="object", help="Text prompt for segmentation.")
    parser.add_argument("--threshold", type=float, default=0.3, help="Confidence threshold for detections.")
    parser.add_argument("--mask_threshold", type=float, default=0.5, help="Mask binarization threshold.")
    parser.add_argument("--merge_masks", action="store_true", help="Merge all detected masks instead of using only the top-scoring one.")
    args = parser.parse_args()

    main(args)
