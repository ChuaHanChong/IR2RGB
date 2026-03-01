import argparse
import json
import os
import sys
from pathlib import Path
from typing import Literal

import outlines
import pandas as pd
import PIL
import torch
import transformers
from outlines.inputs import Chat, Image
from pydantic import BaseModel
from tqdm import tqdm
from transformers import AutoConfig, AutoProcessor


class OutputResponse(BaseModel):
    answer: Literal["Yes", "No"]


def main(args):
    with open(args.category_mapping) as f:
        mapping = json.load(f)

    all_paths = sorted(Path(args.input_folder).rglob("*.jpg"))
    if not all_paths:
        print(f"No .jpg images found under {args.input_folder}", file=sys.stderr)
        sys.exit(1)

    done = set()
    if Path(args.output_file).exists():
        done = set(pd.read_csv(args.output_file)["image_path"].tolist())
    remaining = [p for p in all_paths if str(p) not in done]
    print(f"Total: {len(all_paths)} | Already done: {len(done)} | Remaining: {len(remaining)}")
    if not remaining:
        print("Nothing to do.")
        return

    hf_model = getattr(transformers, AutoConfig.from_pretrained(args.model_id).architectures[0]).from_pretrained(
        args.model_id,
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="auto",
    )
    hf_processor = AutoProcessor.from_pretrained(args.model_id)
    model = outlines.from_transformers(hf_model, hf_processor)

    batches = [remaining[i : i + args.batch_size] for i in range(0, len(remaining), args.batch_size)]
    total_written = 0
    for batch in tqdm(batches, desc="Analyzing images"):
        rows = []
        try:
            prompts = []
            for path in batch:
                cat_name = mapping[path.parent.name]
                prompt_text = args.prompt.format(category=cat_name)
                prompts.append(Chat([{
                    "role": "user",
                    "content": [
                        {"type": "image", "image": Image(PIL.Image.open(str(path)))},
                        {"type": "text", "text": prompt_text},
                    ],
                }]))

            outputs = model.batch(prompts, output_type=OutputResponse, max_new_tokens=20)

            for path, output in zip(batch, outputs):
                rows.append({"image_path": str(path), "is_clear": OutputResponse.model_validate_json(output).answer == "Yes"})

        except Exception as e:
            print(f"Error processing paths:")
            for p in batch:
                print(f"  {p}")
            print(f"Error message: {e}")
            continue

        df = pd.DataFrame(rows, columns=["image_path", "is_clear"])
        header = not Path(args.output_file).exists()
        os.makedirs(os.path.dirname(os.path.abspath(args.output_file)), exist_ok=True)
        df.to_csv(args.output_file, mode="a", index=False, header=header)
        total_written += len(rows)

    print(f"Saved {total_written} new rows to {args.output_file}")

    df = pd.read_csv(args.output_file)
    print(f"\nTotal: {len(df)} | True: {df.is_clear.sum()} | False: {(~df.is_clear).sum()}")
    print(f"Pass rate: {df.is_clear.mean():.1%}\n")
    df["cat"] = df["image_path"].apply(lambda p: Path(p).parent.name)
    print(df.groupby("cat")["is_clear"].agg(["sum", "count", "mean"]).rename(columns={"sum": "true", "count": "total", "mean": "rate"}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Filter IR images using a VLM with structured Yes/No output.")
    parser.add_argument("--model-id", default="Qwen/Qwen3-VL-8B-Instruct", help="HuggingFace model ID.")
    parser.add_argument("--input_folder", required=True, help="Root folder of the dataset.")
    parser.add_argument("--category-mapping", required=True, help="JSON file mapping category IDs to names.")
    parser.add_argument("--output_file", required=True, help="Output CSV file path.")
    parser.add_argument("--batch-size", type=int, default=8, help="Number of images per inference batch.")
    parser.add_argument(
        "--prompt",
        default="Is the {category} visible and recognizable? Answer with 'Yes' or 'No'.",
        help="Prompt template. Use {category} as placeholder for the category name.",
    )
    args = parser.parse_args()

    main(args)
