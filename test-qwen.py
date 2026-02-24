import argparse
import os
from pathlib import Path

import torch
from diffusers import QwenImageEditPlusPipeline
from PIL import Image
from tqdm import tqdm

pipeline = QwenImageEditPlusPipeline.from_pretrained("Qwen/Qwen-Image-Edit-2511", torch_dtype=torch.bfloat16)
pipeline.enable_sequential_cpu_offload()


def main(args):
    input_folder = Path(args.input_folder)
    input_paths = list(input_folder.glob("*.jpg"))

    output_folder = Path(args.output_folder)
    os.makedirs(output_folder, exist_ok=True)

    for input_path in tqdm(input_paths, total=len(input_paths)):
        try:
            image = Image.open(input_path).convert("RGB")
            output_path = output_folder / input_path.name

            inputs = {
                "image": [image],
                "prompt": args.prompt,
                "generator": torch.manual_seed(0),
                "true_cfg_scale": args.true_cfg_scale,
                "negative_prompt": " ",
                "num_inference_steps": args.num_inference_steps,
                "guidance_scale": 1.0,
                "num_images_per_prompt": 1,
            }

            with torch.inference_mode():
                output = pipeline(**inputs)
                output_image = output.images[0]
                output_image.save(str(output_path))

        except Exception as e:
            print(f"Error processing {input_path.name}: {e}")


argparser = argparse.ArgumentParser(description="Qwen-Image Edit Inference Script")
argparser.add_argument(
    "--input_folder",
    type=str,
    required=True,
    help="Path to the folder containing input images.",
)
argparser.add_argument(
    "--output_folder",
    type=str,
    required=True,
    help="Path to the folder to save edited images.",
)
argparser.add_argument(
    "--prompt",
    type=str,
    default="Turn this thermal infrared image of a marine vessel into a visually realistic RGB image as it would appear under visible light.",
    help="Prompt to guide the image editing process.",
)
argparser.add_argument(
    "--num_inference_steps",
    type=int,
    default=40,
    help="Number of inference steps to use for image generation.",
)
argparser.add_argument(
    "--true_cfg_scale",
    type=float,
    default=4.0,
    help="CFG scale to use for image generation.",
)

args = argparser.parse_args()
main(args)
