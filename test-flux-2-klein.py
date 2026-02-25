import argparse
import os
from pathlib import Path

import torch
from diffusers import Flux2KleinPipeline
from PIL import Image
from tqdm import tqdm

device = "cuda"
dtype = torch.bfloat16

pipe = Flux2KleinPipeline.from_pretrained("black-forest-labs/FLUX.2-klein-4B", torch_dtype=dtype)
pipe.enable_model_cpu_offload()  # save some VRAM by offloading the model to CPU


def main(args):
    input_folder = Path(args.input_folder)
    input_paths = list(input_folder.glob("*.jpg"))

    output_folder = Path(args.output_folder)
    os.makedirs(output_folder, exist_ok=True)

    for input_path in tqdm(input_paths, total=len(input_paths)):
        try:
            image_path = input_path
            output_path = output_folder / input_path.name

            with torch.inference_mode():
                ouput_image = pipe(
                    prompt=args.prompt,
                    image=[Image.open(image_path).convert("RGB")],
                    height=1024,
                    width=1024,
                    guidance_scale=args.guidance_scale,
                    num_inference_steps=args.num_inference_steps,
                    generator=torch.Generator(device=device).manual_seed(0),
                ).images[0]
                ouput_image.save(output_path)

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
    default=4,
    help="Number of inference steps to use for image generation.",
)
argparser.add_argument(
    "--guidance_scale",
    type=float,
    default=1.0,
    help="Guidance scale to use for image generation.",
)

args = argparser.parse_args()
main(args)
