import argparse
from pathlib import Path

import numpy as np
import torch
from diffusers import Flux2KleinPipeline, Flux2Pipeline
from PIL import Image
from skimage import exposure
from tqdm import tqdm


def main(args):
    input_folder = Path(args.input_folder)
    input_paths = sorted(input_folder.rglob("*.jpg"))
    if not input_paths:
        print(f"No .jpg images found under {args.input_folder}")
        return

    device = "cuda"
    dtype = torch.bfloat16

    if args.model_name in ["black-forest-labs/FLUX.2-klein-4B", "black-forest-labs/FLUX.2-klein-9B"]:
        pipe = Flux2KleinPipeline.from_pretrained(args.model_name, torch_dtype=dtype)
    elif args.model_name == "black-forest-labs/FLUX.2-dev":
        pipe = Flux2Pipeline.from_pretrained(args.model_name, torch_dtype=dtype)
    else:
        print(f"Unsupported model name: {args.model_name}")
        return

    if args.lora_weights:
        pipe.load_lora_weights(args.lora_weights)
        pipe.fuse_lora()
        pipe.unload_lora_weights()
        print(f"Loaded and fused LoRA weights from {args.lora_weights}")

    pipe.enable_model_cpu_offload()

    output_folder = Path(args.output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    for input_path in tqdm(input_paths, desc="Generating images"):
        try:
            input_image = Image.open(input_path).convert("RGB")
            w_in, h_in = input_image.size
            gen_h = args.height if args.height is not None else h_in
            gen_w = args.width if args.width is not None else w_in

            with torch.inference_mode():
                output_image = pipe(
                    prompt=args.prompt,
                    image=[input_image],
                    height=gen_h,
                    width=gen_w,
                    guidance_scale=args.guidance_scale,
                    num_inference_steps=args.num_inference_steps,
                    generator=torch.Generator(device=device).manual_seed(args.seed),
                ).images[0]

            relative_path = input_path.relative_to(input_folder)

            # Save grayscale (default)
            gray_img = output_image.convert("L")
            if args.contrast_stretch:
                arr = np.array(gray_img)
                p2, p98 = np.percentile(arr, (2, 98))
                gray_img = Image.fromarray(exposure.rescale_intensity(arr, in_range=(p2, p98)))
            gray_path = output_folder / relative_path
            gray_path.parent.mkdir(parents=True, exist_ok=True)
            gray_img.save(gray_path)

            # Optionally also save RGB to a separate folder
            if args.rgb_folder:
                rgb_path = Path(args.rgb_folder) / relative_path
                rgb_path.parent.mkdir(parents=True, exist_ok=True)
                output_image.save(rgb_path)

        except Exception as e:
            print(f"Error processing {input_path.name}: {e}")


argparser = argparse.ArgumentParser(description="FLUX.2 Image-to-Image Inference Script")
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
    "--model_name",
    type=str,
    default="black-forest-labs/FLUX.2-klein-4B",
    help="Name of the open-source FLUX.2 model to use for image editing.",
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
argparser.add_argument(
    "--seed",
    type=int,
    default=0,
    help="Random seed for reproducible generation.",
)
argparser.add_argument(
    "--lora_weights",
    type=str,
    default=None,
    help="Path to LoRA weights directory (containing pytorch_lora_weights.safetensors).",
)
argparser.add_argument(
    "--height",
    type=int,
    default=None,
    help="Height of generated images. If not set, matches input image height.",
)
argparser.add_argument(
    "--width",
    type=int,
    default=None,
    help="Width of generated images. If not set, matches input image width.",
)
argparser.add_argument(
    "--contrast_stretch",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Apply contrast stretching (p2/p98 rescale) to grayscale output. Use --no-contrast_stretch to disable.",
)
argparser.add_argument(
    "--rgb_folder",
    type=str,
    default=None,
    help="If set, also save RGB output to this folder (in addition to grayscale in --output_folder).",
)

args = argparser.parse_args()
main(args)
