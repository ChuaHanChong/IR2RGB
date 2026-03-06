import argparse
from pathlib import Path

import torch
from diffusers import Flux2KleinPipeline, Flux2Pipeline
from PIL import Image
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
    
    pipe.enable_model_cpu_offload()  # save some VRAM by offloading the model to CPU

    output_folder = Path(args.output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    for input_path in tqdm(input_paths, desc="Generating images"):
        try:
            with torch.inference_mode():
                ouput_image = pipe(
                    prompt=args.prompt,
                    image=[Image.open(input_path).convert("RGB")],
                    height=1024,
                    width=1024,
                    guidance_scale=args.guidance_scale,
                    num_inference_steps=args.num_inference_steps,
                    generator=torch.Generator(device=device).manual_seed(0),
                ).images[0]

            relative_path = input_path.relative_to(input_folder)
            output_path = output_folder / relative_path
            if not output_path.parent.exists():
                output_path.parent.mkdir(parents=True, exist_ok=True)
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

args = argparser.parse_args()
main(args)
