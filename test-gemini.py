from dotenv import load_dotenv

load_dotenv(".env")

import argparse
import os
from io import BytesIO
from pathlib import Path
from time import sleep

from google import genai
from google.genai import types
from PIL import Image
from tqdm import tqdm

client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

MAX_CALLS = 10
GENERATION_MODEL = "gemini-2.5-flash-image"


class ImageGenerationPipeline:
    def __init__(self, model: str = GENERATION_MODEL):
        self.model = model
        self.generate_content_config = types.GenerateContentConfig(
            temperature=1.0,
            response_modalities=["image", "text"],
            response_mime_type="text/plain",
        )

    def __call__(
        self,
        input_image: str,
        prompt: str,
        output_path: str = None,
    ) -> Image.Image:
        with open(input_image, "rb") as image_file:
            image_bytes = image_file.read()

        contents = [
            types.Content(
                role="user",
                parts=[
                    types.Part.from_bytes(data=image_bytes, mime_type="image/png"),
                    types.Part.from_text(text=prompt),
                ],
            ),
        ]

        num_calls = 0
        while num_calls < MAX_CALLS:
            try:
                response = client.models.generate_content(
                    model=self.model,
                    contents=contents,
                    config=self.generate_content_config,
                )

                if (
                    response.candidates is None
                    or response.candidates[0].content is None
                    or response.candidates[0].content.parts is None
                ):
                    print("No content returned in generation response. Retrying after 5 seconds...")
                    sleep(5)
                    num_calls += 1
                    continue

                if (
                    response.candidates[0].content.parts[0].inline_data
                    and response.candidates[0].content.parts[0].inline_data.data
                ):
                    part = response.candidates[0].content.parts[0]
                    output_image = Image.open(BytesIO((part.inline_data.data)))
                    print(f"Generated image size: {output_image.size}")

                    if output_path is not None:
                        output_image.save(output_path)
                        print(f"Saved generated image to {output_path}")

                    return output_image

                else:
                    print(f"No inline data found in generation response. Retrying after 5 seconds...")
                    sleep(5)
                    num_calls += 1
                    continue

            except Exception as e:
                print(f"No response received from generation API: {e}. Retrying after 5 seconds...")
                sleep(5)
                num_calls += 1


pipe_generate = ImageGenerationPipeline()


def main(args):
    input_folder = Path(args.input_folder)
    input_paths = list(input_folder.glob("*.jpg"))

    output_folder = Path(args.output_folder)
    os.makedirs(output_folder, exist_ok=True)

    for input_path in tqdm(input_paths, total=len(input_paths)):
        try:
            image_path = input_path
            output_path = output_folder / input_path.name

            pipe_generate(
                input_image=str(image_path),
                prompt=args.prompt,
                output_path=str(output_path),
            )

        except Exception as e:
            print(f"Error processing {input_path.name}: {e}")


argparser = argparse.ArgumentParser(description="Gemini Image Edit Inference Script")
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

args = argparser.parse_args()
main(args)
