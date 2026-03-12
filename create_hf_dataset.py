"""Create a HuggingFace DatasetDict from paired RGB/IR directory structure.

Converts:
  train/rgb/0/*.jpg + train/ir/0/*.jpg  →  HF Dataset with columns:
    - cond_image (Image): RGB input (condition for I2I)
    - target_image (Image): IR target (what the model learns to generate)
    - caption (str): the training prompt
"""

from datasets import Dataset, DatasetDict, Image
from pathlib import Path

DATA_ROOT = Path("/data/hanchong/other-infrared-datasets/processed-data/红外船舶数据库_Selected/train")
SAVE_PATH = "/data/hanchong/other-infrared-datasets/processed-data/红外船舶数据库_HFDataset"
PROMPT = "turn the visible image of Marine Vessel into sks infrared"

rgb_dir = DATA_ROOT / "rgb"
ir_dir = DATA_ROOT / "ir"

records = {"cond_image": [], "target_image": [], "caption": []}
for subdir in sorted(rgb_dir.iterdir()):
    if not subdir.is_dir():
        continue
    for img_file in sorted(subdir.glob("*.jpg")):
        ir_file = ir_dir / subdir.name / img_file.name
        if ir_file.exists():
            records["cond_image"].append(str(img_file))
            records["target_image"].append(str(ir_file))
            records["caption"].append(PROMPT)

ds = Dataset.from_dict(records)
ds = ds.cast_column("cond_image", Image())
ds = ds.cast_column("target_image", Image())

dd = DatasetDict({"train": ds})
dd.save_to_disk(SAVE_PATH)
print(f"Saved {len(ds)} examples to {SAVE_PATH}")
print(f"Columns: {ds.column_names}")
print(f"Sample: {ds[0]}")
