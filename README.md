# IR to RGB Image Conversion

## Venv

### HunyuanImage-3.0

```bash
conda create --name hunyuan python=3.12 -y
conda activate hunyuan

pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r HunyuanImage-3.0/requirements.txt

hf download tencent/HunyuanImage-3.0-Instruct
```

### Qwen-Image

```bash
conda create --name qwen python=3.12 -y
conda activate qwen

pip install torch torchvision
pip install transformers diffusers accelerate

hf download Qwen/Qwen-Image-Edit-2511
```

### test FLUX

```bash
conda create --name flux python=3.12 -y
conda activate flux

pip install torch torchvision
pip install transformers diffusers accelerate torchmetrics opencv-python-headless pandas

pip install packaging
pip install ninja
pip install flash-attn --no-build-isolation

hf download black-forest-labs/FLUX.2-klein-4B
hf download black-forest-labs/FLUX.2-klein-9B
```

### Gemini

```bash
conda create --name gemini python=3.12 -y
conda activate gemini

pip install google-genai python-dotenv
```
