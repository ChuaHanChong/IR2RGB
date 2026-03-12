# IR to RGB Image Conversion

## Venv

### test FLUX

```bash
conda create --name flux python=3.12 -y
conda activate flux

pip install torch torchvision
pip install transformers diffusers accelerate torchmetrics 
pip install opencv-python-headless pandas
pip install packaging
pip install ninja
pip install flash-attn --no-build-isolation

hf download black-forest-labs/FLUX.2-klein-4B
hf download black-forest-labs/FLUX.2-klein-9B
```

## References

1. [HF DreamBooth Training](https://github.com/huggingface/diffusers/tree/main/examples/dreambooth)