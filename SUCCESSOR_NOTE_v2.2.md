# PixStu v2.2.0-pre Successor Note

## New
- Pipelines: txt2img, img2img, inpainting, txt2gif
- Retro-modern UI tabs for each
- Guardrails, cache, device fallback, self-heal retained

## Quickstart
pip install pillow gradio huggingface_hub imageio
# Optional GPU:
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
# pip install diffusers transformers accelerate xformers

python -m pixstu.app.studio
