# PixStu v2.1.0 Successor Note

## Immutable Rules
- Single-character only
- Blank backgrounds
- No text/captions/multi-panels
- Retro-modern UI
- Fallback: CUDA → ZLUDA → MPS → CPU

## Features
- Guardrails
- Retro UI
- Inpainting with Diffusers fallback
- LoRA scaffold + Downloads
- Cache with self-heal
- Self-healing framework

## Quickstart
pip install pillow gradio huggingface_hub
# Optional GPU stack:
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
# pip install diffusers transformers accelerate xformers

python tools/test_ui_smoke.py
python tools/test_device.py
python -m pixstu.app.studio
