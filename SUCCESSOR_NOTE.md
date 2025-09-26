# PixStu v2.2.1 — Successor Note

## Pipelines
- ✍️ Txt2Img · 🖌️ Img2Img · 🎨 Inpainting · 🎞️ Txt2GIF

## Guardrails
- Single-character only, no text/captions/panels, blank background edges

## Infra
- Device fallback: CUDA → ZLUDA → MPS → CPU
- Self-healing: missing assets, dtype retry, sqlite corruption reset, import drift
- Cache: SQLite, robust, image helpers
- Downloads: Hugging Face LoRAs → loras/

## Quickstart
pip install pillow gradio imageio huggingface_hub
# Optional full GPU stack:
# pip uninstall torch torchvision torchaudio -y
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
# pip install diffusers transformers accelerate xformers

python tools/test_ui_smoke.py
python tools/test_device.py
python -m pixstu.app.studio
