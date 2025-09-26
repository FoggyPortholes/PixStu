# PixStu v2.2.3 — Successor Note

## What Changed
- Added requirements.txt for easy installation
- Confirmed UI works with Txt2Img, Img2Img, Inpainting, Txt2GIF, Gallery, Downloads

## Quickstart
pip install -r requirements.txt

# Optional GPU stack:
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
# pip install diffusers transformers accelerate xformers

python tools/test_ui_smoke.py
python tools/test_device.py
python -m pixstu.app.studio
