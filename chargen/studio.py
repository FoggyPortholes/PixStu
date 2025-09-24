import gradio as gr
from pathlib import Path
from datetime import datetime
from PIL import Image

from .inpaint import inpaint

GALLERY_DIR = Path("outputs/gallery")
META = GALLERY_DIR / "index.json"
GALLERY_DIR.mkdir(parents=True, exist_ok=True)

# Lightweight metadata index (no heavy deps)
import json

def _load_index():
    if META.exists():
        try:


    return demo


if __name__ == "__main__":
    studio().launch()
