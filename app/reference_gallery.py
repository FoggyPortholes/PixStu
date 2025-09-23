import os
import gradio as gr
from .paths import GALLERY

os.makedirs(GALLERY, exist_ok=True)


def gallery_ui(on_select=None):
    exts = (".png", ".webp", ".jpg", ".jpeg")
    files = [os.path.join(GALLERY, f) for f in os.listdir(GALLERY) if f.lower().endswith(exts)]
    files.sort()
    gal = gr.Gallery(value=files, label="Reference Characters", show_label=True, elem_id="reference-gallery")
    if on_select is not None:
        gal.select(on_select, inputs=None, outputs=None)
    return gal
