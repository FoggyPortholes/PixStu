import os
from typing import Callable, Iterable, Optional

import gradio as gr

from .model_setup import GALLERY

os.makedirs(GALLERY, exist_ok=True)

SUPPORTED_EXTENSIONS = (".png", ".webp", ".jpg", ".jpeg")


def _list_gallery_files() -> Iterable[str]:
    files = [os.path.join(GALLERY, name) for name in os.listdir(GALLERY) if name.lower().endswith(SUPPORTED_EXTENSIONS)]
    files.sort()
    return files


def build_gallery(on_select: Optional[Callable] = None) -> gr.Gallery:
    gallery = gr.Gallery(
        value=list(_list_gallery_files()),
        label="Reference Characters",
        show_label=True,
        elem_id="reference-gallery",
    )
    if on_select is not None:
        gallery.select(on_select, inputs=None, outputs=None)
    return gallery
