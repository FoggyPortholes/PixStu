import re
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

import gradio as gr
from PIL import Image

from .model_setup import GALLERY

GALLERY_PATH = Path(GALLERY)
GALLERY_PATH.mkdir(parents=True, exist_ok=True)

SUPPORTED_EXTENSIONS = (".png", ".webp", ".jpg", ".jpeg")


def _list_gallery_files() -> list[str]:
    files = [
        path
        for path in GALLERY_PATH.glob("*")
        if path.suffix.lower() in SUPPORTED_EXTENSIONS and path.is_file()
    ]
    files.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    return [str(path) for path in files]


def list_gallery() -> list[str]:
    """Return all gallery images sorted by most recent first."""

    return _list_gallery_files()


def _sanitize_label(label: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9._-]+", "_", label).strip("._-")
    return cleaned or "untitled"


def _ensure_pil(image: object) -> Image.Image:
    if isinstance(image, Image.Image):
        return image
    if isinstance(image, str):
        try:
            with Image.open(image) as handle:
                return handle.copy()
        except Exception as exc:  # pragma: no cover - runtime safeguard
            raise ValueError("Unable to load image from path") from exc
    raise ValueError("Unsupported image type for gallery save")


def save_to_gallery(image: object, label: Optional[str] = None) -> str:
    """Persist ``image`` to the gallery directory and return the saved path."""

    if image is None:
        raise ValueError("No image provided")
    pil_image = _ensure_pil(image)
    safe_label = _sanitize_label((label or "untitled").strip())
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    filename = f"{safe_label}_{timestamp}.png"
    destination = GALLERY_PATH / filename
    pil_image.save(destination)
    return str(destination)


def build_gallery(on_select: Optional[Callable] = None) -> gr.Gallery:
    gallery = gr.Gallery(
        value=list_gallery(),
        label="Reference Characters",
        show_label=True,
        elem_id="reference-gallery",
    )
    if on_select is not None:
        gallery.select(on_select, inputs=None, outputs=None)
    return gallery
