import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable, Optional

import gradio as gr
from PIL import Image

from pixstu_config import load_config
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


def cleanup_gallery(now: Optional[datetime] = None) -> list[str]:
    """Remove gallery entries according to the configured retention policy."""

    config = load_config().get("gallery", {})
    max_items = int(config.get("max_items", 0) or 0)
    ttl_days = config.get("ttl_days")
    cutoff = None
    if ttl_days:
        try:
            cutoff = (now or datetime.utcnow()) - timedelta(days=float(ttl_days))
        except (TypeError, ValueError):
            cutoff = None

    files = [
        path
        for path in GALLERY_PATH.glob("*")
        if path.suffix.lower() in SUPPORTED_EXTENSIONS and path.is_file()
    ]
    files.sort(key=lambda path: path.stat().st_mtime, reverse=True)

    removed: list[str] = []

    if cutoff is not None:
        for path in list(files):
            mtime = datetime.utcfromtimestamp(path.stat().st_mtime)
            if mtime < cutoff:
                path.unlink(missing_ok=True)
                files.remove(path)
                removed.append(str(path))

    if max_items and len(files) > max_items:
        for path in files[max_items:]:
            path.unlink(missing_ok=True)
            removed.append(str(path))
        files = files[:max_items]

    return removed


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


def _main() -> None:  # pragma: no cover - CLI utility
    removed = cleanup_gallery()
    if removed:
        for path in removed:
            print(f"Removed {path}")
    else:
        print("Gallery already clean")


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    _main()
