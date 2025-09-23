"""Helpers for exporting PixStu animation frames."""

from __future__ import annotations

import os
import time
from typing import Iterable, List, Sequence, Tuple

from PIL import Image

__all__ = [
    "OUTPUTS_DIR",
    "apply_palette",
    "extract_palette",
    "nn_resize",
    "save_gif",
    "save_sprite_sheet",
    "to_rgba",
]


OUTPUTS_DIR = os.environ.get(
    "PCS_OUTPUTS_DIR",
    os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "outputs")),
)
os.makedirs(OUTPUTS_DIR, exist_ok=True)


def to_rgba(img: Image.Image) -> Image.Image:
    """Return a copy of ``img`` in RGBA mode."""

    return img.convert("RGBA") if img.mode != "RGBA" else img


def nn_resize(img: Image.Image, size: Tuple[int, int]) -> Image.Image:
    """Nearest-neighbour resize helper that preserves crisp edges."""

    return img.resize(size, Image.NEAREST)


def extract_palette(img: Image.Image, max_colors: int = 32) -> Image.Image:
    """Return a palettized version of *img* suitable for palette donation."""

    return img.convert("P", palette=Image.ADAPTIVE, colors=max_colors)


def apply_palette(img: Image.Image, palette_donor: Image.Image) -> Image.Image:
    """Quantize ``img`` using the palette from ``palette_donor``."""

    quantized = img.convert("RGB").quantize(palette=palette_donor)
    return quantized.convert("RGBA")


def _ensure_rgba_frames(frames: Iterable[Image.Image]) -> List[Image.Image]:
    return [to_rgba(frame) for frame in frames]


def save_gif(
    frames: Sequence[Image.Image],
    duration_ms: int = 90,
    loop: int = 0,
    basename: str = "pixstu_anim",
    lock_palette: bool = True,
) -> str:
    """Write ``frames`` to an animated GIF and return the output path."""

    if not frames:
        raise ValueError("No frames to save")

    timestamp = int(time.time())
    out_path = os.path.join(OUTPUTS_DIR, f"{basename}_{timestamp}.gif")

    rgba_frames = _ensure_rgba_frames(frames)

    if lock_palette:
        palette = extract_palette(rgba_frames[0], max_colors=32)
        palettized = [apply_palette(frame, palette) for frame in rgba_frames]
    else:
        palettized = [frame.convert("P", palette=Image.ADAPTIVE, colors=32) for frame in rgba_frames]

    palettized[0].save(
        out_path,
        save_all=True,
        append_images=palettized[1:],
        duration=duration_ms,
        loop=loop,
        disposal=2,
        transparency=0,
        optimize=False,
    )
    return out_path


def save_sprite_sheet(
    frames: Sequence[Image.Image],
    columns: int = 4,
    padding: int = 0,
    bgcolor: Tuple[int, int, int, int] = (0, 0, 0, 0),
    basename: str = "pixstu_sheet",
) -> str:
    """Save ``frames`` to a sprite sheet PNG and return the output path."""

    if not frames:
        raise ValueError("No frames to save")

    rgba_frames = _ensure_rgba_frames(frames)
    width, height = rgba_frames[0].size
    rows = (len(rgba_frames) + columns - 1) // columns

    sheet_width = columns * width + padding * (columns - 1)
    sheet_height = rows * height + padding * (rows - 1)
    sheet = Image.new("RGBA", (sheet_width, sheet_height), bgcolor)

    for index, frame in enumerate(rgba_frames):
        row = index // columns
        col = index % columns
        x = col * (width + padding)
        y = row * (height + padding)
        sheet.paste(frame, (x, y), frame if frame.mode == "RGBA" else None)

    timestamp = int(time.time())
    out_path = os.path.join(OUTPUTS_DIR, f"{basename}_{timestamp}.png")
    sheet.save(out_path)
    return out_path

