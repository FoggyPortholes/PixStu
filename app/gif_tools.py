"""Utilities for saving animation previews produced by PixStu."""
from __future__ import annotations

import os
import time
import uuid
from typing import Sequence, Tuple

from PIL import Image

ROOT = os.path.abspath(os.path.dirname(__file__))
PROJ = os.path.abspath(os.path.join(ROOT, ".."))
OUTPUTS_DIR = os.environ.get("PCS_OUTPUTS_DIR", os.path.join(PROJ, "outputs"))
os.makedirs(OUTPUTS_DIR, exist_ok=True)


def _normalize_frames(frames: Sequence[Image.Image]) -> Tuple[int, int, Sequence[Image.Image]]:
    if not frames:
        raise ValueError("at least one frame is required")

    base_w, base_h = frames[0].size
    normalized = []
    for frame in frames:
        if frame.size != (base_w, base_h):
            normalized.append(nn_resize(frame, (base_w, base_h)))
        else:
            normalized.append(frame)
    return base_w, base_h, normalized


def nn_resize(img: Image.Image, size: Tuple[int, int]) -> Image.Image:
    """Resize ``img`` using nearest-neighbor sampling."""

    if img.size == size:
        return img.copy()
    return img.resize(size, Image.NEAREST)


def _unique_path(suffix: str) -> str:
    stamp = time.strftime("%Y%m%d-%H%M%S")
    ident = uuid.uuid4().hex[:8]
    filename = f"pcs-{stamp}-{ident}.{suffix}"
    return os.path.join(OUTPUTS_DIR, filename)


def save_gif(frames: Sequence[Image.Image], *, duration_ms: int = 120, loop: int = 0) -> str:
    """Save ``frames`` as an animated GIF in the configured outputs directory."""

    base_w, base_h, normalized = _normalize_frames(frames)
    path = _unique_path("gif")
    normalized[0].save(
        path,
        save_all=True,
        append_images=list(normalized[1:]),
        duration=duration_ms,
        loop=loop,
        disposal=2,
    )
    return path


def save_sprite_sheet(
    frames: Sequence[Image.Image],
    *,
    columns: int = 4,
    padding: int = 0,
    background: Tuple[int, int, int, int] | Tuple[int, int, int] = (0, 0, 0, 0),
) -> str:
    """Arrange ``frames`` into a sprite sheet image saved to disk."""

    if columns <= 0:
        raise ValueError("columns must be positive")

    base_w, base_h, normalized = _normalize_frames(frames)
    columns = max(1, columns)
    rows = (len(normalized) + columns - 1) // columns

    sheet_w = columns * base_w + padding * (columns - 1)
    sheet_h = rows * base_h + padding * (rows - 1)
    mode = "RGBA" if len(background) == 4 else "RGB"
    sheet = Image.new(mode, (sheet_w, sheet_h), background)

    for idx, frame in enumerate(normalized):
        col = idx % columns
        row = idx // columns
        x = col * (base_w + padding)
        y = row * (base_h + padding)
        if frame.mode != mode:
            frame = frame.convert(mode)
        sheet.paste(frame, (x, y))

    path = _unique_path("png")
    sheet.save(path)
    return path
