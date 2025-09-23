"""Utility helpers for creating simple animated GIFs from sprite images."""
from __future__ import annotations

import math
import os
import time
from typing import Optional, Tuple

from PIL import Image, ImageOps

ROOT = os.path.abspath(os.path.dirname(__file__))
PROJ = os.path.abspath(os.path.join(ROOT, ".."))
OUTPUTS_DIR = os.environ.get("PCS_OUTPUTS_DIR", os.path.join(PROJ, "outputs"))
os.makedirs(OUTPUTS_DIR, exist_ok=True)


def _timestamped_filename(prefix: str, ext: str) -> str:
    """Return a filename with millisecond precision to avoid collisions."""
    return f"{prefix}_{int(time.time() * 1000)}.{ext}"


def _prepare_base_frame(sprite: Image.Image, frame_size: int) -> Image.Image:
    """Resize the sprite to fit within ``frame_size`` while keeping transparency."""
    frame = Image.new("RGBA", (frame_size, frame_size), (0, 0, 0, 0))
    fitted = ImageOps.contain(sprite.convert("RGBA"), (frame_size, frame_size))
    offset = (
        (frame_size - fitted.width) // 2,
        (frame_size - fitted.height) // 2,
    )
    frame.paste(fitted, offset, fitted)
    return frame


def make_gif_from_sprite(
    *,
    sprite_path: str,
    preset_name: str,
    prompt: Optional[str],
    frames: int,
    frame_size: int,
    duration_ms: int,
    seed: Optional[int],
    seed_jitter: int,
    motion_mode: str,
    img_strength: float,
    lock_palette: bool,
    export_sheet: bool,
) -> Tuple[str, Optional[str]]:
    """Create an animated GIF and optional sprite sheet from a single sprite.

    The implementation is intentionally lightweight: it currently reuses the
    uploaded sprite for every frame while applying basic resizing so that the
    UI flow can be exercised even when a full animation pipeline is not
    available. The extra parameters are accepted to maintain API compatibility
    with the UI and can be used by more advanced implementations in the future.
    """

    if not sprite_path:
        raise ValueError("sprite_path is required")
    if not os.path.exists(sprite_path):
        raise FileNotFoundError(f"Sprite not found: {sprite_path}")

    frames = max(1, int(frames))
    frame_size = max(1, int(frame_size))
    duration_ms = max(1, int(duration_ms))

    sprite = Image.open(sprite_path)
    base_frame = _prepare_base_frame(sprite, frame_size)
    frame_sequence = [base_frame.copy() for _ in range(frames)]

    gif_filename = _timestamped_filename("pixstu_animation", "gif")
    gif_path = os.path.join(OUTPUTS_DIR, gif_filename)
    frame_sequence[0].save(
        gif_path,
        save_all=True,
        append_images=frame_sequence[1:],
        duration=duration_ms,
        loop=0,
        disposal=2,
        transparency=0,
    )

    sheet_path: Optional[str] = None
    if export_sheet:
        columns = min(frames, 8)
        rows = math.ceil(frames / columns)
        sheet = Image.new("RGBA", (frame_size * columns, frame_size * rows), (0, 0, 0, 0))
        for index, frame in enumerate(frame_sequence):
            row = index // columns
            col = index % columns
            sheet.paste(frame, (col * frame_size, row * frame_size), frame)
        sheet_filename = _timestamped_filename("pixstu_sheet", "png")
        sheet_path = os.path.join(OUTPUTS_DIR, sheet_filename)
        sheet.save(sheet_path)

    return gif_path, sheet_path
