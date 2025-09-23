"""Background masking utilities for GIFs and videos."""

from __future__ import annotations

import math
import os
import time
from typing import List, Tuple

from PIL import Image, ImageSequence

from .media_exports import save_gif

try:
    import rembg  # type: ignore[import]

    _HAS_REMBG = True
except Exception:  # pragma: no cover - optional dependency
    _HAS_REMBG = False

OUTPUTS_DIR = os.environ.get(
    "PCS_OUTPUTS_DIR",
    os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "outputs")),
)
os.makedirs(OUTPUTS_DIR, exist_ok=True)


def _rgb(img: Image.Image) -> Image.Image:
    """Return ``img`` converted to RGB if necessary."""

    return img.convert("RGB") if img.mode != "RGB" else img


def estimate_bg_color(img: Image.Image) -> Tuple[int, int, int]:
    """Estimate the dominant background color of ``img``.

    The four corners are sampled and their mode returned. If there is no
    majority color the average of the samples is used instead.
    """

    rgb = _rgb(img)
    width, height = rgb.size
    samples = [
        rgb.getpixel((0, 0)),
        rgb.getpixel((width - 1, 0)),
        rgb.getpixel((0, height - 1)),
        rgb.getpixel((width - 1, height - 1)),
    ]

    counts: dict[Tuple[int, int, int], int] = {}
    for color in samples:
        counts[color] = counts.get(color, 0) + 1

    mode = max(counts.items(), key=lambda kv: kv[1])[0]
    if counts[mode] >= 2:
        return mode

    r = sum(color[0] for color in samples) // 4
    g = sum(color[1] for color in samples) // 4
    b = sum(color[2] for color in samples) // 4
    return (r, g, b)


def _dist(c1: Tuple[int, int, int], c2: Tuple[int, int, int]) -> float:
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(c1, c2)))


def make_alpha(img: Image.Image, bg: Tuple[int, int, int], tol: int = 20) -> Image.Image:
    """Return an RGBA image with background pixels made transparent."""

    rgba = img.convert("RGBA")
    pixels = rgba.load()
    width, height = rgba.size
    for y in range(height):
        for x in range(width):
            r, g, b, a = pixels[x, y]
            if _dist((r, g, b), bg) <= tol:
                pixels[x, y] = (r, g, b, 0)
    return rgba


def smart_alpha(
    img: Image.Image,
    tolerance: int = 20,
    bg: Tuple[int, int, int] | None = None,
) -> Image.Image:
    """Return an alpha-masked image using ``rembg`` when available."""

    if _HAS_REMBG:
        return rembg.remove(img.convert("RGBA"))

    if bg is None:
        bg = estimate_bg_color(img)

    return make_alpha(img, bg, tol=int(tolerance))


def mask_gif(input_path: str, tolerance: int = 20, lock_palette: bool = True) -> str:
    """Apply background masking to a GIF file and return the output path."""

    with Image.open(input_path) as payload:
        info = dict(payload.info)
        bg = estimate_bg_color(payload.convert("RGB"))
        frames: List[Image.Image] = []
        for frame in ImageSequence.Iterator(payload):
            rgba = smart_alpha(
                frame.convert("RGBA"), tolerance=int(tolerance), bg=bg
            )
            frames.append(rgba)

    out_path = save_gif(
        frames,
        duration_ms=info.get("duration", 90),
        loop=info.get("loop", 0),
        lock_palette=lock_palette,
    )
    return out_path


def mask_video_to_outputs(
    input_path: str,
    tolerance: int = 20,
    target_size: Tuple[int, int] | None = None,
    export_gif: bool = True,
    export_png_seq: bool = True,
    bg_override: Tuple[int, int, int] | None = None,
) -> Tuple[str | None, str | None]:
    """Mask frames from ``input_path`` and export as GIF and/or PNG sequence."""

    import imageio.v3 as iio

    frames: List[Image.Image] = []
    bg: Tuple[int, int, int] | None = None

    for index, frame in enumerate(iio.imiter(input_path)):
        img = Image.fromarray(frame)
        if target_size:
            img = img.resize(target_size, Image.NEAREST)
        if index == 0:
            bg = bg_override or estimate_bg_color(img)
        assert bg is not None
        frames.append(smart_alpha(img, tolerance=int(tolerance), bg=bg))

    gif_path: str | None = None
    png_dir: str | None = None

    if export_gif and frames:
        gif_path = save_gif(frames, duration_ms=90, loop=0, lock_palette=True)

    if export_png_seq and frames:
        timestamp = int(time.time())
        png_dir = os.path.join(OUTPUTS_DIR, f"masked_seq_{timestamp}")
        os.makedirs(png_dir, exist_ok=True)
        for idx, frame in enumerate(frames):
            frame.save(os.path.join(png_dir, f"frame_{idx:04d}.png"))

    return gif_path, png_dir
