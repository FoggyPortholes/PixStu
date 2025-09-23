"""Utilities for estimating background colours and generating alpha masks."""

from __future__ import annotations

import math
from typing import Tuple

from PIL import Image

RGBColor = Tuple[int, int, int]


def estimate_bg_color(img: Image.Image) -> RGBColor:
    """Estimate the predominant background colour of *img*.

    The heuristic mirrors :mod:`app.media_masking` and :mod:`app.media_exports`:
    the four corners are sampled and their mode is returned.  When no colour
    appears at least twice the average of the corners is used instead.  Keeping
    the approach consistent across modules avoids subtle behavioural
    differences when generating masks from different entry points.
    """

    rgb = img.convert("RGB")
    width, height = rgb.size
    if width == 0 or height == 0:
        raise ValueError("cannot estimate background colour of an empty image")

    samples = [
        rgb.getpixel((0, 0)),
        rgb.getpixel((width - 1, 0)),
        rgb.getpixel((0, height - 1)),
        rgb.getpixel((width - 1, height - 1)),
    ]

    counts: dict[RGBColor, int] = {}
    for colour in samples:
        counts[colour] = counts.get(colour, 0) + 1

    mode = max(counts.items(), key=lambda kv: kv[1])[0]
    if counts[mode] >= 2:
        return mode

    red = sum(colour[0] for colour in samples) // 4
    green = sum(colour[1] for colour in samples) // 4
    blue = sum(colour[2] for colour in samples) // 4
    return (red, green, blue)


def _dist(c1: RGBColor, c2: RGBColor) -> float:
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(c1, c2)))


def make_alpha(img: Image.Image, bg_color: RGBColor | None = None, tol: int = 0) -> Image.Image:
    """Return a copy of *img* with the background made transparent.

    Args:
        img: Source Pillow image.
        bg_color: Background colour to remove.  If omitted the colour is
            estimated with :func:`estimate_bg_color`.
        tol: Maximum per-channel deviation from the background colour that will
            still be treated as background.
    """

    if bg_color is None:
        bg_color = estimate_bg_color(img)

    rgba = img.convert("RGBA")
    pixels = rgba.load()
    width, height = rgba.size

    for y in range(height):
        for x in range(width):
            r, g, b, a = pixels[x, y]
            if _dist((r, g, b), bg_color) <= tol:
                pixels[x, y] = (r, g, b, 0)
            else:
                pixels[x, y] = (r, g, b, a)

    return rgba
