"""Utilities for masking out background colors from images."""

from typing import Tuple

from PIL import Image


RGBColor = Tuple[int, int, int]


def _ensure_rgb(image: Image.Image) -> Image.Image:
    """Return a copy of *image* converted to RGB."""

    if image.mode == "RGB":
        return image
    return image.convert("RGB")


def estimate_bg_color(image: Image.Image) -> RGBColor:
    """Estimate the background color of *image*.

    The estimation samples the outer border pixels and returns their average
    color.  This works well for the simple solid background images used in the
    tests.
    """

    rgb_image = _ensure_rgb(image)
    width, height = rgb_image.size
    if width == 0 or height == 0:
        raise ValueError("Image must have non-zero dimensions")

    border_pixels: list[RGBColor] = []
    pixels = rgb_image.load()

    for x in range(width):
        border_pixels.append(pixels[x, 0])
        if height > 1:
            border_pixels.append(pixels[x, height - 1])

    for y in range(1, height - 1):
        border_pixels.append(pixels[0, y])
        if width > 1:
            border_pixels.append(pixels[width - 1, y])

    r_total = sum(pixel[0] for pixel in border_pixels)
    g_total = sum(pixel[1] for pixel in border_pixels)
    b_total = sum(pixel[2] for pixel in border_pixels)
    count = len(border_pixels)
    return (
        int(round(r_total / count)),
        int(round(g_total / count)),
        int(round(b_total / count)),
    )


def _color_distance(c1: RGBColor, c2: RGBColor) -> int:
    return max(abs(a - b) for a, b in zip(c1, c2))


def make_alpha(image: Image.Image, bg_color: RGBColor, *, tol: int = 0) -> Image.Image:
    """Return a copy of *image* with the background made transparent.

    Pixels whose color is within ``tol`` of ``bg_color`` (using the maximum
    per-channel difference) have their alpha channel set to zero.
    """

    if tol < 0:
        raise ValueError("tol must be non-negative")

    rgba_image = image.convert("RGBA")
    width, height = rgba_image.size
    pixels = rgba_image.load()

    for y in range(height):
        for x in range(width):
            r, g, b, a = pixels[x, y]
            if _color_distance((r, g, b), bg_color) <= tol:
                pixels[x, y] = (r, g, b, 0)

    return rgba_image

