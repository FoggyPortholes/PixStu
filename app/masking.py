"""Utilities for estimating background colours and generating alpha masks."""
from collections import Counter
from typing import Iterable, Tuple

from PIL import Image

RGBColor = Tuple[int, int, int]


def _iter_border_pixels(img: Image.Image) -> Iterable[RGBColor]:
    """Yield RGB pixels from the border of *img*.

    The helper normalises the image to RGB so the public functions can work
    with any Pillow mode that supports RGB data.
    """

    if img.width == 0 or img.height == 0:
        return []

    rgb_img = img.convert("RGB")
    width, height = rgb_img.size
    pixels = rgb_img.load()

    # Top and bottom rows
    for x in range(width):
        yield pixels[x, 0]
        if height > 1:
            yield pixels[x, height - 1]

    # Left and right columns (excluding already processed corners)
    for y in range(1, height - 1):
        yield pixels[0, y]
        if width > 1:
            yield pixels[width - 1, y]


def estimate_bg_color(img: Image.Image) -> RGBColor:
    """Estimate the predominant background colour of *img*.

    The heuristic samples the outer border of the image under the assumption
    that backgrounds typically extend to the edges.  The most common colour on
    that border is returned.
    """

    border_pixels = list(_iter_border_pixels(img))
    if not border_pixels:
        # Fallback to converting the image to RGB and returning the top-left
        # pixel if the image had no area.
        return img.convert("RGB").getpixel((0, 0))

    return Counter(border_pixels).most_common(1)[0][0]


def _is_within_tolerance(color: RGBColor, reference: RGBColor, tol: int) -> bool:
    return all(abs(component - ref) <= tol for component, ref in zip(color, reference))


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
            if _is_within_tolerance((r, g, b), bg_color, tol):
                pixels[x, y] = (r, g, b, 0)
            else:
                pixels[x, y] = (r, g, b, a)

    return rgba
