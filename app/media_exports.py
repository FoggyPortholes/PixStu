"""Helpers for exporting PixStu animation frames."""

from __future__ import annotations

import json
import math
import os
import time
from typing import Iterable, List, Sequence, Tuple

from PIL import Image, ImageSequence

try:
    import rembg  # type: ignore[import]

    _HAS_REMBG = True
except Exception:  # pragma: no cover - optional dependency
    _HAS_REMBG = False

__all__ = [
    "OUTPUTS_DIR",
    "apply_palette",
    "estimate_bg_color",
    "extract_palette",
    "make_alpha",
    "smart_alpha",
    "mask_gif",
    "mask_video_to_outputs",
    "nn_resize",
    "save_gif",
    "save_sprite_sheet",
    "write_frames",
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

    rgba = img.convert("RGBA")
    alpha = rgba.getchannel("A")
    quantized = rgba.convert("RGB").quantize(palette=palette_donor)
    palettized = quantized.convert("RGBA")
    palettized.putalpha(alpha)
    return palettized


def _rgb(img: Image.Image) -> Image.Image:
    """Return a copy of *img* in RGB mode."""

    return img.convert("RGB") if img.mode != "RGB" else img


def estimate_bg_color(img: Image.Image) -> Tuple[int, int, int]:
    """Estimate the background colour of ``img`` by sampling its corners."""

    rgb = _rgb(img)
    width, height = rgb.size
    samples = [
        rgb.getpixel((0, 0)),
        rgb.getpixel((width - 1, 0)),
        rgb.getpixel((0, height - 1)),
        rgb.getpixel((width - 1, height - 1)),
    ]

    counts: dict[Tuple[int, int, int], int] = {}
    for colour in samples:
        counts[colour] = counts.get(colour, 0) + 1

    mode = max(counts.items(), key=lambda kv: kv[1])[0]
    if counts[mode] >= 2:
        return mode

    red = sum(colour[0] for colour in samples) // 4
    green = sum(colour[1] for colour in samples) // 4
    blue = sum(colour[2] for colour in samples) // 4
    return (red, green, blue)


def _dist(c1: Tuple[int, int, int], c2: Tuple[int, int, int]) -> float:
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(c1, c2)))


def make_alpha(img: Image.Image, bg: Tuple[int, int, int], tol: int = 20) -> Image.Image:
    """Return an RGBA copy of ``img`` where colours near ``bg`` are transparent."""

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


def _ensure_rgba_frames(frames: Iterable[Image.Image]) -> List[Image.Image]:
    return [to_rgba(frame) for frame in frames]


def write_frames(
    frames: Sequence[Image.Image],
    session_dir: str | None = None,
    prefix: str = "frame",
) -> str:
    """Write ``frames`` to ``session_dir`` and return the directory path.

    A ``frames.json`` manifest is written alongside the PNG files so that
    the front-end can locate the generated frames.  When ``session_dir`` is
    omitted a timestamped directory inside :data:`OUTPUTS_DIR` is created.
    """

    timestamp = int(time.time())

    if session_dir is None:
        session_dir = os.path.join(OUTPUTS_DIR, f"session_{timestamp}")

    frames_dir = os.path.join(session_dir, "frames")
    os.makedirs(frames_dir, exist_ok=True)

    manifest: List[str] = []
    for index, frame in enumerate(frames):
        file_path = os.path.join(frames_dir, f"{prefix}_{index:04d}.png")
        frame.save(file_path)
        manifest.append(os.path.basename(file_path))

    manifest_path = os.path.join(frames_dir, "frames.json")
    payload = {"frames": manifest, "timestamp": timestamp}
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    return frames_dir


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

    if columns <= 0:
        raise ValueError("columns must be a positive integer")

    if padding < 0:
        raise ValueError("padding cannot be negative")

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


def mask_gif(input_path: str, tolerance: int = 20, lock_palette: bool = True) -> str:
    """Return a transparent GIF created by removing a solid background."""

    with Image.open(input_path) as payload:
        base_rgb = payload.convert("RGB")
        bg_colour = estimate_bg_color(base_rgb)
        frames: List[Image.Image] = []
        for frame in ImageSequence.Iterator(payload):
            rgba = smart_alpha(
                frame.convert("RGBA"), tolerance=int(tolerance), bg=bg_colour
            )
            frames.append(rgba)

        duration = payload.info.get("duration", 90)
        loop = payload.info.get("loop", 0)

    return save_gif(frames, duration_ms=duration, loop=loop, lock_palette=lock_palette)


def mask_video_to_outputs(
    input_path: str,
    tolerance: int = 20,
    target_size: Tuple[int, int] | None = None,
    export_gif: bool = True,
    export_png_seq: bool = True,
    bg_override: Tuple[int, int, int] | None = None,
) -> Tuple[str | None, str | None]:
    """Mask a video file and optionally export a GIF and PNG sequence."""

    import imageio.v3 as iio

    reader = iio.imiter(input_path)
    frames: List[Image.Image] = []
    bg_colour: Tuple[int, int, int] | None = None

    for index, frame in enumerate(reader):
        image = Image.fromarray(frame)
        if target_size:
            image = image.resize(target_size, Image.NEAREST)

        if index == 0:
            bg_colour = bg_override or estimate_bg_color(image)

        if bg_colour is None:
            raise ValueError("Unable to determine background colour")

        frames.append(smart_alpha(image, tolerance=int(tolerance), bg=bg_colour))

    if not frames:
        raise ValueError("No frames extracted from video")

    gif_path: str | None = None
    png_dir: str | None = None

    if export_gif:
        gif_path = save_gif(frames, duration_ms=90, loop=0, lock_palette=True)

    if export_png_seq:
        timestamp = int(time.time())
        png_dir = os.path.join(OUTPUTS_DIR, f"masked_seq_{timestamp}")
        os.makedirs(png_dir, exist_ok=True)
        for idx, frame in enumerate(frames):
            frame.save(os.path.join(png_dir, f"frame_{idx:04d}.png"))

    return gif_path, png_dir

