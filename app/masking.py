"""Helpers for automatically masking sprites, GIFs, and videos."""

from __future__ import annotations

import math
import os
import time
import uuid
from collections import Counter
from typing import Iterable, Sequence, Tuple

import imageio
from PIL import Image, ImageSequence

from .gif_tools import nn_resize, save_gif

ROOT = os.path.abspath(os.path.dirname(__file__))
PROJ = os.path.abspath(os.path.join(ROOT, ".."))
OUTPUTS_DIR = os.environ.get("PCS_OUTPUTS_DIR", os.path.join(PROJ, "outputs"))
os.makedirs(OUTPUTS_DIR, exist_ok=True)

_MAX_DISTANCE = math.sqrt(3 * (255 ** 2))


def _clamp_tolerance(tolerance: int) -> float:
    tol = max(0, min(100, int(tolerance)))
    return (tol / 100.0) * _MAX_DISTANCE


def _color_distance(a: Sequence[int], b: Sequence[int]) -> float:
    return math.sqrt(sum((int(x) - int(y)) ** 2 for x, y in zip(a, b)))


def _iter_sample_coords(width: int, height: int) -> Iterable[Tuple[int, int]]:
    candidates = {
        (0, 0),
        (width - 1, 0),
        (0, height - 1),
        (width - 1, height - 1),
        (width // 2, 0),
        (width // 2, height - 1),
        (0, height // 2),
        (width - 1, height // 2),
        (width // 2, height // 2),
    }
    for x, y in candidates:
        if 0 <= x < width and 0 <= y < height:
            yield x, y


def estimate_bg_color(image: Image.Image) -> Tuple[int, int, int]:
    """Estimate a dominant background color for ``image``."""

    rgba = image.convert("RGBA")
    width, height = rgba.size
    samples: Counter[Tuple[int, int, int]] = Counter()

    for x, y in _iter_sample_coords(width, height):
        r, g, b, a = rgba.getpixel((x, y))
        if a == 0:
            continue
        samples[(r, g, b)] += 1

    if not samples:
        return (0, 0, 0)

    return samples.most_common(1)[0][0]


def make_alpha(image: Image.Image, bg_color: Sequence[int], tol: int) -> Image.Image:
    """Return ``image`` with pixels near ``bg_color`` made transparent."""

    rgba = image.convert("RGBA")
    threshold = _clamp_tolerance(tol)
    bg = tuple(int(c) for c in bg_color[:3])
    pixels = rgba.load()
    for y in range(rgba.height):
        for x in range(rgba.width):
            r, g, b, a = pixels[x, y]
            if a == 0:
                continue
            if _color_distance((r, g, b), bg) <= threshold:
                pixels[x, y] = (r, g, b, 0)
    return rgba


def _timestamped_dir(prefix: str) -> str:
    stamp = time.strftime("%Y%m%d-%H%M%S")
    ident = uuid.uuid4().hex[:6]
    directory = f"{prefix}_{stamp}_{ident}"
    path = os.path.join(OUTPUTS_DIR, directory)
    os.makedirs(path, exist_ok=True)
    return path


def save_png_sequence(frames: Sequence[Image.Image], prefix: str = "masked_seq") -> str:
    """Save ``frames`` to a uniquely named directory and return its path."""

    directory = _timestamped_dir(prefix)
    for index, frame in enumerate(frames):
        filename = f"frame_{index:04d}.png"
        frame.save(os.path.join(directory, filename))
    return directory


def mask_gif(
    path: str,
    *,
    tolerance: int = 20,
    lock_palette: bool = True,
    export_png_seq: bool = False,
) -> Tuple[str, str | None]:
    """Mask a GIF ``path`` and return the output GIF path and optional PNG folder."""

    if not os.path.exists(path):
        raise FileNotFoundError(path)

    with Image.open(path) as gif:
        frames = [frame.convert("RGBA") for frame in ImageSequence.Iterator(gif)]
        if not frames:
            raise ValueError("GIF contains no frames")
        bg = estimate_bg_color(frames[0])
        masked = [make_alpha(frame, bg, tolerance) for frame in frames]
        duration = int(gif.info.get("duration", 120))
        loop = int(gif.info.get("loop", 0))

    gif_path = save_gif(masked, duration_ms=duration, loop=loop, lock_palette=lock_palette)
    png_dir = save_png_sequence(masked) if export_png_seq else None
    return gif_path, png_dir


def _ensure_size(frame: Image.Image, size: Tuple[int, int]) -> Image.Image:
    return nn_resize(frame, size) if size and frame.size != size else frame.copy()


def mask_video_to_outputs(
    path: str,
    *,
    tolerance: int = 20,
    target_size: Tuple[int, int] | None = None,
    export_gif: bool = True,
    export_png_seq: bool = True,
) -> Tuple[str | None, str | None]:
    """Mask a video and optionally export GIF and PNG sequence outputs."""

    if not os.path.exists(path):
        raise FileNotFoundError(path)

    reader = imageio.get_reader(path)
    try:
        metadata = reader.get_meta_data()
    except Exception:
        metadata = {}

    fps = metadata.get("fps") or metadata.get("fps_n") or 8
    try:
        fps = float(fps)
        duration_ms = max(1, int(1000 / fps))
    except Exception:
        duration_ms = 120

    frames: list[Image.Image] = []
    try:
        for frame in reader:
            pil_frame = Image.fromarray(frame)
            if target_size:
                pil_frame = _ensure_size(pil_frame, target_size)
            frames.append(pil_frame.convert("RGBA"))
    finally:
        reader.close()

    if not frames:
        raise ValueError("Video contains no frames")

    bg = estimate_bg_color(frames[0])
    masked_frames = [make_alpha(frame, bg, tolerance) for frame in frames]

    gif_path: str | None = None
    if export_gif:
        gif_path = save_gif(masked_frames, duration_ms=duration_ms, loop=0, lock_palette=True)

    png_dir: str | None = None
    if export_png_seq:
        png_dir = save_png_sequence(masked_frames)

    return gif_path, png_dir


__all__ = [
    "estimate_bg_color",
    "make_alpha",
    "mask_gif",
    "mask_video_to_outputs",
    "save_png_sequence",
]

