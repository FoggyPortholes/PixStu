"""Shared helpers for lightweight animation generation.

The original project provides fairly involved animation pipelines that rely on
heavy machine learning dependencies.  Those pipelines are overkill for the
purposes of automated tests where we simply need deterministic, low-cost
behaviour.  The utilities in this module provide small, dependency-light
helpers that are reused by the ``txt2gif``, ``img2gif`` and ``txt2vid``
wrappers.  They aim to keep graceful fallbacks so that the user experience is
predictable even when optional runtime dependencies (CUDA, diffusers, ffmpeg,
etc.) are missing.
"""

from __future__ import annotations

import math
import os
import random
import tempfile
from typing import Iterable, List, Optional, Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFont

from .generator import BulletProofGenerator
from .presets import get_preset, missing_assets


def _safe_frame_count(value: int | float | None, default: int = 6) -> int:
    """Normalize the requested number of frames.

    ``gradio`` widgets often send the slider value as a float, so we coerce the
    input to an integer, clamp it to a sensible minimum and use a default when
    the value cannot be interpreted.
    """

    try:
        count = int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default
    return max(count, 1)


def _seed_colour(seed: int) -> tuple[int, int, int]:
    rng = random.Random(int(seed))
    return tuple(rng.randint(0, 255) for _ in range(3))


def placeholder_frame(prompt: str, seed: int, size: tuple[int, int] = (512, 512)) -> Image.Image:
    """Create a deterministic placeholder frame.

    When the heavyweight diffusion pipelines are unavailable we still want the
    UI to render *something*.  This helper draws the prompt text on top of a
    colour derived from the seed so that successive frames differ while still
    remaining reproducible.
    """

    image = Image.new("RGB", size, _seed_colour(seed))
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    wrapped = "\n".join(prompt.split()) or "frame"
    text_w, text_h = draw.multiline_textsize(wrapped, font=font)
    x = (size[0] - text_w) / 2
    y = (size[1] - text_h) / 2
    draw.multiline_text((x, y), wrapped, font=font, fill="white", align="center")
    return image


def _load_generator(preset_name: Optional[str]) -> Optional[BulletProofGenerator]:
    if not preset_name:
        return None
    preset = get_preset(preset_name)
    if missing_assets(preset):  # pragma: no cover - depends on runtime assets
        raise RuntimeError("Preset assets missing.")
    try:
        return BulletProofGenerator(preset)
    except Exception:  # pragma: no cover - heavy dependency failure
        return None


def prompt_frames(
    preset_name: Optional[str],
    prompt: str,
    n_frames: int | float | None,
    seed: int | float | None,
    *,
    size: tuple[int, int] = (512, 512),
) -> List[Image.Image]:
    """Generate frames for a prompt.

    The helper attempts to use the real diffusion pipeline when available and
    falls back to deterministic placeholder images when that fails.  This keeps
    both development environments (where the ML models might not be present)
    and production scenarios happy.
    """

    count = _safe_frame_count(n_frames)
    base_seed = int(seed or 42)
    generator = _load_generator(preset_name)
    frames: List[Image.Image] = []
    for index in range(count):
        frame_seed = base_seed + index
        if generator is None:
            frame = placeholder_frame(prompt, frame_seed, size=size)
        else:  # pragma: no branch - straight-line logic
            try:
                frame = generator.generate(prompt, seed=frame_seed)
            except Exception:  # pragma: no cover - runtime inference error
                frame = placeholder_frame(prompt, frame_seed, size=size)
        frames.append(frame.convert("RGB"))
    return frames


def mutate_image_frames(
    image: Image.Image,
    n_frames: int | float | None,
) -> List[Image.Image]:
    """Produce a small wobble animation from a single image.

    The effect simply modulates brightness and rotates the image a little bit.
    This keeps the code light while still providing a pleasant preview.
    """

    count = _safe_frame_count(n_frames)
    img = image.convert("RGB")
    enhancer = ImageEnhance.Brightness(img)
    frames: List[Image.Image] = []
    for index in range(count):
        factor = 0.8 + 0.4 * math.sin(2 * math.pi * index / max(count, 1))
        adjusted = enhancer.enhance(factor)
        rotated = adjusted.rotate(math.sin(index) * 3, resample=Image.BICUBIC)
        frames.append(rotated)
    return frames


def save_gif(frames: Sequence[Image.Image], duration: int = 120) -> str:
    if not frames:
        raise ValueError("No frames to save")
    with tempfile.NamedTemporaryFile(suffix=".gif", delete=False) as handle:
        path = handle.name
    first, *rest = [frame.convert("P", palette=Image.ADAPTIVE) for frame in frames]
    first.save(path, save_all=True, append_images=rest, loop=0, duration=duration)
    return path


def save_mp4(frames: Sequence[Image.Image], fps: int = 4) -> Optional[str]:
    if not frames:
        return None
    try:
        import imageio.v2 as imageio
    except Exception:  # pragma: no cover - optional dependency missing
        return None
    try:
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as handle:
            path = handle.name
        writer = imageio.get_writer(path, fps=fps)
        for frame in frames:
            writer.append_data(np.asarray(frame.convert("RGB")))
        writer.close()
    except Exception:  # pragma: no cover - codec failure
        return None
    return path


def ensure_path_exists(path: Optional[str]) -> Optional[str]:
    if path and os.path.exists(path):
        return path
    return None


__all__ = [
    "placeholder_frame",
    "prompt_frames",
    "mutate_image_frames",
    "save_gif",
    "save_mp4",
    "ensure_path_exists",
]

