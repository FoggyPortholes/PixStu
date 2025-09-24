"""Image-to-GIF helper."""

from __future__ import annotations

from typing import Optional

from PIL import Image

from ._animation_utils import ensure_path_exists, mutate_image_frames, prompt_frames, save_gif


def img2gif(
    preset_name: str,
    image: Optional[Image.Image],
    prompt: str,
    *,
    n_frames: int | float | None = 6,
    seed: int | float | None = 42,
    duration: int = 120,
) -> str:
    """Animate a still image with a subtle wobble effect.

    If an image is not provided we gracefully fall back to the text-to-GIF
    implementation so that the function always returns a usable animation.
    """

    frames = []
    if image is not None:
        frames = mutate_image_frames(image, n_frames)
    if not frames:
        frames = prompt_frames(preset_name, prompt, n_frames, seed)
    path = save_gif(frames, duration=duration)
    return ensure_path_exists(path) or path


__all__ = ["img2gif"]

