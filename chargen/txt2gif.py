"""Text-to-GIF helper."""

from __future__ import annotations

from ._animation_utils import ensure_path_exists, prompt_frames, save_gif


def txt2gif(
    preset_name: str,
    prompt: str,
    *,
    n_frames: int | float | None = 6,
    seed: int | float | None = 42,
    duration: int = 120,
) -> str:
    """Generate a small GIF for the given prompt.

    The function intentionally keeps its dependencies light so that the Studio
    UI remains usable in CPU-only environments.  It will use the configured
    preset if available and otherwise fall back to placeholder frames.
    """

    frames = prompt_frames(preset_name, prompt, n_frames, seed)
    path = save_gif(frames, duration=duration)
    return ensure_path_exists(path) or path


__all__ = ["txt2gif"]

