"""Diffusers-based text-to-video placeholder implementation."""

from __future__ import annotations

from ._animation_utils import ensure_path_exists, prompt_frames, save_gif, save_mp4


def txt2vid_diffusers(
    prompt: str,
    *,
    n_frames: int | float | None = 6,
    seed: int | float | None = 42,
    fps: int = 4,
    duration: int = 120,
) -> str:
    """Lightweight stand-in for the heavier diffusers workflow.

    We generate placeholder frames using the same helper used by the GIF flow
    and then attempt to encode them as an MP4.  If the system lacks an MP4
    encoder we gracefully fall back to a GIF so that the caller always receives
    a valid file path.
    """

    frames = prompt_frames(None, prompt, n_frames, seed)
    video_path = save_mp4(frames, fps=fps)
    if video_path:
        return video_path
    path = save_gif(frames, duration=duration)
    return ensure_path_exists(path) or path


__all__ = ["txt2vid_diffusers"]

