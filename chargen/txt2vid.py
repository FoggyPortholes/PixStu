"""Text-to-video (MP4) helper built atop the GIF frames."""

from __future__ import annotations

from ._animation_utils import ensure_path_exists, prompt_frames, save_gif, save_mp4


def txt2vid(
    preset_name: str,
    prompt: str,
    *,
    n_frames: int | float | None = 6,
    seed: int | float | None = 42,
    fps: int = 4,
    duration: int = 120,
) -> str:
    frames = prompt_frames(preset_name, prompt, n_frames, seed)
    video_path = save_mp4(frames, fps=fps)
    if video_path:
        return video_path
    path = save_gif(frames, duration=duration)
    return ensure_path_exists(path) or path


__all__ = ["txt2vid"]

