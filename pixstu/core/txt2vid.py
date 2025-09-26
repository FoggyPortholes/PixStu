"""
Text-to-GIF pipeline: jitter txt2img frames into a GIF.
"""
from __future__ import annotations

import base64
import io
import random
import time
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image

from .txt2img import txt2img

try:  # optional dependency
    import imageio.v2 as imageio
except Exception:  # pragma: no cover - exercised when dependency missing
    imageio = None  # type: ignore[assignment]


def txt2gif(
    prompt: str,
    frames: int = 12,
    duration_ms: int = 100,
    seed: Optional[int] = None,
) -> Tuple[Image.Image, Dict[str, Any]]:
    start = time.time()
    rng = random.Random(seed)
    images: List[Image.Image] = []

    for index in range(max(1, int(frames))):
        frame_prompt = f"{prompt}, frame {index + 1}"
        frame_seed = None if seed is None else seed + index + rng.randint(0, 3)
        image, _ = txt2img(frame_prompt, seed=frame_seed)
        images.append(image)

    gif_b64 = ""
    buffer = io.BytesIO()
    try:
        if imageio is not None and len(images) > 1:
            imageio.mimsave(
                buffer,
                [im.convert("RGBA") for im in images],
                format="GIF",
                duration=max(1, int(duration_ms)) / 1000.0,
            )
        else:
            frames = [im.convert("RGBA") for im in images]
            frames[0].save(
                buffer,
                format="GIF",
                save_all=len(frames) > 1,
                append_images=frames[1:],
                loop=0,
                duration=max(1, int(duration_ms)),
            )
        gif_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
    except Exception:
        gif_b64 = ""

    return images[0], {
        "prompt": prompt,
        "frames": len(images),
        "duration_ms": int(duration_ms),
        "seed": seed,
        "duration_s": time.time() - start,
        "gif_b64": gif_b64,
    }
