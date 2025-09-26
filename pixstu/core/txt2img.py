"""
Text-to-Image pipeline with graceful fallback.
"""
from __future__ import annotations

import json
import time
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image, ImageDraw

from ..tools.cache import Cache
from ..tools.device import pick_device
from ..tools.guardrails import check_blank_background, check_prompt
from ..tools.util import set_seed
from .lora import prepare_lora_kwargs

try:  # optional heavy deps
    import torch
    from diffusers import StableDiffusionXLPipeline
except Exception:  # pragma: no cover - exercised when deps missing
    torch = None  # type: ignore[assignment]
    StableDiffusionXLPipeline = None  # type: ignore[assignment]


def _cache_key(**payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, default=str)


def _fallback(width: int, height: int) -> Image.Image:
    image = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    margin = 10
    draw.rectangle(
        [(margin, margin), (width - margin, height - margin)],
        outline=(255, 0, 255, 255),
        width=3,
    )
    return image


def txt2img(
    prompt: str,
    negative: str = "",
    steps: int = 40,
    guidance_scale: float = 7.5,
    width: int = 768,
    height: int = 768,
    seed: Optional[int] = None,
    loras: Optional[List[Tuple[str, float]]] = None,
    dtype: Optional[str] = None,
    disable_safety_checker: bool = False,
) -> Tuple[Image.Image, Dict[str, Any]]:
    start = time.time()
    check_prompt(prompt)
    set_seed(seed)

    key = _cache_key(
        prompt=prompt,
        negative=negative,
        steps=int(steps),
        guidance=float(guidance_scale),
        width=int(width),
        height=int(height),
        seed=seed,
        loras=loras or [],
        dtype=dtype,
    )

    cache = Cache("txt2img", ttl=int(12 * 3600))
    cached = cache.get_image(key)
    if cached is not None:
        check_blank_background(cached)
        return cached, {
            "prompt": prompt,
            "device": "cached",
            "seed": seed,
            "duration_s": 0.0,
        }

    if torch is None or StableDiffusionXLPipeline is None:
        image = _fallback(int(width), int(height))
        check_blank_background(image)
        return image, {
            "prompt": prompt,
            "device": "stub",
            "seed": seed,
            "duration_s": time.time() - start,
        }

    device = pick_device()
    use_fp16 = dtype == "float16" or (dtype is None and getattr(device, "type", "") == "cuda")
    torch_dtype = torch.float16 if use_fp16 else torch.float32  # type: ignore[attr-defined]

    pipe = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch_dtype,
    ).to(device)

    if disable_safety_checker:
        pipe.safety_checker = None

    generator = None
    if seed is not None:
        generator = torch.Generator(device=device).manual_seed(seed)

    lora_kwargs = prepare_lora_kwargs(loras or []) if loras else {}

    result = pipe(
        prompt=prompt,
        negative_prompt=negative or None,
        num_inference_steps=int(steps),
        guidance_scale=float(guidance_scale),
        width=int(width),
        height=int(height),
        generator=generator,
        **lora_kwargs,
    )

    image = result.images[0].convert("RGBA")
    check_blank_background(image)

    cache.put_image(key, image)

    return image, {
        "prompt": prompt,
        "device": str(device),
        "seed": seed,
        "duration_s": time.time() - start,
    }
