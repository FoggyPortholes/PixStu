"""
Image-to-Image pipeline (no mask).
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from PIL import Image, ImageDraw

from ..tools.cache import Cache
from ..tools.device import pick_device
from ..tools.guardrails import check_blank_background, check_prompt
from ..tools.util import set_seed
from .lora import prepare_lora_kwargs

try:  # optional heavy deps
    import torch
    from diffusers import StableDiffusionXLImg2ImgPipeline
except Exception:  # pragma: no cover - exercised when deps missing
    torch = None  # type: ignore[assignment]
    StableDiffusionXLImg2ImgPipeline = None  # type: ignore[assignment]

ImageLike = Union[str, Path, Image.Image]


def _read(image: ImageLike) -> Image.Image:
    if isinstance(image, Image.Image):
        return image.convert("RGBA")
    return Image.open(image).convert("RGBA")


def _cache_key(**payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, default=str)


def _image_stamp(src: ImageLike) -> Optional[str]:
    if isinstance(src, (str, Path)):
        path = Path(src)
        try:
            return f"{path.resolve()}@{path.stat().st_mtime_ns}"
        except Exception:
            return str(path.resolve())
    return None


def _fallback(base: Image.Image) -> Image.Image:
    out = base.copy()
    ImageDraw.Draw(out).rectangle([(10, 10), (min(base.width, 100), min(base.height, 100))], outline=(0, 255, 255, 255), width=3)
    return out


def img2img(
    prompt: str,
    init_image: ImageLike,
    strength: float = 0.65,
    steps: int = 40,
    guidance_scale: float = 7.5,
    seed: Optional[int] = None,
    negative: str = "",
    loras: Optional[List[Tuple[str, float]]] = None,
    dtype: Optional[str] = None,
    disable_safety_checker: bool = False,
) -> Tuple[Image.Image, Dict[str, Any]]:
    start = time.time()
    check_prompt(prompt)
    init = _read(init_image)
    set_seed(seed)

    stamp = _image_stamp(init_image)
    key = _cache_key(
        prompt=prompt,
        negative=negative,
        strength=float(strength),
        steps=int(steps),
        guidance=float(guidance_scale),
        seed=seed,
        loras=loras or [],
        dtype=dtype,
        init=stamp,
    )

    cache = None
    if stamp is not None:
        cache = Cache("img2img", ttl=int(12 * 3600))
        cached = cache.get_image(key)
        if cached is not None:
            check_blank_background(cached)
            return cached, {
                "prompt": prompt,
                "device": "cached",
                "seed": seed,
                "duration_s": 0.0,
            }

    if torch is None or StableDiffusionXLImg2ImgPipeline is None:
        image = _fallback(init)
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

    pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
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
        image=init.convert("RGB"),
        strength=float(strength),
        num_inference_steps=int(steps),
        guidance_scale=float(guidance_scale),
        negative_prompt=negative or None,
        generator=generator,
        **lora_kwargs,
    )

    image = result.images[0].convert("RGBA")
    check_blank_background(image)

    if cache is not None:
        cache.put_image(key, image)

    return image, {
        "prompt": prompt,
        "device": str(device),
        "seed": seed,
        "duration_s": time.time() - start,
    }
