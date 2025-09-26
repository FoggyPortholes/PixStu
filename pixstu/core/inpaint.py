"""
Inpainting pipeline with guardrails + fallback.
"""
from __future__ import annotations

import importlib
import json
import random
import time
from pathlib import Path
from typing import Iterable, Optional, Tuple

from PIL import Image, ImageDraw

from ..tools.cache import Cache
from ..tools.device import pick_device
from ..tools.guardrails import check_blank_background, check_prompt
from .lora import prepare_lora_kwargs

_torch = None
_pipeline_cls = None


def _load_torch():
    global _torch
    if _torch is not None:
        return _torch
    spec = importlib.util.find_spec("torch")
    if spec is None:
        return None
    _torch = importlib.import_module("torch")
    return _torch


def _load_pipeline():
    global _pipeline_cls
    if _pipeline_cls is not None:
        return _pipeline_cls
    spec = importlib.util.find_spec("diffusers")
    if spec is None:
        return None
    module = importlib.import_module("diffusers")
    cls = getattr(module, "StableDiffusionXLInpaintPipeline", None)
    if cls is None:
        try:
            pipeline_module = importlib.import_module(
                "diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl_inpaint"
            )
            cls = getattr(pipeline_module, "StableDiffusionXLInpaintPipeline", None)
        except Exception:
            cls = None
    _pipeline_cls = cls
    return cls


def _read(path: str | Path) -> Image.Image:
    return Image.open(path).convert("RGBA")


def _seed(seed: Optional[int]) -> None:
    if seed is None:
        return
    torch = _load_torch()
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except Exception:
        pass
    if torch is not None:
        torch.manual_seed(seed)


def _cache_key(prompt: str, init_image: str | Path, mask_image: str | Path, **extra) -> str:
    payload = {
        "prompt": prompt,
        "init": Path(init_image).stat().st_mtime_ns,
        "mask": Path(mask_image).stat().st_mtime_ns,
    }
    payload.update(extra)
    return json.dumps(payload, sort_keys=True)


def _fallback(init: Image.Image) -> Image.Image:
    out = init.copy()
    ImageDraw.Draw(out).rectangle([10, 10, 60, 60], outline=(0, 255, 255))
    return out


def inpaint(
    prompt: str,
    init_image: str | Path,
    mask_image: str | Path,
    steps: int = 40,
    guidance_scale: float = 7.5,
    seed: Optional[int] = None,
    disable_safety_checker: bool = False,
    loras: Optional[Iterable[Tuple[str, float]]] = None,
    dtype: Optional[str] = None,
):
    start = time.time()
    check_prompt(prompt)
    init = _read(init_image)
    mask = _read(mask_image)
    _seed(seed)

    key = _cache_key(
        prompt,
        init_image,
        mask_image,
        steps=steps,
        guidance=guidance_scale,
        seed=seed,
    )

    cache = Cache("inpaint", ttl=int(12 * 3600))
    cached = cache.get_image(key)
    if cached is not None:
        check_blank_background(cached)
        return cached, {"device": "cached", "elapsed": 0.0}

    torch = _load_torch()
    pipeline_cls = _load_pipeline()

    if torch is None or pipeline_cls is None:
        fallback = _fallback(init)
        check_blank_background(fallback)
        return fallback, {"device": "stub", "elapsed": time.time() - start}

    device = pick_device()
    torch_dtype = torch.float16 if (dtype == "float16" or (dtype is None and device.type == "cuda")) else torch.float32

    pipe = pipeline_cls.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch_dtype,
    ).to(device)

    if disable_safety_checker:
        pipe.safety_checker = None

    generator = torch.Generator(device=device) if seed is not None else None
    if generator is not None:
        generator = generator.manual_seed(seed)

    lora_kwargs = prepare_lora_kwargs(loras or []) if loras else {}

    result = pipe(
        prompt=prompt,
        image=init.convert("RGB"),
        mask_image=mask.convert("RGB"),
        num_inference_steps=steps,
        guidance_scale=guidance_scale,
        generator=generator,
        **lora_kwargs,
    )
    image = result.images[0].convert("RGBA")
    check_blank_background(image)

    cache.put_image(key, image)

    return image, {"device": str(device), "elapsed": time.time() - start}
