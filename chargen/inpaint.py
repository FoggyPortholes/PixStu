#!/usr/bin/env python3
"""Hardened inpainting helpers used by PixStu."""

from __future__ import annotations

import hashlib
import os
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
from PIL import Image, ImageOps
from diffusers import StableDiffusionInpaintPipeline

from tools.device import pick_device

try:  # pragma: no cover - optional dependency in some builds
    from tools.cache import Cache
except Exception:  # pragma: no cover - cache support is optional
    Cache = None  # type: ignore

DEFAULT_MODEL = os.environ.get(
    "PIXSTU_INPAINT_MODEL", "runwayml/stable-diffusion-inpainting"
)
DEFAULT_GUIDANCE_SCALE = 7.5
DEFAULT_STEPS = 50

_PIPELINE: Optional[StableDiffusionInpaintPipeline] = None
_PIPELINE_DEVICE: Optional[torch.device] = None
_PIPELINE_MODEL_ID: Optional[str] = None


def _has_env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _hash_inputs(*items: object) -> str:
    digest = hashlib.sha256()

    def _update(value: object) -> None:
        if isinstance(value, Image.Image):
            digest.update(value.mode.encode("utf-8"))
            digest.update(str(value.size).encode("utf-8"))
            digest.update(value.tobytes())
        elif isinstance(value, Path):
            digest.update(value.as_posix().encode("utf-8"))
            if value.exists():
                digest.update(str(value.stat().st_mtime_ns).encode("utf-8"))
        elif isinstance(value, (bytes, bytearray)):
            digest.update(value)
        else:
            digest.update(str(value).encode("utf-8"))

    for item in items:
        _update(item)
    return digest.hexdigest()


def _prep_mask(mask: Image.Image, threshold: Optional[float] = None) -> Image.Image:
    if mask.mode not in {"L", "1"}:
        mask = ImageOps.grayscale(mask)
    if threshold is not None:
        limit = float(threshold)

        def _threshold(pixel: int) -> int:
            return 255 if pixel >= limit else 0

        mask = mask.point(_threshold)
    else:
        mask = mask.convert("L")
    return mask
def pick_dtype(device: torch.device) -> torch.dtype:
    if device.type in {"cuda", "mps"}:
        return torch.float16
    return torch.float32


def load_pipeline(
    model_id: str = DEFAULT_MODEL,
    disable_safety_checker: Optional[bool] = None,
) -> Tuple[StableDiffusionInpaintPipeline, torch.device]:
    device = pick_device()
    dtype = pick_dtype(device)

    if disable_safety_checker is None:
        disable_safety_checker = _has_env_flag("PIXSTU_DISABLE_SAFETY")

    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        model_id,
        torch_dtype=dtype,
        safety_checker=None if disable_safety_checker else None,
    )
    pipe = pipe.to(device)

    if device.type != "cuda":
        try:
            pipe.enable_attention_slicing()
        except Exception:  # pragma: no cover - not all backends support it
            pass

    return pipe, device


def _ensure_pipeline(
    model_id: Optional[str], disable_safety_checker: Optional[bool]
) -> Tuple[StableDiffusionInpaintPipeline, torch.device]:
    global _PIPELINE, _PIPELINE_DEVICE, _PIPELINE_MODEL_ID

    requested_model = model_id or DEFAULT_MODEL
    if (
        _PIPELINE is None
        or _PIPELINE_DEVICE is None
        or _PIPELINE_MODEL_ID != requested_model
    ):
        _PIPELINE, _PIPELINE_DEVICE = load_pipeline(
            requested_model, disable_safety_checker=disable_safety_checker
        )
        _PIPELINE_MODEL_ID = requested_model

    return _PIPELINE, _PIPELINE_DEVICE


def _as_image(image: Image.Image | str | Path, mode: str = "RGB") -> Image.Image:
    if isinstance(image, Image.Image):
        return image.convert(mode)
    path = Path(image)
    with Image.open(path) as handle:
        return handle.convert(mode)


def _maybe_cache_get(key: str) -> Optional[Image.Image]:
    if Cache is None:
        return None
    with Cache(namespace="inpaint") as cache:  # pragma: no cover - optional path
        return cache.get_image(key)


def _maybe_cache_put(key: str, image: Image.Image) -> None:
    if Cache is None:
        return
    with Cache(namespace="inpaint") as cache:  # pragma: no cover - optional path
        cache.put_image(key, image)


def inpaint_region(
    init_image: Image.Image | str | Path,
    mask_image: Image.Image | str | Path,
    *,
    prompt: str,
    guidance_scale: float = DEFAULT_GUIDANCE_SCALE,
    steps: int = DEFAULT_STEPS,
    threshold: Optional[float] = None,
    seed: Optional[int] = None,
    model_id: Optional[str] = None,
    disable_safety_checker: Optional[bool] = None,
    use_cache: bool = True,
) -> Image.Image:
    init = _as_image(init_image, "RGB")
    mask = _as_image(mask_image, "L")
    mask = mask.resize(init.size)
    mask = _prep_mask(mask, threshold=threshold)

    cache_key = _hash_inputs(
        prompt,
        guidance_scale,
        steps,
        model_id or DEFAULT_MODEL,
        seed if seed is not None else "",
        init,
        mask,
    )

    if use_cache:
        cached = _maybe_cache_get(cache_key)
        if cached is not None:
            return cached

    pipe, device = _ensure_pipeline(model_id, disable_safety_checker)

    if seed is not None:
        generator = torch.Generator(device=device)
        generator.manual_seed(int(seed))
    else:
        generator = None

    autocast = (
        torch.cuda.amp.autocast
        if device.type == "cuda"
        else torch.autocast if device.type == "mps" and hasattr(torch, "autocast") else None
    )

    def _run() -> Image.Image:
        result = pipe(
            prompt=prompt,
            image=init,
            mask_image=mask,
            guidance_scale=float(guidance_scale),
            num_inference_steps=int(steps),
            generator=generator,
        )
        output = result.images[0]
        if not isinstance(output, Image.Image):
            output = Image.fromarray(output)
        return output

    context = autocast() if autocast is not None else nullcontext()
    with context:
        generated = _run()

    if use_cache:
        _maybe_cache_put(cache_key, generated)

    return generated


def inpaint(
    *,
    prompt: str,
    init_image: Image.Image | str | Path,
    mask_image: Image.Image | str | Path,
    guidance_scale: float = DEFAULT_GUIDANCE_SCALE,
    steps: int = DEFAULT_STEPS,
    threshold: Optional[float] = None,
    seed: Optional[int] = None,
    model_id: Optional[str] = None,
    disable_safety_checker: Optional[bool] = None,
    use_cache: bool = True,
) -> Tuple[Image.Image, Dict[str, object]]:
    """Run an inpaint operation and return the resulting image and metadata."""

    image = inpaint_region(
        init_image=init_image,
        mask_image=mask_image,
        prompt=prompt,
        guidance_scale=guidance_scale,
        steps=steps,
        threshold=threshold,
        seed=seed,
        model_id=model_id,
        disable_safety_checker=disable_safety_checker,
        use_cache=use_cache,
    )

    metadata: Dict[str, object] = {
        "prompt": prompt,
        "guidance_scale": float(guidance_scale),
        "steps": int(steps),
        "model_id": model_id or DEFAULT_MODEL,
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }

    if seed is not None:
        metadata["seed"] = int(seed)
    if threshold is not None:
        metadata["threshold"] = float(threshold)
    if disable_safety_checker is not None:
        metadata["disable_safety_checker"] = bool(disable_safety_checker)
    if isinstance(init_image, (str, Path)):
        metadata["init_image"] = str(init_image)
    if isinstance(mask_image, (str, Path)):
        metadata["mask_image"] = str(mask_image)

    return image, metadata


__all__ = [
    "DEFAULT_GUIDANCE_SCALE",
    "DEFAULT_MODEL",
    "DEFAULT_STEPS",
    "inpaint",
    "inpaint_region",
    "load_pipeline",
    "pick_device",
    "pick_dtype",
]

