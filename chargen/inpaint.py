"""Stable Diffusion based inpainting utilities used by the Pin Editor."""

from __future__ import annotations

import logging
import threading
from io import BytesIO
from typing import Optional, Tuple

import torch
from PIL import Image

try:  # pragma: no cover - optional dependency
    from diffusers import StableDiffusionInpaintPipeline
except Exception:  # pragma: no cover - optional dependency fallback
    StableDiffusionInpaintPipeline = None  # type: ignore

from tools.cache import Cache
from tools.cache_keys import inpaint_key

logger = logging.getLogger(__name__)

_PIPELINE_LOCK = threading.Lock()
_PIPELINE: Optional[StableDiffusionInpaintPipeline] = None
_PIPELINE_DEVICE: Optional[str] = None
_CACHE = Cache(namespace="inpaint")


def _detect_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    try:  # AMD via ZLUDA
        import zluda  # type: ignore  # pragma: no cover - optional dependency

        return "cuda"
    except Exception:  # pragma: no cover - optional
        pass
    try:  # Intel via zkluda
        import zkluda  # type: ignore  # pragma: no cover - optional dependency

        return "cuda"
    except Exception:  # pragma: no cover - optional
        pass
    return "cpu"


def _select_dtype(device: str) -> Optional[torch.dtype]:
    if device in {"cuda", "mps"} and hasattr(torch, "float16"):
        return torch.float16
    if hasattr(torch, "bfloat16") and device == "cpu":  # pragma: no cover - depends on build
        return torch.bfloat16
    return torch.float32 if hasattr(torch, "float32") else None


def _ensure_pipeline(model_id: str = "runwayml/stable-diffusion-inpainting") -> Tuple[StableDiffusionInpaintPipeline, str]:
    global _PIPELINE, _PIPELINE_DEVICE

    if StableDiffusionInpaintPipeline is None:  # pragma: no cover - runtime guard
        raise RuntimeError("StableDiffusionInpaintPipeline is unavailable; install diffusers[torch]")

    if _PIPELINE is not None and _PIPELINE_DEVICE is not None:
        return _PIPELINE, _PIPELINE_DEVICE

    with _PIPELINE_LOCK:
        if _PIPELINE is not None and _PIPELINE_DEVICE is not None:
            return _PIPELINE, _PIPELINE_DEVICE

        device = _detect_device()
        dtype = _select_dtype(device)
        load_kwargs = {"torch_dtype": dtype} if dtype is not None else {}
        try:
            pipe = StableDiffusionInpaintPipeline.from_pretrained(model_id, **load_kwargs)
        except Exception as exc:  # pragma: no cover - runtime failure guard
            raise RuntimeError(f"Failed to load inpainting pipeline {model_id}: {exc}") from exc

        if hasattr(pipe, "to"):
            pipe = pipe.to(device)

        if device == "cuda":
            try:
                if hasattr(pipe, "enable_xformers_memory_efficient_attention"):
                    pipe.enable_xformers_memory_efficient_attention()
                if hasattr(torch, "compile") and hasattr(pipe, "unet"):
                    pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead", fullgraph=True)  # type: ignore[attr-defined]
            except Exception as exc:  # pragma: no cover - optional accel
                logger.warning("CUDA acceleration features unavailable: %s", exc)
        else:
            try:
                if hasattr(pipe, "enable_vae_slicing"):
                    pipe.enable_vae_slicing()
                if hasattr(pipe, "enable_vae_tiling"):
                    pipe.enable_vae_tiling()
            except Exception as exc:  # pragma: no cover - optional accel
                logger.warning("VAE optimisations unavailable: %s", exc)

        _PIPELINE = pipe
        _PIPELINE_DEVICE = device
        return _PIPELINE, _PIPELINE_DEVICE


def _prepare_mask(mask: Image.Image, size: Tuple[int, int]) -> Image.Image:
    if mask is None:
        raise ValueError("mask must not be None")
    processed = mask.convert("L")
    if processed.size != size:
        processed = processed.resize(size, Image.NEAREST)
    # Ensure binary mask with white=edit, black=preserve
    return processed.point(lambda px: 255 if px >= 128 else 0)


def _prepare_image(image: Image.Image, size: Optional[Tuple[int, int]] = None) -> Image.Image:
    processed = image.convert("RGB")
    if size is not None and processed.size != size:
        processed = processed.resize(size, Image.BICUBIC)
    return processed


def inpaint_region(
    base_img: Image.Image,
    mask: Image.Image,
    prompt: str = "",
    ref_img: Optional[Image.Image] = None,
    *,
    guidance_scale: float = 7.5,
    steps: int = 50,
) -> Image.Image:
    """Run Stable Diffusion inpainting over ``base_img`` using ``mask``.

    Args:
        base_img: The source image that should be modified.
        mask: Mask image where white pixels denote regions that may change.
        prompt: Optional text prompt that guides the inpainting result.
        ref_img: Optional reference image; if provided it is resized to the
            base image and used as the initial image for inpainting.
        guidance_scale: Classifier-free guidance used by the pipeline.
        steps: Number of diffusion steps to execute.

    Returns:
        A PIL image containing the inpainted result.
    """

    if base_img is None:
        raise ValueError("base_img must not be None")

    pipe, _ = _ensure_pipeline()

    base_processed = _prepare_image(base_img)
    mask_image = _prepare_mask(mask, base_processed.size)
    ref_processed = _prepare_image(ref_img, base_processed.size) if ref_img is not None else None
    source = ref_processed or base_processed

    cache_key = inpaint_key(
        base_processed,
        mask_image,
        prompt or "",
        guidance_scale=float(guidance_scale),
        steps=int(steps),
        ref_img=ref_processed,
    )

    cached = _CACHE.get(cache_key)
    if cached:
        with Image.open(BytesIO(cached)) as handle:
            return handle.copy()

    try:
        result = pipe(
            prompt=prompt or "",
            image=source,
            mask_image=mask_image,
            guidance_scale=float(guidance_scale),
            num_inference_steps=int(steps),
        )
    except Exception as exc:  # pragma: no cover - runtime failure guard
        raise RuntimeError(f"Inpainting failed: {exc}") from exc

    images = getattr(result, "images", None)
    if not images:
        raise RuntimeError("Inpainting pipeline returned no images")
    final = images[0]

    try:
        buffer = BytesIO()
        final.save(buffer, format="PNG")
        _CACHE.set(cache_key, buffer.getvalue())
    except Exception as exc:  # pragma: no cover - best-effort cache
        logger.debug("Failed to persist inpaint cache entry: %s", exc)

    return final
