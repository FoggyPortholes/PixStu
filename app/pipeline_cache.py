"""Shared Stable Diffusion pipeline cache utilities."""
from __future__ import annotations

import contextlib
import os
import sys
from typing import Dict, List, Optional, Sequence, Tuple

import torch

try:  # pragma: no cover - optional imports depend on diffusers version
    from diffusers import (
        DiffusionPipeline,
        DPMSolverMultistepScheduler,
        LCMScheduler,
        StableDiffusionXLImg2ImgPipeline,
        StableDiffusionXLPipeline,
    )
except Exception:  # pragma: no cover - diffusers not installed for some tests
    DiffusionPipeline = None  # type: ignore
    DPMSolverMultistepScheduler = None  # type: ignore
    LCMScheduler = None  # type: ignore
    StableDiffusionXLImg2ImgPipeline = None  # type: ignore
    StableDiffusionXLPipeline = None  # type: ignore


ROOT = os.path.abspath(os.path.dirname(__file__))
PROJ = os.path.abspath(os.path.join(ROOT, ".."))
EXE_DIR = os.path.dirname(sys.executable) if getattr(sys, "frozen", False) else PROJ
MODELS_ROOT = os.getenv("PCS_MODELS_ROOT", os.path.join(EXE_DIR, "models"))

DEFAULT_MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"


def _device() -> Tuple[str, torch.dtype, str]:
    if torch.cuda.is_available():
        return "cuda", torch.float16, "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps", torch.float16, "mps"
    return "cpu", torch.float32, "cpu"


DEV_KIND, DTYPE, DEVICE = _device()

_TXT2IMG_CACHE: Dict[str, DiffusionPipeline] = {}
_IMG2IMG_CACHE: Dict[str, DiffusionPipeline] = {}
_PIPE_ACTIVE_ADAPTERS: Dict[int, List[str]] = {}


def resolve_under_models(path: Optional[str]) -> Optional[str]:
    """Return ``path`` resolved under ``MODELS_ROOT`` if applicable."""

    if not path:
        return path
    expanded = os.path.expanduser(str(path))
    if os.path.isabs(expanded):
        return os.path.normpath(expanded)
    normalized = os.path.normpath(expanded)
    project_candidate = os.path.abspath(os.path.join(PROJ, normalized))
    if os.path.exists(project_candidate):
        return os.path.normpath(project_candidate)
    parts = normalized.split(os.sep)
    if parts and parts[0] == "models":
        normalized = os.path.join(*parts[1:]) if len(parts) > 1 else ""
    return os.path.normpath(os.path.abspath(os.path.join(MODELS_ROOT, normalized)))


def _resolve_model_id(model_id: Optional[str], local_dir: Optional[str]) -> str:
    mid = model_id or DEFAULT_MODEL_ID
    if local_dir:
        candidate = resolve_under_models(local_dir)
        if candidate and os.path.isdir(candidate):
            mid = candidate
    resolved = resolve_under_models(mid)
    if resolved and os.path.isdir(resolved):
        mid = resolved
    return mid


def _load_pipeline(loader, model_id: str) -> DiffusionPipeline:
    if DiffusionPipeline is None:
        raise RuntimeError("diffusers is required to load Stable Diffusion pipelines")

    pipe = loader.from_pretrained(  # type: ignore[attr-defined]
        model_id,
        use_safetensors=True,
        torch_dtype=(DTYPE if DEV_KIND != "dml" else torch.float32),
        variant=("fp16" if DTYPE == torch.float16 else None),
    )
    pipe.to(DEVICE)
    with contextlib.suppress(Exception):
        pipe.enable_vae_tiling()
    if DEV_KIND == "cuda":
        with contextlib.suppress(Exception):
            pipe.enable_xformers_memory_efficient_attention()
    return pipe


def _apply_scheduler(pipe: DiffusionPipeline, quality: str) -> DiffusionPipeline:
    if LCMScheduler and quality == "Fast (LCM)":
        pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
    elif DPMSolverMultistepScheduler:
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    return pipe


def get_txt2img(
    model_id: Optional[str],
    *,
    quality: str,
    local_dir: Optional[str] = None,
) -> DiffusionPipeline:
    """Return a cached text-to-image pipeline configured for ``quality``."""

    resolved = _resolve_model_id(model_id, local_dir)
    pipe = _TXT2IMG_CACHE.get(resolved)
    if pipe is None:
        loader = StableDiffusionXLPipeline or DiffusionPipeline
        pipe = _load_pipeline(loader, resolved)
        _TXT2IMG_CACHE[resolved] = pipe
    return _apply_scheduler(pipe, quality)


def get_img2img(
    model_id: Optional[str],
    *,
    quality: str,
    local_dir: Optional[str] = None,
) -> DiffusionPipeline:
    """Return a cached image-to-image pipeline configured for ``quality``."""

    resolved = _resolve_model_id(model_id, local_dir)
    pipe = _IMG2IMG_CACHE.get(resolved)
    if pipe is None:
        loader = StableDiffusionXLImg2ImgPipeline or DiffusionPipeline
        pipe = _load_pipeline(loader, resolved)
        _IMG2IMG_CACHE[resolved] = pipe
    return _apply_scheduler(pipe, quality)


def apply_loras(
    pipe: DiffusionPipeline,
    *,
    quality_mode: str,
    lcm_dir: Optional[str] = None,
    loras: Optional[Sequence[str]] = None,
    lora_weights: Optional[Dict[str, float]] = None,
) -> List[str]:
    """Load LoRA adapters for ``pipe`` and return the active adapter names."""

    lora_weights = lora_weights or {}
    adapters: List[str] = []
    weights: List[float] = []
    if quality_mode == "Fast (LCM)" and lcm_dir and os.path.isdir(lcm_dir):
        try:
            pipe.load_lora_weights(lcm_dir, adapter_name="lcm")
        except Exception as exc:  # pragma: no cover - runtime guard
            print("[WARN] LCM:", exc)
        else:
            adapters.append("lcm")
            weights.append(1.0)

    for path in loras or []:
        try:
            adapter_name = os.path.splitext(os.path.basename(path))[0]
            try:
                pipe.load_lora_into_unet(path, adapter_name=adapter_name)
            except Exception:
                pipe.load_lora_weights(path, adapter_name=adapter_name)
            adapters.append(adapter_name)
            weights.append(float(lora_weights.get(path, 1.0)))
        except Exception as exc:  # pragma: no cover - runtime guard
            print("[WARN] LoRA:", path, exc)

    key = id(pipe)
    if adapters:
        try:
            pipe.set_adapters(adapters, adapter_weights=weights)
        except Exception as exc:  # pragma: no cover - runtime guard
            print("[WARN] set_adapters:", exc)
    else:
        previous = _PIPE_ACTIVE_ADAPTERS.get(key) or []
        if previous:
            try:
                pipe.set_adapters([])
            except Exception as exc:  # pragma: no cover - runtime guard
                print("[WARN] clear adapters:", exc)
            try:
                pipe.unload_lora_weights()
            except Exception as exc:  # pragma: no cover - runtime guard
                print("[WARN] unload adapters:", exc)

    _PIPE_ACTIVE_ADAPTERS[key] = adapters[:]
    return adapters[:]


def get_active_adapters(pipe: DiffusionPipeline) -> List[str]:
    """Return adapter names currently configured on ``pipe``."""

    return _PIPE_ACTIVE_ADAPTERS.get(id(pipe), [])[:]


__all__ = [
    "DEFAULT_MODEL_ID",
    "DEV_KIND",
    "DTYPE",
    "DEVICE",
    "MODELS_ROOT",
    "PROJ",
    "ROOT",
    "get_img2img",
    "get_txt2img",
    "apply_loras",
    "get_active_adapters",
    "resolve_under_models",
]
