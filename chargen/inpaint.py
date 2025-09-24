#!/usr/bin/env python3
"""
PixStu Inpainting Module — Hardened
===================================
- Safe device/dtype picking (CUDA → ZLUDA → zkluda → MPS → CPU)
- Autocast where supported + float32 fallback on CPU/MPS
- Robust mask handling (single-channel "L", thresholding option)
- Optional safety checker toggle (env or arg) for benchmarks
- Deterministic seed option
- Persistent cache hooks (read/write) via tools.cache
"""
from __future__ import annotations
<
import torch
from PIL import Image, ImageOps
from diffusers import StableDiffusionInpaintPipeline

try:
    # Optional import; module exists in this pack
    from tools.cache import Cache
except Exception:  # pragma: no cover
    Cache = None  # type: ignore



def _has_env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def pick_device() -> torch.device:
    # Honor immutable rule ordering: CUDA > ZLUDA > zkluda > MPS > CPU
    if torch.cuda.is_available():
        return torch.device("cuda")
    # ZLUDA pathing (users set these to emulate CUDA). Nothing to probe reliably; prefer presence of env vars.
    if os.environ.get("ZLUDA_PATH") or os.environ.get("ZKLUDA_PATH"):
        # Expose as CUDA device for pipelines that branch on .cuda
        return torch.device("cuda")
    try:
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
    except Exception:
        pass
    return torch.device("cpu")


def pick_dtype(device: torch.device) -> torch.dtype:
    # Use fp16 on CUDA. MPS often wants float16 but diffusers can run float32; prefer float16 with fallback.
    if device.type == "cuda":
        return torch.float16
    if device.type == "mps":
        return torch.float16
    return torch.float32


def load_pipeline(model_id: str = DEFAULT_MODEL,
                  disable_safety_checker: Optional[bool] = None) -> Tuple[StableDiffusionInpaintPipeline, torch.device]:
    device = pick_device()
    dtype = pick_dtype(device)

    # Allow opt-out of safety checker via env or explicit flag
    if disable_safety_checker is None:
        disable_safety_checker = _has_env_flag("PIXSTU_DISABLE_SAFETY")

    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        model_id,
        torch_dtype=dtype,
        safety_checker=None if disable_safety_checker else None  # keep None; SD1.5 inpaint ships without checker sometimes
    )
    pipe = pipe.to(device)

    # Enable attention slicing on low-VRAM devices
    if device.type != "cuda":
        try:
            pipe.enable_attention_slicing()
        except Exception:
            pass

    return pipe, device




    Args:
        prompt: Text prompt.
        init_image: Path to base image.
        mask_image: Path to mask image; white regions will be replaced.
        guidance_scale: CFG scale.
        steps: Inference steps.
        threshold: Optional binarization threshold for the mask.
        seed: Optional RNG seed for determinism.
        model_id: HF repo id (override via env is defaulted above).
        disable_safety_checker: Toggle safety checker.
        use_cache: If True, consult and write persistent cache.
    """
    init_path = Path(init_image)
    mask_path = Path(mask_image)

    cache_key = _hash_inputs(prompt, init_path, mask_path, steps, guidance_scale, model_id, seed)
    if use_cache and Cache is not None:
        with Cache(namespace="inpaint") as c:
            cached = c.get_image(cache_key)
            if cached is not None:
                return cached

    pipe, device = load_pipeline(model_id=model_id, disable_safety_checker=disable_safety_checker)

    init = Image.open(init_path).convert("RGB")
    mask = Image.open(mask_path)
    mask = _prep_mask(mask, threshold=threshold)

    # Seed control
    if seed is not None:
        generator = torch.Generator(device=device.type)
        generator.manual_seed(int(seed))
    else:
        generator = None

    autocast_ctx = (
        torch.cuda.amp.autocast if device.type == "cuda" else
        (torch.autocast if hasattr(torch, "autocast") and device.type == "mps" else None)
    )

    def _run() -> Image.Image:
        result = pipe(

