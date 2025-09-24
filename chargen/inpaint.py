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
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
import os
import hashlib

import torch
from PIL import Image, ImageOps
from diffusers import StableDiffusionInpaintPipeline

try:
    # Optional import; module exists in this pack
    from tools.cache import Cache
except Exception:  # pragma: no cover
    Cache = None  # type: ignore

DEFAULT_MODEL = os.environ.get("PIXSTU_INPAINT_MODEL", "runwayml/stable-diffusion-inpainting")


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


def _prep_mask(mask: Image.Image, threshold: Optional[int] = None) -> Image.Image:
    """Ensure single-channel mask. White/1 = to be **inpainted** (Diffusers convention).
    If threshold supplied, convert to binary L-mode mask.
    """
    if mask.mode != "L":
        mask = mask.convert("L")
    if threshold is not None:
        # Binarize
        mask = mask.point(lambda p: 255 if p >= threshold else 0)
    return mask


def _hash_inputs(prompt: str, init_path: Path, mask_path: Path, steps: int, guidance: float, model_id: str, seed: Optional[int]) -> str:
    h = hashlib.sha256()
    h.update(prompt.encode("utf-8"))
    for p in (init_path, mask_path):
        h.update(Path(p).read_bytes())
    h.update(f"{steps}|{guidance}|{model_id}|{seed}".encode("utf-8"))
    return h.hexdigest()


def inpaint(prompt: str,
            init_image: Path | str,
            mask_image: Path | str,
            guidance_scale: float = 7.5,
            steps: int = 50,
            threshold: Optional[int] = None,
            seed: Optional[int] = None,
            model_id: str = DEFAULT_MODEL,
            disable_safety_checker: Optional[bool] = None,
            use_cache: bool = True) -> Image.Image:
    """Main inpaint entrypoint.

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
            prompt=prompt,
            image=init,
            mask_image=mask,
            guidance_scale=guidance_scale,
            num_inference_steps=steps,
            generator=generator
        ).images[0]
        return result

    img: Image.Image
    if autocast_ctx is not None:
        try:
            with autocast_ctx(device_type=device.type):
                img = _run()
        except Exception:
            # Fallback to float32 on CPU if autocast path fails
            pipe.to(torch.device("cpu"))
            img = _run()
    else:
        img = _run()

    if use_cache and Cache is not None:
        with Cache(namespace="inpaint") as c:
            c.put_image(cache_key, img)

    return img


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="PixStu Inpainting Entrypoint")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--init", required=True)
    parser.add_argument("--mask", required=True)
    parser.add_argument("--out", default="out.png")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--guidance", type=float, default=7.5)
    parser.add_argument("--threshold", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--disable-safety", action="store_true")
    parser.add_argument("--no-cache", action="store_true")
    args = parser.parse_args()

    img = inpaint(
        prompt=args.prompt,
        init_image=args.init,
        mask_image=args.mask,
        steps=args.steps,
        guidance_scale=args.guidance,
        threshold=args.threshold,
        seed=args.seed,
        disable_safety_checker=args.disable_safety,
        use_cache=not args.no_cache,
    )
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    img.save(args.out)
    print(f"[PixStu] Inpainting complete → {args.out}")
