import contextlib
import io
import logging
import os
from functools import lru_cache
import warnings

logger = logging.getLogger(__name__)

_NOISY_XFORMERS_MESSAGES = (
    "xFormers can't load C++/CUDA extensions",
    "Memory-efficient attention, SwiGLU, sparse and more won't be available.",
)


class _SilenceXformersWarnings(logging.Filter):
    """Filter out noisy xFormers warnings about optional CUDA features."""

    def filter(self, record: logging.LogRecord) -> bool:  # pragma: no cover - log side-effect
        message = record.getMessage()
        return not any(noise in message for noise in _NOISY_XFORMERS_MESSAGES)


@lru_cache(maxsize=None)
def _install_xformers_noise_filter() -> None:
    """Mute noisy xFormers logs when CUDA acceleration is unavailable."""

    filter_ = _SilenceXformersWarnings()
    for name in (
        "xformers",
        "xformers.ops",
        "xformers.components",
        "diffusers.utils.import_utils",
    ):
        logging.getLogger(name).addFilter(filter_)


import torch
from PIL import Image

from tools.device import pick_device


if not torch.cuda.is_available():  # pragma: no cover - depends on hardware
    _install_xformers_noise_filter()


try:
    from diffusers import StableDiffusionXLPipeline, StableDiffusionXLControlNetPipeline
except Exception:  # pragma: no cover - optional dependency fallback
    StableDiffusionXLPipeline = StableDiffusionXLControlNetPipeline = None  # type: ignore
def _enable_xformers_if_safe(pipe: object) -> None:
    """Attempt to enable xFormers attention while silencing noisy warnings."""

    disable_flag = os.environ.get("PCS_ENABLE_XFORMERS", "").strip().lower()
    if disable_flag in {"0", "false", "no", "off"}:
        return
    if disable_flag == "" and not torch.cuda.is_available():
        # Default to skipping on non-CUDA setups unless explicitly requested.
        return
    if not hasattr(pipe, "enable_xformers_memory_efficient_attention"):
        return

    # Silence the extremely loud stderr/stdout noise emitted by incompatible
    # xFormers builds. We still surface actionable failures through logging.
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r".*xFormers can't load C\+\+/CUDA extensions.*",
            category=UserWarning,
        )
        buffer = io.StringIO()
        try:
            with contextlib.redirect_stdout(buffer), contextlib.redirect_stderr(buffer):
                pipe.enable_xformers_memory_efficient_attention()
        except Exception as exc:  # pragma: no cover - optional acceleration
            logger.warning("CUDA speed opts not applied: %s", exc)


class BulletProofGenerator:
    """Minimal generator facade that enforces positives/negatives and device fallbacks."""

    def __init__(self, preset: dict):
        self.preset = preset or {}
        self.device = pick_device()
        if self.device.type != "cuda":
            _install_xformers_noise_filter()
        model_id = self.preset.get("model", "stabilityai/stable-diffusion-xl-base-1.0")
        dtype = getattr(torch, "float16", None) if self.device.type == "cuda" else getattr(torch, "float32", None)
        pipeline_cls = StableDiffusionXLPipeline
        if self.preset.get("controlnets") and StableDiffusionXLControlNetPipeline is not None:
            pipeline_cls = StableDiffusionXLControlNetPipeline
        if pipeline_cls is None:
            raise RuntimeError("Diffusers pipelines are unavailable; install diffusers[torch]")
        try:
            kwargs = {"torch_dtype": dtype} if dtype is not None else {}
            if hasattr(pipeline_cls, "from_pretrained"):
                self.pipe = pipeline_cls.from_pretrained(model_id, **kwargs)
            else:  # lightweight test doubles
                self.pipe = pipeline_cls(model_id, **kwargs)
        except Exception as exc:  # pragma: no cover - runtime failure
            raise RuntimeError(f"Failed to load pipeline {model_id}: {exc}")
        if hasattr(self.pipe, "to"):
            self.pipe = self.pipe.to(self.device)

        # Speed options
        if self.device.type == "cuda":
            _enable_xformers_if_safe(self.pipe)
            try:
                if hasattr(torch, "compile") and hasattr(self.pipe, "unet"):
                    self.pipe.unet = torch.compile(self.pipe.unet, mode="reduce-overhead", fullgraph=True)
            except Exception as exc:  # pragma: no cover
                logger.warning("CUDA speed opts not applied: %s", exc)
        else:
            try:
                if hasattr(self.pipe, "enable_vae_slicing"):
                    self.pipe.enable_vae_slicing()
                if hasattr(self.pipe, "enable_vae_tiling"):
                    self.pipe.enable_vae_tiling()
            except Exception as exc:  # pragma: no cover
                logger.warning("VAE opts not applied: %s", exc)

        # LoRA adapters
        for entry in self.preset.get("loras", []):
            path = entry.get("resolved_path") or entry.get("path")
            weight = float(entry.get("weight", 1.0))
            if not path or not hasattr(self.pipe, "load_lora_weights"):
                continue
            try:
                self.pipe.load_lora_weights(path, weight=weight)
            except Exception as exc:  # pragma: no cover
                logger.warning("LoRA load failed (%s): %s", path, exc)

    def generate(self, prompt: str, seed: int = 42) -> Image.Image:
        generator = torch.Generator(device=self.device).manual_seed(int(seed))
        positive_terms = list(self.preset.get("positive", []))
        negative_terms = set(self.preset.get("negative", []))
        negative_terms.update({"duplicate", "text", "caption", "speech bubble", "watermark", "logo"})
        steps = int(self.preset.get("steps", 30))
        cfg = float(self.preset.get("cfg", 7.5))
        resolution = int(self.preset.get("resolution", 768))
        composed_prompt = f"{prompt}, " + ", ".join(positive_terms) if positive_terms else prompt
        try:
            result = self.pipe(
                prompt=composed_prompt,
                negative_prompt=", ".join(sorted(negative_terms)),
                num_inference_steps=steps,
                guidance_scale=cfg,
                width=resolution,
                height=resolution,
                generator=generator,
            )
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(f"Generation failed: {exc}") from exc
        return result.images[0]
