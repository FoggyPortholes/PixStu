import logging

import torch
from PIL import Image

try:
    from diffusers import StableDiffusionXLPipeline, StableDiffusionXLControlNetPipeline
except Exception:  # pragma: no cover - optional dependency fallback
    StableDiffusionXLPipeline = StableDiffusionXLControlNetPipeline = None  # type: ignore

logger = logging.getLogger(__name__)


def _detect_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    try:  # AMD via ZLUDA
        import zluda  # type: ignore  # pragma: no cover

        return "cuda"
    except Exception:  # pragma: no cover - optional
        pass
    try:  # Intel via zkluda
        import zkluda  # type: ignore  # pragma: no cover

        return "cuda"
    except Exception:  # pragma: no cover - optional
        pass
    return "cpu"


class BulletProofGenerator:
    """Minimal generator facade that enforces positives/negatives and device fallbacks."""

    def __init__(self, preset: dict):
        self.preset = preset or {}
        self.device = _detect_device()
        model_id = self.preset.get("model", "stabilityai/stable-diffusion-xl-base-1.0")
        dtype = getattr(torch, "float16", None) if self.device == "cuda" else getattr(torch, "float32", None)
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
        if self.device == "cuda":
            try:
                if hasattr(self.pipe, "enable_xformers_memory_efficient_attention"):
                    self.pipe.enable_xformers_memory_efficient_attention()
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
            path = entry.get("path")
            weight = float(entry.get("weight", 1.0))
            if not path or not hasattr(self.pipe, "load_lora_weights"):
                continue
            try:
                self.pipe.load_lora_weights(path, weight=weight)
            except Exception as exc:  # pragma: no cover
                logger.warning("LoRA load failed (%s): %s", path, exc)

    def generate(self, prompt: str, seed: int = 42) -> Image.Image:
        generator = torch.Generator(self.device).manual_seed(int(seed))
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
