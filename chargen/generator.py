import torch, logging
from PIL import Image
try:
    from diffusers import StableDiffusionXLPipeline, StableDiffusionXLControlNetPipeline
except Exception:
    StableDiffusionXLPipeline = StableDiffusionXLControlNetPipeline = None  # type: ignore

logging.basicConfig(level=logging.INFO)


def _device():
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    # AMD ZLUDA stub
    try:
        import zluda  # noqa: F401
        return "cuda"
    except Exception:
        pass
    # Intel zkluda stub
    try:
        import zkluda  # noqa: F401
        return "cuda"
    except Exception:
        pass
    return "cpu"


class BulletProofGenerator:
    def __init__(self, preset: dict):
        self.preset = preset
        self.device = _device()
        model_id = preset.get("model", "stabilityai/stable-diffusion-xl-base-1.0")
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        try:
            if preset.get("controlnets") and StableDiffusionXLControlNetPipeline is not None:
                self.pipe = StableDiffusionXLControlNetPipeline.from_pretrained(model_id, torch_dtype=dtype)
            else:
                self.pipe = StableDiffusionXLPipeline.from_pretrained(model_id, torch_dtype=dtype)
        except Exception as e:
            logging.error(f"Failed to load pipeline for {model_id}: {e}")
            raise
        self.pipe = self.pipe.to(self.device)

        # Speed opts (best effort)
        if self.device == "cuda":
            try:
                if hasattr(self.pipe, "enable_xformers_memory_efficient_attention"):
                    self.pipe.enable_xformers_memory_efficient_attention()
                if hasattr(torch, "compile") and hasattr(self.pipe, "unet"):
                    self.pipe.unet = torch.compile(self.pipe.unet, mode="reduce-overhead", fullgraph=True)
            except Exception as e:
                logging.warning(f"CUDA speed opts not applied: {e}")
        else:
            try:
                if hasattr(self.pipe, "enable_vae_slicing"):
                    self.pipe.enable_vae_slicing()
                if hasattr(self.pipe, "enable_vae_tiling"):
                    self.pipe.enable_vae_tiling()
            except Exception as e:
                logging.warning(f"VAE opts not applied: {e}")

        # Load LoRAs
        for l in preset.get("loras", []):
            try:
                if hasattr(self.pipe, "load_lora_weights"):
                    self.pipe.load_lora_weights(l["path"], weight=l.get("weight", 1.0))
            except Exception as e:
                logging.warning(f"LoRA load failed ({l.get('path')}): {e}")

    def generate(self, prompt: str, seed: int = 42) -> Image.Image:
        g = torch.Generator(self.device).manual_seed(int(seed))
        pos = list(self.preset.get("positive", []))
        neg = set(self.preset.get("negative", []))
        neg.update({"duplicate", "text", "caption", "speech bubble", "watermark", "logo"})
        try:
            out = self.pipe(
                prompt=f"{prompt}, "+", ".join(pos),
                negative_prompt=", ".join(sorted(neg)),
                num_inference_steps=self.preset.get("steps", 30),
                guidance_scale=self.preset.get("cfg", 7.5),
                width=self.preset.get("resolution", 768),
                height=self.preset.get("resolution", 768),
                generator=g,
            ).images[0]
            return out
        except Exception as e:
            logging.error(f"Generation failed: {e}")
            raise
