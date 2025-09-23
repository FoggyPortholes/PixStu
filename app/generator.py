import os
import time
from typing import Optional
from PIL import Image
import torch
from diffusers import StableDiffusionXLPipeline, StableDiffusionXLImg2ImgPipeline
from .paths import OUTPUTS, MODELS, ROOT


class CharacterGenerator:
    def __init__(self, preset: dict):
        self.preset = preset or {}
        self.dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        self.txt2img = None
        self.img2img = None

    def _apply_loras(self, pipe):
        adapters = []
        weights = []
        for idx, entry in enumerate(self.preset.get("loras", [])):
            raw_path = entry.get("path", "")
            if not raw_path:
                continue
            if not os.path.isabs(raw_path):
                candidates = [os.path.join(ROOT, raw_path), os.path.join(MODELS, raw_path)]
                full_path = next((c for c in candidates if os.path.exists(c)), candidates[0])
            else:
                full_path = raw_path
            weight = float(entry.get("weight", 1.0))
            load_kwargs = {}
            load_path = full_path
            if os.path.isfile(full_path):
                load_kwargs["weight_name"] = os.path.basename(full_path)
                load_path = os.path.dirname(full_path)
            adapter_name = entry.get("name") or f"preset_lora_{idx}"
            try:
                pipe.load_lora_weights(load_path, adapter_name=adapter_name, **load_kwargs)
            except Exception as exc:
                print(f"[WARN] LoRA failed: {exc}")
                continue
            adapters.append(adapter_name)
            weights.append(weight)
        if adapters:
            try:
                pipe.set_adapters(adapters, adapter_weights=weights)
            except Exception as exc:
                try:
                    # Fallback: fuse single adapter if advanced control is unavailable.
                    if len(adapters) == 1 and hasattr(pipe, "fuse_lora"):
                        pipe.fuse_lora(lora_scale=weights[0])
                    else:
                        raise exc
                except Exception as fuse_exc:
                    print(f"[WARN] Unable to set LoRA weights: {fuse_exc}")
        return pipe

    def _ensure_txt2img(self):
        if self.txt2img is None:
            base = self.preset.get("base_model", "stabilityai/stable-diffusion-xl-base-1.0")
            self.txt2img = StableDiffusionXLPipeline.from_pretrained(base, torch_dtype=self.dtype)
            self._apply_loras(self.txt2img)
        return self.txt2img

    def _ensure_img2img(self):
        if self.img2img is None:
            base = self.preset.get("base_model", "stabilityai/stable-diffusion-xl-base-1.0")
            self.img2img = StableDiffusionXLImg2ImgPipeline.from_pretrained(base, torch_dtype=self.dtype)
            self._apply_loras(self.img2img)
        return self.img2img

    def generate(
        self,
        prompt: str,
        seed: int = 42,
        steps: Optional[int] = None,
        guidance: Optional[float] = None,
        size: int = 512,
    ) -> str:
        steps = int(steps or self.preset.get("suggested", {}).get("steps", 30))
        guidance = float(guidance or self.preset.get("suggested", {}).get("guidance", 7.0))
        gen = torch.manual_seed(int(seed))
        pipe = self._ensure_txt2img()
        image = pipe(
            prompt=prompt,
            num_inference_steps=steps,
            guidance_scale=guidance,
            generator=gen,
            height=size,
            width=size,
        ).images[0]
        path = os.path.join(OUTPUTS, f"char_{int(time.time())}.png")
        image.save(path)
        return path

    def refine(
        self,
        ref_image_path: str,
        prompt: str,
        strength: float = 0.35,
        seed: int = 42,
        steps: Optional[int] = None,
        guidance: Optional[float] = None,
        size: int = 512,
    ) -> str:
        steps = int(steps or self.preset.get("suggested", {}).get("steps", 30))
        guidance = float(guidance or self.preset.get("suggested", {}).get("guidance", 7.0))
        gen = torch.manual_seed(int(seed))
        base = Image.open(ref_image_path).convert("RGBA").resize((size, size), Image.NEAREST)
        pipe = self._ensure_img2img()
        out = pipe(
            prompt=prompt,
            image=base,
            strength=float(strength),
            guidance_scale=guidance,
            num_inference_steps=steps,
            generator=gen,
        ).images[0]
        path = os.path.join(OUTPUTS, f"char_refined_{int(time.time())}.png")
        out.save(path)
        return path
