import os
import time
from typing import Optional

from PIL import Image
import torch
from diffusers import StableDiffusionXLPipeline, StableDiffusionXLImg2ImgPipeline

from . import model_setup


class CharacterGenerator:
    def __init__(self, preset: dict):
        self.preset = preset or {}
        self.dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        self.txt2img = None
        self.img2img = None

    def _apply_loras(self, pipe):
        adapters = []
        weights = []
        lora_entries = self.preset.get("loras", [])
        for idx, entry in enumerate(lora_entries):
            raw_path = entry.get("local_path") or entry.get("path") or ""
            record = model_setup.resolve_path(raw_path)
            if record is None and entry.get("name"):
                record = model_setup.find_record(entry["name"])
            if record and not record.exists and entry.get("repo_id"):
                print(
                    f"[INFO] Downloading LoRA '{record.name}' for preset '{self.preset.get('name', 'Unnamed')}'."
                )
                print(f"[INFO] {model_setup.download(record.name)}")
                record = model_setup.resolve_path(record.path) or record
            elif record is None and entry.get("repo_id") and entry.get("name"):
                print(
                    f"[INFO] Attempting to download LoRA '{entry['name']}' for preset '{self.preset.get('name', 'Unnamed')}'."
                )
                print(f"[INFO] {model_setup.download(entry['name'])}")
                record = model_setup.find_record(entry["name"])
            if record is None:
                print(f"[WARN] LoRA entry {entry!r} could not be resolved.")
                continue
            if not record.exists:
                print(f"[WARN] LoRA '{record.name}' is not available locally.")
                continue
            local_path = record.local_path
            weight = float(entry.get("weight", 1.0))
            adapter_name = entry.get("name") or record.name or f"preset_lora_{idx}"
            load_dir = os.path.dirname(local_path)
            weight_name = os.path.basename(local_path)
            try:
                pipe.load_lora_weights(load_dir, adapter_name=adapter_name, weight_name=weight_name)
            except Exception as exc:
                print(f"[WARN] LoRA load failed for {adapter_name}: {exc}")
                continue
            adapters.append(adapter_name)
            weights.append(weight)
        if adapters:
            try:
                pipe.set_adapters(adapters, adapter_weights=weights)
            except Exception as exc:
                try:
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
        model_setup.ensure_directories()
        path = os.path.join(model_setup.OUTPUTS, f"char_{int(time.time())}.png")
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
        model_setup.ensure_directories()
        path = os.path.join(model_setup.OUTPUTS, f"char_refined_{int(time.time())}.png")
        out.save(path)
        return path
