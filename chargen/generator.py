import logging
import os
import time
from typing import Callable, Optional

from PIL import Image
import torch
from diffusers import StableDiffusionXLPipeline, StableDiffusionXLImg2ImgPipeline

from . import model_setup


logger = logging.getLogger(__name__)


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
                logger.warning("LoRA entry could not be resolved: %s", entry)
                continue
            if not record.exists:
                logger.warning("LoRA '%s' is not available locally.", record.name)
                continue
            local_path = record.local_path
            weight = float(entry.get("weight", 1.0))
            adapter_name = entry.get("name") or record.name or f"preset_lora_{idx}"
            load_dir = os.path.dirname(local_path)
            weight_name = os.path.basename(local_path)
            try:
                pipe.load_lora_weights(load_dir, adapter_name=adapter_name, weight_name=weight_name)
            except Exception as exc:
                logger.warning("LoRA load failed for %s: %s", adapter_name, exc)
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
                    logger.warning("Unable to set LoRA weights: %s", fuse_exc)
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

    def _make_progress_callbacks(
        self,
        pipe,
        total_steps: int,
        progress_callback: Optional[Callable[[int, int, Image.Image], None]],
        progress_interval: int,
    ):
        if progress_callback is None:
            return None, None

        interval = max(1, int(progress_interval))

        def _decode_images(pipeline, latents):
            vae = getattr(pipeline, "vae", None)
            device = getattr(pipeline, "device", latents.device)
            latents = latents.to(device=device, dtype=getattr(vae, "dtype", torch.float32))
            if hasattr(pipeline, "decode_latents"):
                decoded = pipeline.decode_latents(latents)
            elif vae is not None:
                scaling = getattr(getattr(vae, "config", object()), "scaling_factor", 0.18215)
                latents = latents / scaling
                decoded = vae.decode(latents).sample
            else:
                raise AttributeError("Pipeline has no decoder for preview frames.")
            return pipeline.image_processor.postprocess(decoded, output_type="pil")

        def _emit(pipeline, step: int, latents):
            if step % interval and step != total_steps - 1:
                return
            try:
                with torch.no_grad():
                    images = _decode_images(pipeline, latents)
                if images:
                    progress_callback(step, total_steps, images[0])
            except Exception as exc:  # pragma: no cover - preview failures are non-fatal
                logger.warning("Preview callback failed: %s", exc)

        def _legacy_callback(step: int, _timestep: int, latents):
            if latents is None:
                return
            _emit(pipe, step, latents)

        def _step_end_callback(pipeline, step: int, timestep, callback_kwargs):  # noqa: D401
            latents = callback_kwargs.get("latents") if isinstance(callback_kwargs, dict) else None
            if latents is not None:
                _emit(pipeline, step, latents)
            return {}

        return _legacy_callback, _step_end_callback

    def generate(
        self,
        prompt: str,
        seed: int = 42,
        steps: Optional[int] = None,
        guidance: Optional[float] = None,
        size: int = 512,
        progress_callback: Optional[Callable[[int, int, Image.Image], None]] = None,
        progress_interval: int = 4,
    ) -> str:
        steps = int(steps or self.preset.get("suggested", {}).get("steps", 30))
        guidance = float(guidance or self.preset.get("suggested", {}).get("guidance", 7.0))
        gen = torch.manual_seed(int(seed))
        pipe = self._ensure_txt2img()
        logger.info(
            "Starting txt2img generation | preset=%s seed=%s size=%s steps=%s guidance=%s",
            self.preset.get("name"),
            seed,
            size,
            steps,
            guidance,
        )
        legacy_cb, step_end_cb = self._make_progress_callbacks(pipe, steps, progress_callback, progress_interval)
        callback_kwargs = {}
        if step_end_cb:
            callback_kwargs["callback_on_step_end"] = step_end_cb
        elif legacy_cb:
            callback_kwargs["callback"] = legacy_cb
            callback_kwargs["callback_steps"] = 1
        image = pipe(
            prompt=prompt,
            num_inference_steps=steps,
            guidance_scale=guidance,
            generator=gen,
            height=size,
            width=size,
            **callback_kwargs,
        ).images[0]
        model_setup.ensure_directories()
        path = os.path.join(model_setup.OUTPUTS, f"char_{int(time.time())}.png")
        image.save(path)
        logger.info("Generation complete | path=%s", path)
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
        progress_callback: Optional[Callable[[int, int, Image.Image], None]] = None,
        progress_interval: int = 4,
    ) -> str:
        steps = int(steps or self.preset.get("suggested", {}).get("steps", 30))
        guidance = float(guidance or self.preset.get("suggested", {}).get("guidance", 7.0))
        gen = torch.manual_seed(int(seed))
        base = Image.open(ref_image_path).convert("RGBA").resize((size, size), Image.NEAREST)
        pipe = self._ensure_img2img()
        logger.info(
            "Starting img2img refinement | preset=%s seed=%s size=%s strength=%s",
            self.preset.get("name"),
            seed,
            size,
            strength,
        )
        legacy_cb, step_end_cb = self._make_progress_callbacks(pipe, steps, progress_callback, progress_interval)
        callback_kwargs = {}
        if step_end_cb:
            callback_kwargs["callback_on_step_end"] = step_end_cb
        elif legacy_cb:
            callback_kwargs["callback"] = legacy_cb
            callback_kwargs["callback_steps"] = 1
        out = pipe(
            prompt=prompt,
            image=base,
            strength=float(strength),
            guidance_scale=guidance,
            num_inference_steps=steps,
            generator=gen,
            **callback_kwargs,
        ).images[0]
        model_setup.ensure_directories()
        path = os.path.join(model_setup.OUTPUTS, f"char_refined_{int(time.time())}.png")
        out.save(path)
        logger.info("Refinement complete | path=%s", path)
        return path
