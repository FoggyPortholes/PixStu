import logging
import os
import time
from typing import Callable, List, Optional

from PIL import Image
import torch
from diffusers import (
    StableDiffusionXLPipeline,
    StableDiffusionXLImg2ImgPipeline,
    StableDiffusionXLControlNetPipeline,
    StableDiffusionXLControlNetImg2ImgPipeline,
    ControlNetModel,
    StableDiffusionXLInpaintPipeline,
)

from . import model_setup


logger = logging.getLogger(__name__)


class CharacterGenerator:
    def __init__(self, preset: dict):
        self.preset = preset or {}
        if torch.cuda.is_available():
            self.device = "cuda"
            self.dtype = torch.float16
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            self.device = "mps"
            self.dtype = torch.float32
        else:
            self.device = "cpu"
            self.dtype = torch.float32
        self.txt2img = None
        self.img2img = None
        self._inpaint = None
        self._warnings: List[str] = []

    def _record_warning(self, message: str) -> None:
        if message not in self._warnings:
            logger.warning(message)
            self._warnings.append(message)

    def _apply_loras(self, pipe):
        adapters = []
        weights = []
        lora_entries = self.preset.get("loras", [])
        preset_name = self.preset.get("name", "Unnamed")
        for idx, entry in enumerate(lora_entries):
            entry = dict(entry or {})
            raw_path = entry.get("local_path") or entry.get("path") or ""
            record = model_setup.resolve_path(raw_path) if raw_path else None
            if record is None and entry.get("name"):
                record = model_setup.find_record(entry["name"])
            if record and not record.exists and entry.get("repo_id"):
                logger.info(
                    "Downloading LoRA '%s' for preset '%s'.",
                    record.name,
                    preset_name,
                )
                logger.info(model_setup.download(record.name))
                record = model_setup.resolve_path(record.path) or record
            elif record is None and entry.get("repo_id") and entry.get("name"):
                logger.info(
                    "Attempting to download LoRA '%s' for preset '%s'.",
                    entry["name"],
                    preset_name,
                )
                logger.info(model_setup.download(entry["name"]))
                record = model_setup.find_record(entry["name"])
            if record is None:
                label = entry.get("name") or entry.get("path") or entry.get("local_path") or f"index {idx}"
                self._record_warning(
                    f"Preset '{preset_name}' skipped LoRA '{label}': unable to resolve path."
                )
                continue
            if not record.exists:
                self._record_warning(
                    f"Preset '{preset_name}' skipped LoRA '{record.name}': file not available locally."
                )
                continue
            local_path = record.local_path
            weight = float(entry.get("weight", 1.0))
            adapter_name = entry.get("name") or record.name or f"preset_lora_{idx}"
            load_dir = os.path.dirname(local_path)
            weight_name = os.path.basename(local_path)
            try:
                pipe.load_lora_weights(load_dir, adapter_name=adapter_name, weight_name=weight_name)
            except Exception as exc:
                self._record_warning(f"Failed to load LoRA '{adapter_name}': {exc}")
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
                    self._record_warning(
                        f"Unable to set LoRA weights for preset '{preset_name}': {fuse_exc}"
                    )
        elif lora_entries:
            self._record_warning(
                f"Preset '{preset_name}' has no usable LoRAs; generation will proceed without them."
            )
        return pipe

    def consume_warnings(self) -> List[str]:
        warnings = list(self._warnings)
        self._warnings.clear()
        return warnings

    def _ensure_txt2img(self):
        if self.txt2img is None:
            base = self.preset.get("base_model", "stabilityai/stable-diffusion-xl-base-1.0")
            self.txt2img = StableDiffusionXLPipeline.from_pretrained(base, torch_dtype=self.dtype)
            self._apply_loras(self.txt2img)
            if hasattr(self.txt2img, "to"):
                self.txt2img.to(self.device)
        return self.txt2img

    def _ensure_img2img(self):
        if self.img2img is None:
            base = self.preset.get("base_model", "stabilityai/stable-diffusion-xl-base-1.0")
            self.img2img = StableDiffusionXLImg2ImgPipeline.from_pretrained(base, torch_dtype=self.dtype)
            self._apply_loras(self.img2img)
            if hasattr(self.img2img, "to"):
                self.img2img.to(self.device)
        return self.img2img

    def _create_pipeline(
        self,
        use_img2img: bool,
        controlnet_config: Optional[dict] = None,
        ip_adapter_config: Optional[dict] = None,
    ):
        base_model = self.preset.get("base_model", "stabilityai/stable-diffusion-xl-base-1.0")
        pipe = None
        if controlnet_config:
            record = controlnet_config.get("record")
            if record is None:
                self._record_warning("ControlNet configuration missing record; falling back to base pipeline.")
            else:
                control_source = getattr(record, "source", None)
                try:
                    control_model = ControlNetModel.from_pretrained(control_source, torch_dtype=self.dtype)
                except Exception as exc:
                    self._record_warning(f"Failed to load ControlNet '{record.name}': {exc}")
                    control_model = None
                if control_model is not None:
                    if use_img2img:
                        pipe = StableDiffusionXLControlNetImg2ImgPipeline.from_pretrained(
                            base_model,
                            controlnet=control_model,
                            torch_dtype=self.dtype,
                        )
                    else:
                        pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
                            base_model,
                            controlnet=control_model,
                            torch_dtype=self.dtype,
                        )
        if pipe is None:
            if use_img2img:
                pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(base_model, torch_dtype=self.dtype)
            else:
                pipe = StableDiffusionXLPipeline.from_pretrained(base_model, torch_dtype=self.dtype)

        if hasattr(pipe, "to"):
            pipe.to(self.device)
        self._apply_loras(pipe)
        extra_kwargs = {}
        if ip_adapter_config:
            extra_kwargs.update(self._apply_ip_adapter(pipe, ip_adapter_config))
        return pipe, extra_kwargs

    def _apply_ip_adapter(self, pipe, config: dict) -> dict:
        record = config.get("record")
        if record is None:
            return {}
        source = getattr(record, "source", None)
        kwargs: dict = {}
        try:
            load_kwargs = {"subfolder": getattr(record, "subfolder", None)}
            if getattr(record, "weight_name", None):
                load_kwargs["weight_name"] = record.weight_name
            pipe.load_ip_adapter(source, **load_kwargs)
        except Exception as exc:
            self._record_warning(f"Failed to load IP-Adapter '{getattr(record, "name", "unknown")}': {exc}")
            return {}

        scale = config.get("scale")
        if scale is not None and hasattr(pipe, "set_ip_adapter_scale"):
            try:
                pipe.set_ip_adapter_scale(float(scale))
            except Exception as exc:
                self._record_warning(f"Unable to set IP-Adapter scale: {exc}")
        image = config.get("image")
        if image is not None:
            kwargs["ip_adapter_image"] = image
        return kwargs

    def _ensure_inpaint(self):
        if self._inpaint is None:
            base = self.preset.get("base_model", "stabilityai/stable-diffusion-xl-base-1.0")
            self._inpaint = StableDiffusionXLInpaintPipeline.from_pretrained(base, torch_dtype=self.dtype)
            self._apply_loras(self._inpaint)
            if hasattr(self._inpaint, "to"):
                self._inpaint.to(self.device)
        return self._inpaint

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
        negative_prompt: Optional[str] = None,
        size: int = 512,
        progress_callback: Optional[Callable[[int, int, Image.Image], None]] = None,
        progress_interval: int = 4,
        controlnet_config: Optional[dict] = None,
        ip_adapter_config: Optional[dict] = None,
    ) -> str:
        steps = int(steps or self.preset.get("suggested", {}).get("steps", 30))
        guidance = float(guidance or self.preset.get("suggested", {}).get("guidance", 7.0))
        gen = torch.manual_seed(int(seed))
        if controlnet_config or ip_adapter_config:
            pipe, extra_kwargs = self._create_pipeline(False, controlnet_config, ip_adapter_config)
        else:
            pipe = self._ensure_txt2img()
            extra_kwargs = {}
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
        call_kwargs = dict(extra_kwargs)
        if negative_prompt is not None:
            call_kwargs["negative_prompt"] = negative_prompt
        if controlnet_config:
            call_kwargs["controlnet_conditioning_image"] = controlnet_config.get("image")
            call_kwargs["controlnet_conditioning_scale"] = controlnet_config.get("scale", 1.0)
        image = pipe(
            prompt=prompt,
            num_inference_steps=steps,
            guidance_scale=guidance,
            generator=gen,
            height=size,
            width=size,
            **callback_kwargs,
            **call_kwargs,
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
        negative_prompt: Optional[str] = None,
        size: int = 512,
        progress_callback: Optional[Callable[[int, int, Image.Image], None]] = None,
        progress_interval: int = 4,
        controlnet_config: Optional[dict] = None,
        ip_adapter_config: Optional[dict] = None,
    ) -> str:
        steps = int(steps or self.preset.get("suggested", {}).get("steps", 30))
        guidance = float(guidance or self.preset.get("suggested", {}).get("guidance", 7.0))
        gen = torch.manual_seed(int(seed))
        base = Image.open(ref_image_path).convert("RGBA").resize((size, size), Image.NEAREST)
        if controlnet_config or ip_adapter_config:
            pipe, extra_kwargs = self._create_pipeline(True, controlnet_config, ip_adapter_config)
        else:
            pipe = self._ensure_img2img()
            extra_kwargs = {}
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
        call_kwargs = dict(extra_kwargs)
        if negative_prompt is not None:
            call_kwargs["negative_prompt"] = negative_prompt
        if controlnet_config:
            call_kwargs["controlnet_conditioning_image"] = controlnet_config.get("image")
            call_kwargs["controlnet_conditioning_scale"] = controlnet_config.get("scale", 1.0)
        out = pipe(
            prompt=prompt,
            image=base,
            strength=float(strength),
            guidance_scale=guidance,
            num_inference_steps=steps,
            generator=gen,
            **callback_kwargs,
            **call_kwargs,
        ).images[0]
        model_setup.ensure_directories()
        path = os.path.join(model_setup.OUTPUTS, f"char_refined_{int(time.time())}.png")
        out.save(path)
        logger.info("Refinement complete | path=%s", path)
        return path

    def inpaint(
        self,
        init_image: Image.Image,
        mask_image: Image.Image,
        prompt: str,
        *,
        strength: float = 0.5,
        steps: int = 30,
        guidance: float = 7.0,
        negative_prompt: Optional[str] = None,
    ) -> str:
        pipe = self._ensure_inpaint()
        result = pipe(
            prompt=prompt,
            image=init_image,
            mask_image=mask_image,
            strength=strength,
            num_inference_steps=steps,
            guidance_scale=guidance,
            negative_prompt=negative_prompt,
        ).images[0]
        path = os.path.join(model_setup.OUTPUTS, f"char_edit_{int(time.time())}.png")
        result.save(path)
        logger.info("Inpaint complete | path=%s", path)
        return path


