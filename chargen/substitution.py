import logging
from typing import Optional

import torch
from PIL import Image

from tools.device import pick_device

try:
    from diffusers import (
        ControlNetModel,
        StableDiffusionXLControlNetPipeline,
        StableDiffusionXLPipeline,
    )
except Exception:  # pragma: no cover - optional dependency fallback
    ControlNetModel = StableDiffusionXLControlNetPipeline = StableDiffusionXLPipeline = None  # type: ignore

try:
    from controlnet_aux import OpenposeDetector
except Exception:  # pragma: no cover - optional dependency fallback
    OpenposeDetector = None

logger = logging.getLogger(__name__)


class SubstitutionEngine:
    """Simple identity?pose substitution scaffold."""

    def __init__(self, preset: dict, controlnet_id: str = "lllyasviel/sd-controlnet-openpose") -> None:
        if StableDiffusionXLPipeline is None:
            raise RuntimeError("Diffusers pipelines unavailable; install diffusers")
        self.preset = preset or {}
        self.device = pick_device()
        dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        base_model = self.preset.get("model", "stabilityai/stable-diffusion-xl-base-1.0")
        self.controlnet = None
        if ControlNetModel is not None:
            try:
                self.controlnet = ControlNetModel.from_pretrained(controlnet_id, torch_dtype=dtype)
            except Exception as exc:
                logger.warning("ControlNet load failed: %s", exc)
                self.controlnet = None
        if self.controlnet is not None and StableDiffusionXLControlNetPipeline is not None:
            self.pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
                base_model,
                controlnet=self.controlnet,
                torch_dtype=dtype,
            )
        else:
            self.pipe = StableDiffusionXLPipeline.from_pretrained(base_model, torch_dtype=dtype)
        if hasattr(self.pipe, "to"):
            self.pipe = self.pipe.to(self.device)

        self.pose_detector: Optional[object] = None
        if OpenposeDetector is not None:
            try:
                self.pose_detector = OpenposeDetector.from_pretrained("lllyasviel/ControlNet")
            except Exception as exc:
                logger.warning("OpenPose detector load failed: %s", exc)
                self.pose_detector = None

    def _pose_map(self, pose_image: Optional[Image.Image]):
        if pose_image is None or self.pose_detector is None:
            return None
        try:
            return self.pose_detector(pose_image)
        except Exception as exc:  # pragma: no cover
            logger.warning("Pose mapping failed: %s", exc)
            return None

    def run(
        self,
        char1_identity: Optional[Image.Image],
        char2_pose: Optional[Image.Image],
        prompt: str,
        identity_strength: float = 0.7,
        pose_strength: float = 1.0,
        seed: int = 42,
    ) -> Image.Image:
        generator = torch.Generator(device=self.device).manual_seed(int(seed))
        positives = list(self.preset.get("positive", []))
        negatives = set(self.preset.get("negative", []))
        negatives.update({"duplicate", "text", "caption", "speech bubble", "watermark", "logo"})
        composed_prompt = f"{prompt}, " + ", ".join(positives) if positives else prompt
        kwargs = dict(
            prompt=composed_prompt,
            negative_prompt=", ".join(sorted(negatives)),
            num_inference_steps=self.preset.get("steps", 30),
            guidance_scale=self.preset.get("cfg", 7.5),
            width=self.preset.get("resolution", 768),
            height=self.preset.get("resolution", 768),
            generator=generator,
        )
        if self.controlnet is not None and hasattr(self.pipe, "__call__"):
            pose = self._pose_map(char2_pose) or char2_pose
            kwargs["image"] = pose
            kwargs["controlnet_conditioning_scale"] = pose_strength
        result = self.pipe(**kwargs).images[0]
        # Optional blending toward identity image could be applied here when IP-Adapter is wired.
        return result
