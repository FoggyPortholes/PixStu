from typing import Optional
import torch, logging
from PIL import Image

try:
    from diffusers import (
        StableDiffusionXLPipeline,
        StableDiffusionXLControlNetPipeline,
        ControlNetModel,
    )
except Exception:
    StableDiffusionXLPipeline = StableDiffusionXLControlNetPipeline = ControlNetModel = None  # type: ignore

try:
    from controlnet_aux import OpenposeDetector
except Exception:
    OpenposeDetector = None

logging.basicConfig(level=logging.INFO)


def _device():
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    try:
        import zluda  # noqa: F401
        return "cuda"
    except Exception:
        pass
    try:
        import zkluda  # noqa: F401
        return "cuda"
    except Exception:
        pass
    return "cpu"


class SubstitutionEngine:
    def __init__(self, preset: dict, ip_adapter_path: Optional[str] = None, controlnet_id: str = "lllyasviel/sd-controlnet-openpose"):
        self.preset = preset
        self.device = _device()
        self.model_id = preset.get("model", "stabilityai/stable-diffusion-xl-base-1.0")
        self.controlnet = None
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        if ControlNetModel is not None:
            try:
                self.controlnet = ControlNetModel.from_pretrained(controlnet_id, torch_dtype=dtype)
            except Exception as e:
                logging.warning(f"ControlNet load failed: {e}")
                self.controlnet = None
        if self.controlnet is not None and StableDiffusionXLControlNetPipeline is not None:
            self.pipe = StableDiffusionXLControlNetPipeline.from_pretrained(self.model_id, controlnet=self.controlnet, torch_dtype=dtype)
        else:
            self.pipe = StableDiffusionXLPipeline.from_pretrained(self.model_id, torch_dtype=dtype)
        self.pipe = self.pipe.to(self.device)

        # Optional LoRAs
        for l in self.preset.get("loras", []):
            try:
                if hasattr(self.pipe, "load_lora_weights"):
                    self.pipe.load_lora_weights(l["path"], weight=l.get("weight", 1.0))
            except Exception as e:
                logging.warning(f"LoRA load failed ({l.get('path')}): {e}")

        # Optional IP-Adapter
        self.ip_adapter_loaded = False
        if ip_adapter_path and hasattr(self.pipe, "load_ip_adapter"):
            try:
                self.pipe.load_ip_adapter(ip_adapter_path)
                self.ip_adapter_loaded = True
            except Exception as e:
                logging.warning(f"IP-Adapter load failed: {e}")
                self.ip_adapter_loaded = False

        # Optional OpenPose
        self.pose_detector = None
        if OpenposeDetector is not None:
            try:
                self.pose_detector = OpenposeDetector.from_pretrained("lllyasviel/ControlNet")
            except Exception as e:
                logging.warning(f"OpenPose detector load failed: {e}")
                self.pose_detector = None

    def _openpose_map(self, pose_image: Image.Image):
        if self.pose_detector is None:
            return None
        try:
            return self.pose_detector(pose_image)
        except Exception as e:
            logging.warning(f"OpenPose mapping failed: {e}")
            return None

    def run(self, char1_identity: Optional[Image.Image], char2_pose: Optional[Image.Image], prompt: str, identity_strength: float = 0.7, pose_strength: float = 1.0, seed: int = 42) -> Image.Image:
        g = torch.Generator(self.device).manual_seed(seed)
        positive = list(self.preset.get("positive", []))
        negative = set(self.preset.get("negative", []))
        negative.update({"duplicate", "text", "caption", "speech bubble", "watermark", "logo"})
        pose_cond = None
        if char2_pose is not None:
            pose_cond = self._openpose_map(char2_pose) or char2_pose
        kwargs = dict(
            prompt=f"{prompt}, "+", ".join(positive),
            negative_prompt=", ".join(sorted(negative)),
            num_inference_steps=self.preset.get("steps", 30),
            guidance_scale=self.preset.get("cfg", 7.5),
            width=self.preset.get("resolution", 768), height=self.preset.get("resolution", 768),
            generator=g,
        )
        if self.controlnet is not None and pose_cond is not None and hasattr(self.pipe, "__call__"):
            out = self.pipe(image=pose_cond, controlnet_conditioning_scale=pose_strength, **kwargs).images[0]
        else:
            out = self.pipe(**kwargs).images[0]
        return out
