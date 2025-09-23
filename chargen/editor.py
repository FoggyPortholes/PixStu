"""AI-assisted editing helpers leveraging Stable Diffusion XL inpainting."""

from __future__ import annotations

import os
from typing import Dict, Optional

import numpy as np
from PIL import Image

from .auto_mask import generate_mask
from .generator import CharacterGenerator
from .metadata import save_metadata
from .model_setup import OUTPUTS


def _ensure_mask(image: Image.Image, mask_path: Optional[str], target_region: str, auto_mask_enabled: bool) -> Image.Image:
    if mask_path:
        mask = Image.open(mask_path).convert("L")
        return mask.resize(image.size)

    if not auto_mask_enabled:
        return Image.new("L", image.size, 255)

    array = np.array(image.convert("RGBA"))
    mask_array = generate_mask(array, None if target_region == "entire" else target_region)
    if mask_array is None:
        return Image.new("L", image.size, 255)
    return Image.fromarray(mask_array).resize(image.size)


def apply_edit(
    preset: Dict,
    image_path: str,
    mask_path: Optional[str],
    prompt: str,
    *,
    strength: float = 0.5,
    auto_mask_enabled: bool = False,
    target_region: str = "entire",
) -> Dict[str, str]:
    if not image_path:
        raise ValueError("Upload an image to edit.")
    if not prompt.strip():
        raise ValueError("Provide an edit prompt.")

    base_image = Image.open(image_path).convert("RGB")
    mask_image = _ensure_mask(base_image, mask_path, target_region, auto_mask_enabled)

    generator = CharacterGenerator(preset)
    suggested = preset.get("suggested", {})
    steps = int(suggested.get("steps", 30))
    guidance = float(suggested.get("guidance", 7.0))
    negative_prompt = preset.get("negative_prompt")

    output_path = generator.inpaint(
        base_image,
        mask_image,
        prompt,
        strength=float(strength),
        steps=steps,
        guidance=guidance,
        negative_prompt=negative_prompt,
    )

    metadata = {
        "prompt": prompt,
        "preset": preset.get("name"),
        "type": "ai_edit",
        "source_image": os.path.basename(image_path),
        "mask_source": os.path.basename(mask_path) if mask_path else ("auto" if auto_mask_enabled else "full"),
        "strength": float(strength),
        "target_region": target_region,
    }
    meta_path = save_metadata(os.path.dirname(output_path) or OUTPUTS, metadata)

    return {
        "output_path": output_path,
        "metadata_path": meta_path,
    }
