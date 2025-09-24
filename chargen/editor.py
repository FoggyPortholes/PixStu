"""AI editing helper scaffolds."""

from __future__ import annotations

from typing import Dict, Optional

from PIL import Image

from .presets import get_preset
from .generator import BulletProofGenerator


def apply_edit(
    preset_name: str,
    base_image: Image.Image,
    mask_image: Image.Image | None,
    prompt: str,
    *,
    strength: float = 0.5,
) -> Dict[str, Image.Image | str]:
    """Placeholder edit function that simply regenerates using the preset."""

    if base_image is None:
        raise ValueError("Upload an image to edit.")
    preset = get_preset(preset_name) or {}
    generator = BulletProofGenerator(preset)
    result = generator.generate(prompt, seed=42)
    return {"output": result}
