"""Placeholder inpainting utilities for Pin Editor.

This module provides a minimal implementation so the UI can import
``inpaint_region`` even on platforms where the actual Stable Diffusion
inpainting pipeline is not yet available.  The function simply returns the
input image unchanged, allowing the rest of the application to operate
without crashing.  Once the real inpainting workflow is implemented this
placeholder can be replaced.
"""

from __future__ import annotations

from typing import Optional

from PIL import Image


def inpaint_region(
    base_img: Image.Image,
    mask: Image.Image,
    prompt: str = "",
    ref_img: Optional[Image.Image] = None,
) -> Image.Image:
    """Return the original image for now.

    Args:
        base_img: The image that would be edited.
        mask: The region that would be inpainted.
        prompt: Optional text prompt for future use.
        ref_img: Optional reference image for future use.

    Returns:
        The unmodified ``base_img``.  The arguments are accepted to maintain a
        stable call signature for future implementations.
    """

    if base_img is None:
        raise ValueError("base_img must not be None")

    # The mask, prompt and reference image are unused in this placeholder
    # implementation but retained for API compatibility.
    return base_img
