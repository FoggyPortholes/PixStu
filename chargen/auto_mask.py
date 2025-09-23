"""Automatic masking stubs to prepare for AI-assisted editing."""

from typing import Optional

try:
    import numpy as np
except Exception:  # pragma: no cover - optional dependency
    np = None  # type: ignore


def generate_mask(image_array, region: Optional[str] = None) -> Optional["np.ndarray"]:  # type: ignore[name-defined]
    if np is None:
        return None
    alpha_channel = None
    if image_array.shape[-1] == 4:
        alpha_channel = image_array[:, :, 3]
    if alpha_channel is not None:
        mask = np.where(alpha_channel > 0, 255, 0).astype("uint8")
    else:
        gray = np.dot(image_array[..., :3], [0.299, 0.587, 0.114])
        threshold = np.percentile(gray, 60)
        mask = np.where(gray > threshold, 255, 0).astype("uint8")

    h, w = mask.shape
    if region == "upper":
        mask[h // 2 :, :] = 0
    elif region == "lower":
        mask[: h // 2, :] = 0
    elif region == "left":
        mask[:, w // 2 :] = 0
    elif region == "right":
        mask[:, : w // 2] = 0

    return mask
