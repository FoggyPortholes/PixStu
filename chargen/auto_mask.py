"""Automatic masking stubs to prepare for AI-assisted editing."""

from typing import Optional

try:
    import numpy as np
except Exception:  # pragma: no cover - optional dependency
    np = None  # type: ignore


def generate_mask(image_array) -> Optional["np.ndarray"]:  # type: ignore[name-defined]
    if np is None:
        return None
    return np.zeros_like(image_array[:, :, 0])
