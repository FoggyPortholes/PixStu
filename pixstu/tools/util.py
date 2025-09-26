"""Utility helpers shared across PixStu modules."""

from __future__ import annotations

import random
from typing import Optional


def set_seed(seed: Optional[int]) -> None:
    """Seed random number generators where available."""
    if seed is None:
        return
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except Exception:
        pass
    try:
        import torch

        torch.manual_seed(seed)
    except Exception:
        pass
