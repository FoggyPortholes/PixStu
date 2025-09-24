from __future__ import annotations

import os
from typing import Literal

try:
    import torch
except Exception:  # pragma: no cover - optional dependency
    torch = None  # type: ignore

Device = Literal["cuda", "rocm", "mps", "cpu"]


def detect_device() -> Device:
    if torch is None:
        return "cpu"
    if torch.cuda.is_available():
        if os.getenv("ZLUDA") or os.getenv("ZLUDA_DEVICE"):
            return "rocm"
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    if os.getenv("ZKLUDA"):
        return "rocm"
    return "cpu"
