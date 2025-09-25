"""
Device selector with fallback:
CUDA (NVIDIA) → ZLUDA (Intel/AMD CUDA shim) → MPS (Apple) → CPU
"""
from __future__ import annotations

import os
from typing import Union

try:  # torch is optional for CPU-only installs
    import torch
except ModuleNotFoundError:  # pragma: no cover - exercised without torch
    torch = None  # type: ignore[assignment]


def _zluda_hint() -> str:
    env = (
        os.environ.get("ZLUDA_PATH", "")
        + os.environ.get("LD_PRELOAD", "")
        + os.environ.get("DYLD_INSERT_LIBRARIES", "")
    )
    return env.lower()


def pick_device() -> Union["torch.device", str]:
    if torch is None:
        return "cpu"

    if torch.cuda.is_available():
        return torch.device("cuda")

    if "zluda" in _zluda_hint():
        try:
            if torch.backends.cuda.is_built():
                return torch.device("cuda")
        except Exception:
            pass

    mps_backend = getattr(torch.backends, "mps", None)
    if mps_backend and getattr(mps_backend, "is_available", lambda: False)():
        return torch.device("mps")

    return torch.device("cpu")
