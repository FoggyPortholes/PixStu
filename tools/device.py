"""Device selector for PixStu.

Fallback chain:
CUDA (NVIDIA) → ZLUDA (Intel/AMD CUDA shim) → MPS (Apple) → CPU
"""

from __future__ import annotations

import os

import torch


def pick_device() -> torch.device:
    """Return the best available torch device for PixStu workloads."""

    # 1) Native CUDA
    if torch.cuda.is_available():
        return torch.device("cuda")

    # 2) ZLUDA probe (Intel/AMD CUDA shim)
    # Detect via env or preload hints
    zluda_env = os.environ.get("ZLUDA_PATH") or os.environ.get("LD_PRELOAD", "")
    if "zluda" in zluda_env.lower():
        try:
            if torch.backends.cuda.is_built():
                return torch.device("cuda")
        except Exception:  # pragma: no cover - backend probing best effort
            pass

    # 3) Apple MPS
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")

    # 4) Fallback: CPU
    return torch.device("cpu")


__all__ = ["pick_device"]
