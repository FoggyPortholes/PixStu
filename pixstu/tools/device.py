"""
Device selector with fallback:
CUDA (NVIDIA) → ZLUDA (Intel/AMD CUDA shim) → MPS (Apple) → CPU
"""
import os
import torch


def _zluda_hint() -> str:
    env = (
        os.environ.get("ZLUDA_PATH", "")
        + os.environ.get("LD_PRELOAD", "")
        + os.environ.get("DYLD_INSERT_LIBRARIES", "")
    )
    return env.lower()


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")

    if "zluda" in _zluda_hint():
        try:
            if torch.backends.cuda.is_built():
                return torch.device("cuda")
        except Exception:
            pass

    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")

