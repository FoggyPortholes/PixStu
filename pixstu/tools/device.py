"""
Device selector with fallback:
CUDA (NVIDIA) → ZLUDA (Intel/AMD CUDA shim) → MPS (Apple) → CPU
"""
import os

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - optional install path
    torch = None  # type: ignore[assignment]


def pick_device():
    if torch is None:
        return "cpu"
    # 1) Native CUDA
    if torch.cuda.is_available():
        return torch.device("cuda")
    # 2) ZLUDA probe via env hints
    hint = (os.environ.get("ZLUDA_PATH", "")
            + os.environ.get("LD_PRELOAD", "")
            + os.environ.get("DYLD_INSERT_LIBRARIES", "")).lower()
    if "zluda" in hint:
        try:
            if torch.backends.cuda.is_built():
                return torch.device("cuda")
        except Exception:
            pass
    # 3) Apple MPS
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    # 4) CPU
    return torch.device("cpu")
