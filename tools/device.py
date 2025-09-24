"""Hardware accelerator selection helpers."""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Any

try:  # pragma: no cover - optional dependency in some environments
    import torch
except Exception:  # pragma: no cover - torch may not be installed yet
    torch = None  # type: ignore


class _CPUDevice:
    """Lightweight torch.device stand-in when torch is unavailable."""

    type = "cpu"
    index = None

    def __str__(self) -> str:  # pragma: no cover - simple data holder
        return "cpu"

    def __repr__(self) -> str:  # pragma: no cover - simple data holder
        return "device(type='cpu')"


_CPU_FALLBACK = _CPUDevice()


def _has_env_flag(*names: str) -> bool:
    return any(os.getenv(name) for name in names)


@lru_cache(maxsize=None)
def pick_device() -> Any:
    """Select the best available torch device for model execution."""

    if torch is None:
        return _CPU_FALLBACK

    if torch.cuda.is_available():
        return torch.device("cuda")

    # Vendor-provided CUDA compatibility layers
    if _has_env_flag("ZLUDA_PATH", "ZKLUDA_PATH", "ZLUDA", "ZLUDA_DEVICE", "ZKLUDA"):
        return torch.device("cuda")

    try:  # pragma: no cover - optional acceleration
        import zluda  # type: ignore

        return torch.device("cuda")
    except Exception:  # pragma: no cover - dependency optional
        pass

    try:  # pragma: no cover - optional acceleration
        import zkluda  # type: ignore

        return torch.device("cuda")
    except Exception:  # pragma: no cover - dependency optional
        pass

    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


__all__ = ["pick_device"]
