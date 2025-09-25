codex/rewrite-pixstu-to-version-2.1.0-bq3zn7
"""Device selector with fallback: CUDA → ZLUDA → MPS → CPU."""
from __future__ import annotations

import os
from typing import Union

try:  # optional dependency
    import torch
except ModuleNotFoundError:  # pragma: no cover - exercised without torch installed
    torch = None  # type: ignore[assignment]
=======
"""
Device selector with fallback:
CUDA (NVIDIA) → ZLUDA (Intel/AMD CUDA shim) → MPS (Apple) → CPU
"""
import os
import torch
main


def _zluda_hint() -> str:
    env = (
        os.environ.get("ZLUDA_PATH", "")
        + os.environ.get("LD_PRELOAD", "")
        + os.environ.get("DYLD_INSERT_LIBRARIES", "")
    )
    return env.lower()


codex/rewrite-pixstu-to-version-2.1.0-bq3zn7
def pick_device() -> Union["torch.device", str]:
    if torch is None:
        return "cpu"

=======
def pick_device() -> torch.device:
main
    if torch.cuda.is_available():
        return torch.device("cuda")

    if "zluda" in _zluda_hint():
        try:
            if torch.backends.cuda.is_built():
                return torch.device("cuda")
        except Exception:
            pass

codex/rewrite-pixstu-to-version-2.1.0-bq3zn7
    mps_backend = getattr(torch.backends, "mps", None)
    if mps_backend and getattr(mps_backend, "is_available", lambda: False)():
        return torch.device("mps")

    return torch.device("cpu") if torch is not None else "cpu"
=======
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")
 main

