from __future__ import annotations

import os
from typing import Literal

from tools.device import pick_device

Device = Literal["cuda", "rocm", "mps", "cpu"]


def detect_device() -> Device:
    device = pick_device()
    device_type = getattr(device, "type", str(device))

    if device_type == "cuda":
        if os.getenv("ZLUDA") or os.getenv("ZLUDA_DEVICE") or os.getenv("ZKLUDA"):
            return "rocm"
        return "cuda"

    if device_type == "mps":
        return "mps"

    return "cpu"
