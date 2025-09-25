"""
LoRA blending scaffold — metadata only.
"""
from __future__ import annotations

from typing import Iterable, Sequence


def prepare_lora_kwargs(loras: Iterable[Sequence[object]]) -> dict:
    return {
        "lora_sets": [
            {"path": str(path), "weight": float(weight)} for path, weight in loras
        ]
    }
