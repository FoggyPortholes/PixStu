"""
LoRA blending scaffold — metadata only (hook point for real adapters).
"""
from __future__ import annotations
from typing import List, Tuple, Dict, Any


def prepare_lora_kwargs(loras: List[Tuple[str, float]] | None) -> Dict[str, Any]:
    loras = loras or []
    return {"lora_sets": [{"path": p, "weight": float(w)} for p, w in loras]}
