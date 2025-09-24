"""
Multi‑LoRA blending utilities with Diffusers runtime guards.
- Accepts list of (repo_id_or_path, scale)
- Attempts to use set_adapters when available; falls back to sequential load
- Pure utilities; UI wiring happens in Studio (future PR)
"""
from __future__ import annotations
from typing import List, Tuple


def apply_loras(pipe, loras: List[Tuple[str, float]]):
    if not loras:
        return pipe
    # Prefer official adapter APIs if present
    if hasattr(pipe, "load_lora_weights") and hasattr(pipe, "set_adapters"):
        names = []
        weights = []
        for idx, (rid, scale) in enumerate(loras):
            name = f"lora_{idx}"
            pipe.load_lora_weights(rid, adapter_name=name)
            names.append(name)
            weights.append(float(scale))
        try:
            pipe.set_adapters(names, adapter_weights=weights)
        except Exception:
            # Fallback: last-applied weight sticks
            for rid, scale in loras:
                pipe.load_lora_weights(rid)
                if hasattr(pipe, "fuse_lora"):
                    try:
                        pipe.fuse_lora()
                    except Exception:
                        pass
    else:
        # Legacy fallback: sequential load best-effort
        for rid, _ in loras:
            try:
                pipe.load_lora_weights(rid)
            except Exception:
                pass
    return pipe
