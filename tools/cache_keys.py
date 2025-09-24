"""Deterministic cache keys for PixStu components."""

from __future__ import annotations

import hashlib
import json
from io import BytesIO
from typing import Any, Iterable, Mapping, Optional

from PIL import Image


def _digest(obj: Any) -> str:
    if obj is None:
        return "null"
    if isinstance(obj, (str, int, float, bool)):
        return json.dumps(obj, sort_keys=True)
    if isinstance(obj, Mapping):
        items = sorted((str(k), _digest(v)) for k, v in obj.items())
        return json.dumps(items, sort_keys=True)
    if isinstance(obj, Iterable) and not isinstance(obj, (bytes, bytearray, Image.Image)):
        return json.dumps([_digest(item) for item in obj], sort_keys=True)
    if isinstance(obj, (bytes, bytearray)):
        return hashlib.sha256(obj).hexdigest()
    if isinstance(obj, Image.Image):
        buffer = BytesIO()
        obj.save(buffer, format="PNG")
        return hashlib.sha256(buffer.getvalue()).hexdigest()
    return hashlib.sha256(str(obj).encode("utf-8")).hexdigest()


def inpaint_key(
    base_img: Image.Image,
    mask: Image.Image,
    prompt: str,
    *,
    guidance_scale: float,
    steps: int,
    ref_img: Optional[Image.Image] = None,
) -> str:
    payload = {
        "base": _digest(base_img),
        "mask": _digest(mask),
        "prompt": prompt,
        "guidance": round(float(guidance_scale), 4),
        "steps": int(steps),
        "ref": _digest(ref_img) if ref_img is not None else None,
    }
    serialised = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha256(serialised).hexdigest()


__all__ = ["inpaint_key"]
