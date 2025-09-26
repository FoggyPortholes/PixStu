"""
Guardrails: enforce PixStu rules at prompt + image level.
- Single-character only
- No text/captions/panels/collage
- Blank background edges
"""
from __future__ import annotations
from PIL import Image
import re
import numpy as np

_FORBIDDEN = [
    r"\btext\b", r"\bcaption\b", r"\bquote\b", r"\bspeech\s*bubble\b",
    r"\bpanel(s)?\b", r"\bcomic\s*strip\b", r"\bcollage\b"
]
_MULTI = [
    r"\btwo\b", r"\bthree\b", r"\bfour\b", r"\bgroup\b",
    r"\bduo\b", r"\btrio\b", r"\bmultiple\b", r"\bcrowd\b"
]
_f = [re.compile(p, re.I) for p in _FORBIDDEN]
_m = [re.compile(p, re.I) for p in _MULTI]


def check_prompt(prompt: str):
    p = prompt or ""
    if any(r.search(p) for r in _f):
        raise ValueError("Guardrail: text/captions/panels are not allowed.")
    if any(r.search(p) for r in _m):
        raise ValueError("Guardrail: single-character only.")


def check_blank_background(img: Image.Image, alpha_threshold=20, max_coverage=0.20):
    """
    Consider background blank if edge bands are mostly transparent/light.
    We check edge coverage as a fraction of opaque pixels.
    """
    rgba = img.convert("RGBA")
    a = np.asarray(rgba.split()[-1])
    h, w = a.shape
    band = max(2, min(h, w) // 40)
    edge = np.zeros_like(a, dtype=bool)
    edge[:band, :] = True; edge[-band:, :] = True
    edge[:, :band] = True; edge[:, -band:] = True
    opaque = (a[edge] > (255 - alpha_threshold)).sum()
    coverage = opaque / edge.sum()
    if coverage > max_coverage:
        raise ValueError("Guardrail: background edges are not blank.")
