"""
Guardrails: enforce PixStu rules.
Relaxed: warn on failures unless PIXSTU_STRICT=1 is set.
"""
from __future__ import annotations

import os
import re
import warnings

import numpy as np
from PIL import Image

_FORBIDDEN = [r"\btext\b", r"\bcaption\b", r"\bquote\b", r"\bspeech\s*bubble\b",
              r"\bpanel(s)?\b", r"\bcomic\s*strip\b", r"\bcollage\b"]
_MULTI = [r"\btwo\b", r"\bthree\b", r"\bfour\b", r"\bgroup\b",
          r"\bduo\b", r"\btrio\b", r"\bmultiple\b", r"\bcrowd\b"]
_f = [re.compile(p, re.I) for p in _FORBIDDEN]
_m = [re.compile(p, re.I) for p in _MULTI]

STRICT = os.environ.get("PIXSTU_STRICT", "0") == "1"

def check_prompt(prompt: str):
    p = prompt or ""
    if any(r.search(p) for r in _f):
        msg = "Guardrail: text/captions/panels are not allowed."
        if STRICT: raise ValueError(msg)
        else: warnings.warn(msg)
    if any(r.search(p) for r in _m):
        msg = "Guardrail: single-character only."
        if STRICT: raise ValueError(msg)
        else: warnings.warn(msg)

def check_blank_background(img: Image.Image, alpha_threshold=20, max_coverage=0.20):
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
        msg = f"Guardrail: background edges not blank enough (coverage={coverage:.2%})"
        if STRICT: raise ValueError(msg)
        else: warnings.warn(msg)
