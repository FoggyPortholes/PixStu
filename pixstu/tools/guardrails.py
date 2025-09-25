"""
Guardrails: enforce PixStu rules.
"""
from __future__ import annotations

import re
from typing import Iterable

import numpy as np
from PIL import Image

_FORBIDDEN = [
    r"text",
    r"caption",
    r"quote",
    r"speech\s*bubble",
    r"panel",
    r"comic\s*strip",
    r"collage",
]
_MULTI = [
    r"two",
    r"three",
    r"four",
    r"group",
    r"duo",
    r"trio",
    r"multiple",
    r"crowd",
]


def _matches(patterns: Iterable[str], prompt: str) -> bool:
    return any(re.search(p, prompt, re.IGNORECASE) for p in patterns)


def check_prompt(prompt: str) -> None:
    if _matches(_FORBIDDEN, prompt):
        raise ValueError("Guardrail: text/captions/panels not allowed.")
    if _matches(_MULTI, prompt):
        raise ValueError("Guardrail: single-character only.")


def check_blank_background(img: Image.Image, alpha_thresh: int = 5, max_cov: float = 0.02) -> None:
    alpha = np.asarray(img.convert("RGBA").split()[-1])
    h, w = alpha.shape
    band = max(2, min(h, w) // 40)
    edge = np.zeros_like(alpha, dtype=bool)
    edge[:band, :] = True
    edge[-band:, :] = True
    edge[:, :band] = True
    edge[:, -band:] = True
    if edge.sum() == 0:
        return
    coverage = (alpha[edge] > 255 - alpha_thresh).sum() / edge.sum()
    if coverage > max_cov:
        raise ValueError("Guardrail: background not blank.")
