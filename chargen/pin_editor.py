from typing import List, Dict
from PIL import Image, ImageDraw

class Pin:
    def __init__(self, x: int, y: int, label: str, prompt: str = "", ref_img: Image.Image = None):
        self.x, self.y, self.label, self.prompt, self.ref_img = x, y, label, prompt, ref_img

    def __repr__(self):
        return f"<Pin {self.label} ({self.x},{self.y})>"


def pins_to_mask(base_img: Image.Image, pins: List[Pin], radius: int = 32) -> Dict[str, Image.Image]:
    masks = {}
    w, h = base_img.size
    for p in pins:
        m = Image.new("L", (w, h), 0)
        d = ImageDraw.Draw(m)
        d.ellipse((p.x - radius, p.y - radius, p.x + radius, p.y + radius), fill=255)
        masks[p.label] = m
    return masks


def apply_pin_edits(base_img: Image.Image, pins: List[Pin], editor_fn) -> Dict[str, Image.Image]:
    out = {}
    masks = pins_to_mask(base_img, pins)
    for p in pins:
        out[p.label] = editor_fn(base_img, masks[p.label], p.prompt, p.ref_img)
    return out
