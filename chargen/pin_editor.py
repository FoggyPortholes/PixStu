from typing import List, Dict
from PIL import Image, ImageDraw


class Pin:
    def __init__(self, x: int, y: int, label: str, prompt: str = "", ref_img: Image.Image | None = None):
        self.x = int(x)
        self.y = int(y)
        self.label = label
        self.prompt = prompt
        self.ref_img = ref_img

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        return f"<Pin {self.label} ({self.x},{self.y})>"


def pins_to_mask(base_img: Image.Image, pins: List[Pin], radius: int = 32) -> Dict[str, Image.Image]:
    masks: Dict[str, Image.Image] = {}
    width, height = base_img.size
    for pin in pins:
        mask = Image.new("L", (width, height), 0)
        drawer = ImageDraw.Draw(mask)
        drawer.ellipse((pin.x - radius, pin.y - radius, pin.x + radius, pin.y + radius), fill=255)
        masks[pin.label] = mask
    return masks


def apply_pin_edits(base_img: Image.Image, pins: List[Pin], editor_fn):
    results: Dict[str, Image.Image] = {}
    masks = pins_to_mask(base_img, pins)
    for pin in pins:
        mask = masks.get(pin.label)
        if mask is None:
            continue
        results[pin.label] = editor_fn(base_img, mask, pin.prompt, pin.ref_img)
    return results
