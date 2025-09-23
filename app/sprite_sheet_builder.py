"""Utilities for building sprite sheets from a single source sprite."""
from __future__ import annotations

import json
import math
import os
import time
import zipfile
from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

from PIL import Image, ImageEnhance

ROOT = os.path.abspath(os.path.dirname(__file__))
PROJ = os.path.abspath(os.path.join(ROOT, ".."))
OUTPUTS_DIR = os.environ.get("PCS_OUTPUTS_DIR", os.path.join(PROJ, "outputs"))
os.makedirs(OUTPUTS_DIR, exist_ok=True)


@dataclass(frozen=True)
class FrameSpec:
    name: str
    ops: Sequence[str]


@dataclass(frozen=True)
class LayoutPreset:
    name: str
    description: str
    columns: int
    frames: Sequence[FrameSpec]


_PRESETS: List[LayoutPreset] = [
    LayoutPreset(
        name="Top-Down 4-dir (Idle & Walk)",
        description="12-frame sheet covering idle and walk cycles for down/left/up/right.",
        columns=4,
        frames=[
            FrameSpec("idle_down", []),
            FrameSpec("walk_down_01", ["brightness:1.05"]),
            FrameSpec("walk_down_02", ["brightness:0.92"]),
            FrameSpec("walk_down_03", ["brightness:1.1"]),
            FrameSpec("idle_left", ["flip_h"]),
            FrameSpec("walk_left_01", ["flip_h", "brightness:1.05"]),
            FrameSpec("walk_left_02", ["flip_h", "brightness:0.92"]),
            FrameSpec("walk_left_03", ["flip_h", "brightness:1.1"]),
            FrameSpec("idle_up", ["flip_v"]),
            FrameSpec("walk_up_01", ["flip_v", "brightness:1.05"]),
            FrameSpec("walk_up_02", ["flip_v", "brightness:0.92"]),
            FrameSpec("walk_up_03", ["flip_v", "brightness:1.1"]),
        ],
    ),
    LayoutPreset(
        name="Platformer (Idle/Run/Jump)",
        description="Nine slots for idle, run cycle, and jump frames (square sheet).",
        columns=3,
        frames=[
            FrameSpec("idle", []),
            FrameSpec("idle_blink", ["brightness:0.9"]),
            FrameSpec("idle_ready", ["brightness:1.08"]),
            FrameSpec("run_01", ["offset_shadow"]),
            FrameSpec("run_02", ["brightness:1.05"]),
            FrameSpec("run_03", ["brightness:0.95"]),
            FrameSpec("jump_start", ["brightness:1.1"]),
            FrameSpec("jump_mid", ["brightness:1.0", "flip_h"]),
            FrameSpec("jump_land", ["brightness:0.85"]),
        ],
    ),
]

PRESET_INDEX = {preset.name: preset for preset in _PRESETS}


def list_presets() -> Sequence[LayoutPreset]:
    return _PRESETS[:]


def _fit_sprite(sprite: Image.Image, tile_size: int) -> Image.Image:
    base = Image.new("RGBA", (tile_size, tile_size), (0, 0, 0, 0))
    fitted = Image.new("RGBA", sprite.size, (0, 0, 0, 0))
    fitted.paste(sprite.convert("RGBA"), (0, 0), sprite.convert("RGBA"))
    if sprite.size == (tile_size, tile_size):
        base.paste(fitted, (0, 0), fitted)
        return base
    ratio = min(tile_size / sprite.width, tile_size / sprite.height)
    new_size = (max(1, int(sprite.width * ratio)), max(1, int(sprite.height * ratio)))
    resized = sprite.resize(new_size, Image.NEAREST).convert("RGBA")
    offset = ((tile_size - new_size[0]) // 2, (tile_size - new_size[1]) // 2)
    base.paste(resized, offset, resized)
    return base


def _apply_ops(img: Image.Image, ops: Iterable[str]) -> Image.Image:
    frame = img.convert("RGBA")
    for op in ops:
        if op == "flip_h":
            frame = frame.transpose(Image.FLIP_LEFT_RIGHT)
        elif op == "flip_v":
            frame = frame.transpose(Image.FLIP_TOP_BOTTOM)
        elif op == "flip_both":
            frame = frame.transpose(Image.FLIP_LEFT_RIGHT).transpose(Image.FLIP_TOP_BOTTOM)
        elif op.startswith("brightness:"):
            try:
                factor = float(op.split(":", 1)[1])
            except ValueError:
                factor = 1.0
            frame = ImageEnhance.Brightness(frame).enhance(max(0.1, factor))
        elif op == "offset_shadow":
            shadow = Image.new("RGBA", frame.size, (0, 0, 0, 0))
            shadow_offset = (max(1, frame.width // 12), max(1, frame.height // 12))
            shadow_sprite = frame.copy()
            alpha = shadow_sprite.split()[-1]
            shadow_sprite = Image.new("RGBA", frame.size, (0, 0, 0, 255))
            shadow_sprite.putalpha(alpha)
            shadow.paste(shadow_sprite, shadow_offset, shadow_sprite)
            combined = Image.alpha_composite(shadow, frame)
            frame = combined
        elif op.startswith("opacity:"):
            try:
                pct = float(op.split(":", 1)[1])
            except ValueError:
                pct = 1.0
            pct = max(0.0, min(1.0, pct))
            r, g, b, a = frame.split()
            a = a.point(lambda px: int(px * pct))
            frame.putalpha(a)
        # unknown ops are ignored intentionally
    return frame


def _parse_background(color: str | None) -> Tuple[int, int, int, int]:
    if not color or color.lower() == "transparent":
        return (0, 0, 0, 0)
    color = color.strip()
    if color.startswith("#"):
        color = color[1:]
    if len(color) in (6, 8):
        r = int(color[0:2], 16)
        g = int(color[2:4], 16)
        b = int(color[4:6], 16)
        a = 255 if len(color) == 6 else int(color[6:8], 16)
        return (r, g, b, a)
    NAMED = {
        "black": (0, 0, 0, 255),
        "white": (255, 255, 255, 255),
        "gray": (96, 96, 96, 255),
        "transparent": (0, 0, 0, 0),
    }
    return NAMED.get(color.lower(), (0, 0, 0, 0))


def _timestamp(prefix: str, suffix: str) -> str:
    return f"{prefix}_{int(time.time() * 1000)}{suffix}"


def build_sprite_sheet(
    sprite_path: str,
    *,
    preset_name: str,
    tile_size: int = 128,
    padding: int = 4,
    background: str | None = "transparent",
) -> Tuple[str, str, str, List[Image.Image], dict]:
    if preset_name not in PRESET_INDEX:
        raise ValueError(f"Unknown preset: {preset_name}")
    preset = PRESET_INDEX[preset_name]
    if tile_size <= 0:
        raise ValueError("tile_size must be positive")
    if padding < 0:
        raise ValueError("padding must be >= 0")
    if not os.path.isfile(sprite_path):
        raise FileNotFoundError(sprite_path)

    sprite = Image.open(sprite_path).convert("RGBA")
    base_tile = _fit_sprite(sprite, tile_size)

    frames: List[Image.Image] = []
    frame_metadata: List[dict] = []
    for idx, spec in enumerate(preset.frames):
        transformed = _apply_ops(base_tile, spec.ops)
        frames.append(transformed)
        frame_metadata.append(
            {
                "name": spec.name,
                "index": idx,
                "ops": list(spec.ops),
            }
        )

    columns = max(1, preset.columns)
    rows = math.ceil(len(frames) / columns)
    bg_rgba = _parse_background(background)

    sheet_width = columns * tile_size + padding * (columns - 1)
    sheet_height = rows * tile_size + padding * (rows - 1)
    sheet = Image.new("RGBA", (sheet_width, sheet_height), bg_rgba)

    for idx, frame in enumerate(frames):
        row = idx // columns
        col = idx % columns
        x = col * (tile_size + padding)
        y = row * (tile_size + padding)
        sheet.paste(frame, (x, y), frame)
        frame_metadata[idx].update(
            {
                "row": row,
                "column": col,
                "x": x,
                "y": y,
                "width": frame.width,
                "height": frame.height,
            }
        )

    sheet_path = os.path.join(OUTPUTS_DIR, _timestamp("sprite_sheet", ".png"))
    sheet.save(sheet_path)

    mapping = {
        "preset": preset.name,
        "description": preset.description,
        "tile_size": tile_size,
        "padding": padding,
        "columns": columns,
        "rows": rows,
        "sheet_path": sheet_path,
        "frames": frame_metadata,
    }
    mapping_path = os.path.join(OUTPUTS_DIR, _timestamp("sprite_sheet", ".json"))
    with open(mapping_path, "w", encoding="utf-8") as handle:
        json.dump(mapping, handle, indent=2)

    zip_path = os.path.join(OUTPUTS_DIR, _timestamp("sprite_sheet_pack", ".zip"))
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(sheet_path, arcname=os.path.basename(sheet_path))
        zf.write(mapping_path, arcname=os.path.basename(mapping_path))

    return sheet_path, mapping_path, zip_path, frames, mapping


__all__ = ["list_presets", "build_sprite_sheet", "LayoutPreset", "FrameSpec"]
