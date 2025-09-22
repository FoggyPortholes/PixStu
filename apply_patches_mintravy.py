#!/usr/bin/env python3
"""Utility for generating the patched Mini Travy ``spriterig_v1_2.py`` module.

The script emits the bundled v1.2 implementation next to an existing
``spriterig_v1_1.py`` file while creating a ``.bak`` backup of the original.
It also provides verbose logging to aid debugging in automated environments.
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import sys

LOGGER = logging.getLogger(__name__)

V2_FILENAME = "spriterig_v1_2.py"

SPRITERIG_V1_2 = r'''#!/usr/bin/env python3
# spriterig_v1_2.py — Mini Travy reference implementation (patched)
# Changes vs v1_1:
# - GIF: single master palette, transparent index 0, disposal=2 (no flicker/trails)
# - pack: overflow guard + cross-platform globbing + per-frame pivot map
# - new "slice" command to explode sheets into frames
# - image safety bound + clearer error messages
# Requires: Pillow

from PIL import Image, ImageDraw, ImageFont
import json, os, sys, glob
from dataclasses import dataclass
from typing import Tuple, Dict, List, Optional

# Safety: avoid decompression bombs for mistakenly large images
Image.MAX_IMAGE_PIXELS = 64_000_000  # P5: safety bound

def _open_rgba(path):
    try:
        return Image.open(path).convert('RGBA')
    except Exception as e:
        raise SystemExit(f"[error] failed to open image '{path}': {e}")

@dataclass
class TileSpec:
    w: int
    h: int

@dataclass
class SheetSpec:
    rows: int
    cols: int

def _ensure_rgba(img: Image.Image) -> Image.Image:
    return img if img.mode == "RGBA" else img.convert("RGBA")

def _safe_int(v: float) -> int:
    # round-half-up to keep pivots consistent without bias
    return int(v + 0.5 if v >= 0 else v - 0.5)

def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def validate_rig(rig: Dict) -> TileSpec:
    required = ["rig_name","tile","pivot_norm","joints","bones","tpose_norm"]
    missing = [k for k in required if k not in rig]
    if missing:
        raise ValueError("Rig missing keys: %s" % missing)
    tw, th = int(rig["tile"]["w"]), int(rig["tile"]["h"])
    if tw < 8 or th < 8:
        raise ValueError("Tile size too small")
    px, py = rig["pivot_norm"]
    if not (0 <= px <= 1 and 0 <= py <= 1):
        raise ValueError("pivot_norm must be 0..1")
    joints = set(rig["joints"])
    for a, b in rig["bones"]:
        if a not in joints or b not in joints:
            raise ValueError("Unknown joint in bones: %s" % ((a,b),))
    for j, pt in rig["tpose_norm"].items():
        if j not in joints:
            raise ValueError("Unknown joint in tpose_norm: %s" % j)
        if not (0 <= pt[0] <= 1 and 0 <= pt[1] <= 1):
            raise ValueError("tpose_norm %s out of range" % j)
    return TileSpec(tw, th)

def make_template(tile: TileSpec, sheet: SheetSpec, label_prefix: Optional[str]=None,
                  grid_alpha: int=70, pivot_norm: Tuple[float,float]=(0.5,0.875)) -> Image.Image:
    W, H = tile.w * sheet.cols, tile.h * sheet.rows
    img = Image.new("RGBA", (W, H), (0,0,0,0))
    g = ImageDraw.Draw(img)
    for c in range(sheet.cols + 1):
        x = c * tile.w
        g.line([(x,0),(x,H)], fill=(255,255,255,grid_alpha), width=1)
    for r in range(sheet.rows + 1):
        y = r * tile.h
        g.line([(0,y),(W,y)], fill=(255,255,255,grid_alpha), width=1)
    if label_prefix:
        try:
            font = ImageFont.load_default()
        except Exception:
            font = None
        for r in range(sheet.rows):
            g.text((2, r*tile.h + 1), "%s%d" % (label_prefix, r), fill=(255,255,255,128), font=font)
    px, py = pivot_norm
    for r in range(sheet.rows):
        for c in range(sheet.cols):
            cx = _safe_int(c*tile.w + px*tile.w)
            cy = _safe_int(r*tile.h + py*tile.h)
            g.line([(cx-2,cy),(cx+2,cy)], fill=(255,255,0,150))
            g.line([(cx,cy-2),(cx,cy+2)], fill=(255,255,0,150))
    return img

def paste_with_pivot(sheet_img: Image.Image, frame_img: Image.Image, tile: TileSpec,
                     row: int, col: int, pivot_norm: Tuple[float,float]=(0.5,0.875),
                     frame_pivot: Optional[Tuple[int,int]]=None) -> None:
    sheet_img = _ensure_rgba(sheet_img)
    frame_img = _ensure_rgba(frame_img)
    tw, th = tile.w, tile.h
    sx = _safe_int(col*tw + pivot_norm[0]*tw)
    sy = _safe_int(row*th + pivot_norm[1]*th)
    if frame_pivot is None:
        bbox = frame_img.getbbox()
        if bbox is None:
            return
        fx = bbox[0] + (bbox[2]-bbox[0])//2
        fy = bbox[3]
    else:
        fx, fy = frame_pivot
    dx, dy = sx - fx, sy - fy
    sheet_img.alpha_composite(frame_img, (dx, dy))

def validate_sheet(sheet_img: Image.Image, tile: TileSpec, sheet: SheetSpec):
    sheet_img = _ensure_rgba(sheet_img)
    exp_w, exp_h = tile.w*sheet.cols, tile.h*sheet.rows
    ok_size = (sheet_img.width == exp_w and sheet_img.height == exp_h)
    empty_tiles = []
    non_empty_count = 0
    any_alpha = False
    for r in range(sheet.rows):
        for c in range(sheet.cols):
            box = (c*tile.w, r*tile.h, (c+1)*tile.w, (r+1)*tile.h)
            tile_img = sheet_img.crop(box)
            bbox = tile_img.getbbox()
            if bbox is None:
                empty_tiles.append((r,c))
            else:
                non_empty_count += 1
                if tile_img.mode == "RGBA":
                    alpha_band = tile_img.split()[-1]
                    extrema = alpha_band.getextrema()
                    if extrema and extrema[0] < 255:
                        any_alpha = True
    report = {
        "size_ok": ok_size,
        "expected": [exp_w, exp_h],
        "actual": [sheet_img.width, sheet_img.height],
        "tile_size": [tile.w, tile.h],
        "grid": [sheet.rows, sheet.cols],
        "empty_tiles": empty_tiles,
        "non_empty_count": non_empty_count,
        "alpha_present": any_alpha,
    }
    return (ok_size, report)

def gif_from_row(sheet_img: Image.Image, tile: TileSpec, row: int, frames: int, out_path: str,
                 frame_ms: int=90, optimize: bool=True) -> None:
    """Export an animated GIF using a single master palette and transparent index 0 (no flicker/halo)."""
    sheet_img = _ensure_rgba(sheet_img)
    # 1) Extract RGBA tiles
    tiles = []
    for c in range(frames):
        box = (c*tile.w, row*tile.h, (c+1)*tile.w, (row+1)*tile.h)
        tiles.append(sheet_img.crop(box).convert("RGBA"))

    # 2) Build montage to derive unified palette
    montage = Image.new("RGBA", (tile.w*frames, tile.h), (0,0,0,0))
    for i, t in enumerate(tiles):
        montage.alpha_composite(t, (i*tile.w, 0))

    # 3) Force palette index 0 to transparent key (magenta)
    trans_rgb = (255, 0, 255)
    mastered = montage.convert("P", palette=Image.ADAPTIVE, colors=255, dither=Image.NONE)
    pal = mastered.getpalette()[:255*3]  # keep 255 colors
    pal[0:3] = list(trans_rgb)
    if len(pal) < 256*3:
        pal += [0]*(256*3 - len(pal))
    pal_seed = Image.new("P", (1,1))
    pal_seed.putpalette(pal)

    # 4) Flatten onto key color, quantize to master palette
    pal_frames = []
    for im in tiles:
        base = Image.new("RGBA", im.size, trans_rgb + (0,))
        base.paste(im, (0,0), im.split()[-1])
        pal_frames.append(base.quantize(palette=pal_seed, dither=Image.NONE))

    # 5) Save with transparent index 0 and disposal=2
    pal_frames[0].save(out_path, save_all=True, append_images=pal_frames[1:],
                       duration=frame_ms, loop=0, disposal=2,
                       optimize=optimize, transparency=0)

def main(argv: List[str]) -> None:
    import argparse
    ap = argparse.ArgumentParser(description="Mini Travy SpriteRig toolkit v1.2")
    sub = ap.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("template", help="Generate blank grid template PNG")
    t.add_argument("--tile", type=int, nargs=2, default=[16,16])
    t.add_argument("--sheet", type=int, nargs=2, default=[4,6], help="rows cols")
    t.add_argument("--pivot", type=float, nargs=2, default=[0.5,0.875])
    t.add_argument("-o","--out", default="template_16x16_4x6.png")

    v = sub.add_parser("validate", help="Validate existing sprite sheet")
    v.add_argument("image")
    v.add_argument("--tile", type=int, nargs=2, default=[16,16])
    v.add_argument("--sheet", type=int, nargs=2, default=[4,6])

    p = sub.add_parser("pack", help="Pack individual PNG frames into a sheet with pivot alignment")
    p.add_argument("frames", nargs="+", help="List of frame PNGs (globs allowed)")
    p.add_argument("--tile", type=int, nargs=2, default=[16,16])
    p.add_argument("--sheet", type=int, nargs=2, default=[1,0], help="rows cols (cols defaults to len(frames) if 0)")
    p.add_argument("--pivot", type=float, nargs=2, default=[0.5,0.875])
    p.add_argument("--pivot-map", help="JSON mapping of frame filename -> [x,y] in frame pixels")
    p.add_argument("-o","--out", default="sheet.png")

    g = sub.add_parser("gif", help="Export an animated GIF from a sheet row")
    g.add_argument("image")
    g.add_argument("--tile", type=int, nargs=2, default=[16,16])
    g.add_argument("--row", type=int, default=0)
    g.add_argument("--frames", type=int, default=6)
    g.add_argument("--ms", type=int, default=90)
    g.add_argument("-o","--out", default="anim.gif")

    s = sub.add_parser("slice", help="Explode a sheet into frames (edit round-trip)")
    s.add_argument("image")
    s.add_argument("--tile", type=int, nargs=2, default=[16,16])
    s.add_argument("--sheet", type=int, nargs=2, required=True)
    s.add_argument("-o","--outdir", default="frames")

    args = ap.parse_args(argv)

    if args.cmd == "template":
        tile = TileSpec(*args.tile)
        sheet = SheetSpec(*args.sheet)
        img = make_template(tile, sheet, label_prefix="R", pivot_norm=tuple(args.pivot))
        img.save(args.out)
        print("Wrote %s" % args.out)

    elif args.cmd == "validate":
        tile = TileSpec(*args.tile)
        sheet = SheetSpec(*args.sheet)
        img = _open_rgba(args.image)
        ok, report = validate_sheet(img, tile, sheet)
        print(json.dumps(report, indent=2))
        if not ok:
            sys.exit(1)

    elif args.cmd == "pack":
        tile = TileSpec(*args.tile)
        rows, cols = args.sheet
        # expand globs cross-platform
        expanded: List[str] = []
        for pattern in args.frames:
            matches = glob.glob(pattern)
            if not matches:
                # still allow literal file
                if os.path.exists(pattern):
                    matches = [pattern]
            expanded.extend(matches)
        frames = expanded if expanded else args.frames

        if cols == 0:
            cols = len(frames)
        sheet = SheetSpec(rows, cols)
        # P1: overflow guard
        capacity = sheet.rows * sheet.cols
        if len(frames) > capacity:
            raise SystemExit(
                f"[error] {len(frames)} frames exceed sheet capacity {capacity} "
                f"({rows}x{cols}). Increase --sheet or reduce frames."
            )
        W, H = tile.w*sheet.cols, tile.h*sheet.rows
        out = Image.new("RGBA", (W,H), (0,0,0,0))

        pivot_map: Dict[str, List[int]] = {}
        if args.pivot_map:
            try:
                pivot_map = load_json(args.pivot_map) or {}
            except Exception as e:
                raise SystemExit(f"[error] failed to read pivot map '{args.pivot_map}': {e}")

        r = c = 0
        for fp in frames:
            fr = _open_rgba(fp)
            base = os.path.basename(fp)
            pm = pivot_map.get(base)
            frame_pivot = tuple(pm) if isinstance(pm, (list, tuple)) and len(pm) == 2 else None
            paste_with_pivot(out, fr, tile, r, c, pivot_norm=tuple(args.pivot), frame_pivot=frame_pivot)
            c += 1
            if c >= sheet.cols:
                c, r = 0, r+1
        out.save(args.out)
        print("Wrote %s" % args.out)

    elif args.cmd == "gif":
        tile = TileSpec(*args.tile)
        img = _open_rgba(args.image)
        gif_from_row(img, tile, args.row, args.frames, args.out, args.ms)
        print("Wrote %s" % args.out)

    elif args.cmd == "slice":
        tile = TileSpec(*args.tile); sheet = SheetSpec(*args.sheet)
        img = _open_rgba(args.image)
        os.makedirs(args.outdir, exist_ok=True)
        for r in range(sheet.rows):
            for c in range(sheet.cols):
                box = (c*tile.w, r*tile.h, (c+1)*tile.w, (r+1)*tile.h)
                frame = img.crop(box)
                frame.save(os.path.join(args.outdir, f"r{r}_c{c}.png"))
        print(f"Wrote frames to {args.outdir}")

if __name__ == "__main__":
    main(sys.argv[1:])
'''

def _configure_logging(verbose: bool) -> None:
    """Configure the root logger once for the script run."""

    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate the Mini Travy spriterig_v1_2.py module next to an existing "
            "spriterig_v1_1.py file while creating a .bak backup."
        )
    )
    parser.add_argument(
        "original",
        help="Path to the existing spriterig_v1_1.py file that should be patched.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose debug logging for troubleshooting.",
    )
    return parser.parse_args(argv)


def _create_backup(original: str) -> str:
    """Create a ``.bak`` backup of ``original``.

    Returns the path of the backup file and logs any errors.
    """

    backup_path = f"{original}.bak"
    try:
        shutil.copy2(original, backup_path)
        LOGGER.info("Created backup at %s", backup_path)
    except Exception as exc:  # pragma: no cover - defensive logging path
        LOGGER.warning("Could not create backup for %s: %s", original, exc)
    return backup_path


def _write_patched_module(target_dir: str) -> str:
    """Write the bundled ``spriterig_v1_2.py`` file to ``target_dir``."""

    out_path = os.path.join(target_dir, V2_FILENAME)
    LOGGER.debug("Writing patched module to %s", out_path)
    with open(out_path, "w", encoding="utf-8") as file_handle:
        file_handle.write(SPRITERIG_V1_2)
    LOGGER.info("Patched module written to %s", out_path)
    return out_path


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv if argv is not None else sys.argv[1:])
    _configure_logging(args.verbose)

    original = os.path.abspath(args.original)
    LOGGER.debug("Resolved original path to %s", original)
    if not os.path.exists(original):
        LOGGER.error("File not found: %s", original)
        return 1

    target_dir = os.path.dirname(original)
    LOGGER.debug("Target directory resolved to %s", target_dir)

    backup = _create_backup(original)
    out_path = _write_patched_module(target_dir)

    LOGGER.info("Original backed up at: %s", backup)
    LOGGER.info("Patched module available at: %s", out_path)
    return 0

if __name__ == "__main__":
    sys.exit(main())
