#!/usr/bin/env python3
"""Mini Travy SpriteRig toolkit v1.1 with quality-of-life improvements."""

import glob
import json
import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from PIL import Image, ImageDraw, ImageFont

# P5: guard against decompression bombs from untrusted images
Image.MAX_IMAGE_PIXELS = 64_000_000


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


def _safe_int(value: float) -> int:
    """Round-half-up helper for consistent pivot placement (P7)."""
    return int(value + 0.5) if value >= 0 else int(value - 0.5)


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _open_rgba(path: str) -> Image.Image:
    try:
        return Image.open(path).convert("RGBA")
    except Exception as exc:
        raise SystemExit(f"[error] failed to open image '{path}': {exc}")


def validate_rig(rig: Dict) -> TileSpec:
    required = ["rig_name", "tile", "pivot_norm", "joints", "bones", "tpose_norm"]
    missing = [key for key in required if key not in rig]
    if missing:
        raise ValueError(f"Rig missing keys: {missing}")

    tw = int(rig["tile"]["w"])
    th = int(rig["tile"]["h"])
    if tw < 8 or th < 8:
        raise ValueError("Tile size too small")

    px, py = rig["pivot_norm"]
    if not (0 <= px <= 1 and 0 <= py <= 1):
        raise ValueError("pivot_norm must be in range 0..1")

    joints = set(rig["joints"])
    for a, b in rig["bones"]:
        if a not in joints or b not in joints:
            raise ValueError(f"Unknown joint in bones: {(a, b)}")

    for joint, pt in rig["tpose_norm"].items():
        if joint not in joints:
            raise ValueError(f"Unknown joint in tpose_norm: {joint}")
        if not (0 <= pt[0] <= 1 and 0 <= pt[1] <= 1):
            raise ValueError(f"tpose_norm {joint} out of range")

    return TileSpec(tw, th)


def make_template(
    tile: TileSpec,
    sheet: SheetSpec,
    label_prefix: Optional[str] = None,
    grid_alpha: int = 70,
    pivot_norm: Tuple[float, float] = (0.5, 0.875),
) -> Image.Image:
    W, H = tile.w * sheet.cols, tile.h * sheet.rows
    img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    for c in range(sheet.cols + 1):
        x = _safe_int(c * tile.w)
        draw.line([(x, 0), (x, H)], fill=(255, 255, 255, grid_alpha), width=1)
    for r in range(sheet.rows + 1):
        y = _safe_int(r * tile.h)
        draw.line([(0, y), (W, y)], fill=(255, 255, 255, grid_alpha), width=1)

    if label_prefix:
        try:
            font = ImageFont.load_default()
        except Exception:
            font = None
        for r in range(sheet.rows):
            draw.text((2, r * tile.h + 1), f"{label_prefix}{r}", fill=(255, 255, 255, 128), font=font)

    px, py = pivot_norm
    for r in range(sheet.rows):
        for c in range(sheet.cols):
            cx = _safe_int(c * tile.w + px * tile.w)
            cy = _safe_int(r * tile.h + py * tile.h)
            draw.line([(cx - 2, cy), (cx + 2, cy)], fill=(255, 255, 0, 150))
            draw.line([(cx, cy - 2), (cx, cy + 2)], fill=(255, 255, 0, 150))

    return img


def paste_with_pivot(
    sheet_img: Image.Image,
    frame_img: Image.Image,
    tile: TileSpec,
    row: int,
    col: int,
    pivot_norm: Tuple[float, float] = (0.5, 0.875),
) -> None:
    sheet_img = _ensure_rgba(sheet_img)
    frame_img = _ensure_rgba(frame_img)
    tw, th = tile.w, tile.h

    sx = _safe_int(col * tw + pivot_norm[0] * tw)
    sy = _safe_int(row * th + pivot_norm[1] * th)

    bbox = frame_img.getbbox()
    if bbox is None:
        return
    fx = _safe_int(bbox[0] + (bbox[2] - bbox[0]) / 2.0)
    fy = _safe_int(bbox[3])

    dx, dy = sx - fx, sy - fy
    sheet_img.alpha_composite(frame_img, (dx, dy))


def validate_sheet(sheet_img: Image.Image, tile: TileSpec, sheet: SheetSpec):
    sheet_img = _ensure_rgba(sheet_img)
    exp_w, exp_h = tile.w * sheet.cols, tile.h * sheet.rows
    ok_size = sheet_img.width == exp_w and sheet_img.height == exp_h

    empty_tiles: List[Tuple[int, int]] = []
    non_empty_count = 0
    alpha_present = False

    for r in range(sheet.rows):
        for c in range(sheet.cols):
            box = (c * tile.w, r * tile.h, (c + 1) * tile.w, (r + 1) * tile.h)
            tile_img = sheet_img.crop(box)
            bbox = tile_img.getbbox()
            if bbox is None:
                empty_tiles.append((r, c))
            else:
                non_empty_count += 1
                if tile_img.mode == "RGBA":
                    extrema = tile_img.split()[-1].getextrema()
                    if extrema and extrema[0] < 255:
                        alpha_present = True

    report = {
        "size_ok": ok_size,
        "expected": [exp_w, exp_h],
        "actual": [sheet_img.width, sheet_img.height],
        "tile_size": [tile.w, tile.h],
        "grid": [sheet.rows, sheet.cols],
        "empty_tiles": empty_tiles,
        "non_empty_count": non_empty_count,
        "alpha_present": alpha_present,
    }
    return ok_size, report


def gif_from_row(
    sheet_img: Image.Image,
    tile: TileSpec,
    row: int,
    frames: int,
    out_path: str,
    frame_ms: int = 90,
) -> None:
    sheet_img = _ensure_rgba(sheet_img)
    tiles: List[Image.Image] = []
    for c in range(frames):
        box = (c * tile.w, row * tile.h, (c + 1) * tile.w, (row + 1) * tile.h)
        tiles.append(sheet_img.crop(box).convert("RGBA"))

    montage = Image.new("RGBA", (tile.w * frames, tile.h), (0, 0, 0, 0))
    for i, tile_img in enumerate(tiles):
        montage.alpha_composite(tile_img, (i * tile.w, 0))

    trans_rgb = (255, 0, 255)
    mastered = montage.convert("P", palette=Image.ADAPTIVE, colors=255, dither=Image.NONE)
    palette = mastered.getpalette()[: 255 * 3]
    palette[0:3] = list(trans_rgb)
    if len(palette) < 256 * 3:
        palette += [0] * (256 * 3 - len(palette))
    pal_seed = Image.new("P", (1, 1))
    pal_seed.putpalette(palette)

    pal_frames: List[Image.Image] = []
    for tile_img in tiles:
        base = Image.new("RGBA", tile_img.size, trans_rgb + (0,))
        base.paste(tile_img, (0, 0), tile_img.split()[-1])
        pal_frames.append(base.quantize(palette=pal_seed, dither=Image.NONE))

    pal_frames[0].save(
        out_path,
        save_all=True,
        append_images=pal_frames[1:],
        duration=frame_ms,
        loop=0,
        disposal=2,
        optimize=True,
        transparency=0,
    )


def main(argv: List[str]) -> None:
    import argparse

    ap = argparse.ArgumentParser(description="Mini Travy SpriteRig toolkit v1.1")
    sub = ap.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("template", help="Generate blank grid template PNG")
    t.add_argument("--tile", type=int, nargs=2, default=[16, 16])
    t.add_argument("--sheet", type=int, nargs=2, default=[4, 6], help="rows cols")
    t.add_argument("--pivot", type=float, nargs=2, default=[0.5, 0.875])
    t.add_argument("-o", "--out", default="template_16x16_4x6.png")

    v = sub.add_parser("validate", help="Validate existing sprite sheet")
    v.add_argument("image")
    v.add_argument("--tile", type=int, nargs=2, default=[16, 16])
    v.add_argument("--sheet", type=int, nargs=2, default=[4, 6])

    p = sub.add_parser("pack", help="Pack individual PNG frames into a sheet")
    p.add_argument("frames", nargs="+", help="List of frame PNGs (globs allowed)")
    p.add_argument("--tile", type=int, nargs=2, default=[16, 16])
    p.add_argument("--sheet", type=int, nargs=2, default=[1, 0], help="rows cols (cols defaults to len(frames) if 0)")
    p.add_argument("--pivot", type=float, nargs=2, default=[0.5, 0.875])
    p.add_argument("-o", "--out", default="sheet.png")

    g = sub.add_parser("gif", help="Export an animated GIF from a sheet row")
    g.add_argument("image")
    g.add_argument("--tile", type=int, nargs=2, default=[16, 16])
    g.add_argument("--row", type=int, default=0)
    g.add_argument("--frames", type=int, default=6)
    g.add_argument("--ms", type=int, default=90)
    g.add_argument("-o", "--out", default="anim.gif")

    args = ap.parse_args(argv)

    if args.cmd == "template":
        tile = TileSpec(*args.tile)
        sheet = SheetSpec(*args.sheet)
        img = make_template(tile, sheet, label_prefix="R", pivot_norm=tuple(args.pivot))
        img.save(args.out)
        print(f"Wrote {args.out}")

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
        frames: List[str] = []
        for pattern in args.frames:
            matches = glob.glob(pattern)
            if not matches and os.path.exists(pattern):
                matches = [pattern]
            frames.extend(sorted(matches))
        if not frames:
            frames = args.frames
        if cols == 0:
            cols = len(frames)
        sheet = SheetSpec(rows, cols)
        W, H = tile.w * sheet.cols, tile.h * sheet.rows
        out = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        r = c = 0
        for fp in frames:
            fr = _open_rgba(fp)
            paste_with_pivot(out, fr, tile, r, c, pivot_norm=tuple(args.pivot))
            c += 1
            if c >= sheet.cols:
                c, r = 0, r + 1
        out.save(args.out)
        print(f"Wrote {args.out}")

    elif args.cmd == "gif":
        tile = TileSpec(*args.tile)
        img = _open_rgba(args.image)
        gif_from_row(img, tile, args.row, args.frames, args.out, args.ms)
        print(f"Wrote {args.out}")


if __name__ == "__main__":
    main(sys.argv[1:])
