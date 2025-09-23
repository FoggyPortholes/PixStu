from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _reload_module(tmp_path) -> "module":
    os.environ["PCS_OUTPUTS_DIR"] = str(tmp_path)
    media_exports = importlib.import_module("app.media_exports")
    importlib.reload(media_exports)
    module = importlib.import_module("app.media_masking")
    return importlib.reload(module)


def _make_test_gif(path: Path, frames: list[Image.Image]) -> None:
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=100, loop=0)


def test_estimate_bg_color_mode(tmp_path):
    module = _reload_module(tmp_path)
    img = Image.new("RGB", (4, 4), (0, 0, 0))
    img.putpixel((3, 0), (255, 0, 0))
    img.putpixel((0, 3), (255, 0, 0))

    assert module.estimate_bg_color(img) == (0, 0, 0)


def test_estimate_bg_color_average(tmp_path):
    module = _reload_module(tmp_path)
    img = Image.new("RGB", (2, 2))
    img.putpixel((0, 0), (0, 0, 0))
    img.putpixel((1, 0), (20, 20, 20))
    img.putpixel((0, 1), (40, 40, 40))
    img.putpixel((1, 1), (60, 60, 60))

    assert module.estimate_bg_color(img) == (30, 30, 30)


def test_make_alpha_transparency(tmp_path):
    module = _reload_module(tmp_path)
    img = Image.new("RGBA", (2, 1))
    img.putpixel((0, 0), (10, 10, 10, 255))
    img.putpixel((1, 0), (50, 50, 50, 255))

    result = module.make_alpha(img, (10, 10, 10), tol=0)
    assert result.getpixel((0, 0))[3] == 0
    assert result.getpixel((1, 0))[3] == 255


def test_mask_gif_creates_transparent_output(tmp_path):
    module = _reload_module(tmp_path)
    frames = [
        Image.new("RGBA", (4, 4), (255, 0, 0, 255)),
        Image.new("RGBA", (4, 4), (255, 0, 0, 255)),
    ]
    for frame in frames:
        frame.putpixel((1, 1), (0, 0, 255, 255))

    src = tmp_path / "input.gif"
    _make_test_gif(src, frames)

    out_path = module.mask_gif(str(src), tolerance=0, lock_palette=False)
    assert os.path.isfile(out_path)

    with Image.open(out_path) as result:
        first = result.convert("RGBA")
        assert first.getpixel((0, 0))[3] == 0
        assert first.getpixel((1, 1))[3] == 255


def test_mask_video_to_outputs_exports(tmp_path):
    module = _reload_module(tmp_path)
    frames = [Image.new("RGBA", (3, 3), (0, 255, 0, 255)) for _ in range(2)]
    for frame in frames:
        frame.putpixel((1, 1), (255, 0, 0, 255))

    src = tmp_path / "input.gif"
    _make_test_gif(src, frames)

    gif_path, png_dir = module.mask_video_to_outputs(
        str(src),
        tolerance=0,
        target_size=(2, 2),
        export_gif=True,
        export_png_seq=True,
    )

    assert gif_path is not None and os.path.isfile(gif_path)
    assert png_dir is not None and os.path.isdir(png_dir)

    png_path = Path(png_dir)
    png_files = sorted(png_path.glob("*.png"))
    assert png_files, "expected at least one exported frame"

    with Image.open(png_files[0]) as png:
        assert png.size == (2, 2)
        assert png.getpixel((0, 0))[3] == 0

