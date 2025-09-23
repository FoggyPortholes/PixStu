from __future__ import annotations

import importlib
import json
import os
import sys
from pathlib import Path
from typing import List

import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _reload_module(tmp_path) -> "module":
    os.environ["PCS_OUTPUTS_DIR"] = str(tmp_path)
    module = importlib.import_module("app.media_exports")
    return importlib.reload(module)


def _solid_frames(count: int, size: int = 8) -> List[Image.Image]:
    frames: List[Image.Image] = []
    for idx in range(count):
        color = (idx * 32 % 256, (255 - idx * 16) % 256, (idx * 64) % 256, 255)
        frames.append(Image.new("RGBA", (size, size), color))
    return frames

def test_save_gif_creates_animation(tmp_path):
    module = _reload_module(tmp_path)
    frames = _solid_frames(3)

    out_path = module.save_gif(frames, duration_ms=100, basename="test_anim")

    assert os.path.isfile(out_path)
    assert out_path.endswith(".gif")
    assert os.path.commonpath([str(tmp_path), out_path]) == str(tmp_path)

    with Image.open(out_path) as payload:
        assert payload.is_animated
        assert payload.n_frames == len(frames)
        assert payload.info["duration"] == 100


def test_save_sprite_sheet_layout(tmp_path):
    module = _reload_module(tmp_path)
    frames = _solid_frames(5, size=10)

    out_path = module.save_sprite_sheet(frames, columns=3, padding=2, basename="test_sheet")

    assert os.path.isfile(out_path)
    assert out_path.endswith(".png")
    assert os.path.commonpath([str(tmp_path), out_path]) == str(tmp_path)

    with Image.open(out_path) as sheet:
        # Expect two rows (3 columns -> 3 + 2 frames)
        assert sheet.size == (3 * 10 + 2 * 2, 2 * 10 + 1 * 2)


def test_save_functions_reject_empty_sequences(tmp_path):
    module = _reload_module(tmp_path)

    with pytest.raises(ValueError):
        module.save_gif([], basename="empty")

    with pytest.raises(ValueError):
        module.save_sprite_sheet([], basename="empty")


def test_write_frames_generates_manifest(tmp_path):
    module = _reload_module(tmp_path)
    frames = _solid_frames(3)

    out_dir = Path(module.write_frames(frames, prefix="snap"))

    assert out_dir.exists() and out_dir.is_dir()
    assert out_dir.parent.parent == tmp_path

    manifest_path = out_dir / "frames.json"
    assert manifest_path.is_file()

    data = json.loads(manifest_path.read_text())
    assert isinstance(data["timestamp"], int)
    assert data["frames"] == [f"snap_{idx:04d}.png" for idx in range(len(frames))]

    for name in data["frames"]:
        assert (out_dir / name).is_file()


def test_write_frames_respects_custom_session_dir(tmp_path):
    module = _reload_module(tmp_path)
    frames = _solid_frames(2)

    session_dir = tmp_path / "custom_session"
    frames_dir = module.write_frames(frames, session_dir=str(session_dir), prefix="frame")

    assert Path(frames_dir).parent == session_dir

