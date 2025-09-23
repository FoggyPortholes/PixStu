import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PIL import Image


def _reload_gif_creator():
    from importlib import reload

    import app.media_exports as media_exports
    import app.gif_creator as gif_creator

    reload(media_exports)
    return reload(gif_creator)


@pytest.fixture
def gif_creator(monkeypatch, tmp_path):
    monkeypatch.setenv("PCS_OUTPUTS_DIR", str(tmp_path))
    return _reload_gif_creator()


def test_make_gif_from_sprite_can_export_preview(gif_creator, tmp_path):
    sprite_path = tmp_path / "sprite.png"
    Image.new("RGBA", (8, 8), (255, 0, 0, 255)).save(sprite_path)

    gif_path, sheet_path, preview_dir = gif_creator.make_gif_from_sprite(
        sprite_path=str(sprite_path),
        preset_name="default",
        prompt=None,
        frames=3,
        frame_size=8,
        duration_ms=50,
        seed=None,
        seed_jitter=0,
        motion_mode="none",
        img_strength=1.0,
        lock_palette=False,
        export_sheet=False,
        export_preview=True,
    )

    assert gif_path.endswith(".gif")
    assert sheet_path is None
    assert preview_dir is not None

    manifest_path = Path(preview_dir) / "frames.json"
    assert manifest_path.exists()

    payload = json.loads(manifest_path.read_text("utf-8"))
    assert payload["frames"]
    for filename in payload["frames"]:
        frame_path = Path(preview_dir) / filename
        assert frame_path.exists()
