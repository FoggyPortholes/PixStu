import json
import os
from pathlib import Path

import pytest
from PIL import Image

from chargen.sprites import builder as sprite_sheet_builder


@pytest.fixture
def sprite_path(tmp_path):
    path = tmp_path / "sprite.png"
    Image.new("RGBA", (32, 48), (255, 0, 0, 255)).save(path)
    return path


@pytest.fixture(autouse=True)
def _outputs_dir(tmp_path, monkeypatch):
    out_dir = tmp_path / "outputs"
    monkeypatch.setenv("PCS_OUTPUTS_DIR", str(out_dir))
    # reload module so it picks up new env var
    from importlib import reload

    reload(sprite_sheet_builder)
    yield
    reload(sprite_sheet_builder)


def test_build_sprite_sheet_creates_files_and_mapping(sprite_path):
    preset = sprite_sheet_builder.list_presets()[0]
    sheet, mapping, zpath, frames, mapping_dict = sprite_sheet_builder.build_sprite_sheet(
        str(sprite_path),
        preset_name=preset.name,
        tile_size=64,
        padding=2,
        background="transparent",
    )

    assert Path(sheet).is_file()
    assert Path(mapping).is_file()
    assert Path(zpath).is_file()
    import zipfile

    with zipfile.ZipFile(zpath, "r") as zf:
        names = set(zf.namelist())
    assert os.path.basename(sheet) in names
    assert os.path.basename(mapping) in names
    assert frames, "frames list should not be empty"
    with open(mapping, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    assert payload["preset"] == preset.name
    assert len(payload["frames"]) == len(preset.frames)
    assert payload["columns"] == preset.columns
    assert payload["tile_size"] == 64


def test_build_sprite_sheet_missing_sprite_raises(tmp_path):
    missing = tmp_path / "missing.png"
    with pytest.raises(FileNotFoundError):
        sprite_sheet_builder.build_sprite_sheet(
            str(missing),
            preset_name=sprite_sheet_builder.list_presets()[0].name,
        )
