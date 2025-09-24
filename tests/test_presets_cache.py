import json
import os
import time

import pytest

import chargen.presets as presets


@pytest.fixture(autouse=True)
def _reset_cache():
    presets.load_presets(force_reload=True)
    yield
    presets.load_presets(force_reload=True)


def _write_presets(path, payload):
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle)


def test_get_preset_returns_defensive_copy(tmp_path, monkeypatch):
    preset_path = tmp_path / "presets.json"
    _write_presets(
        preset_path,
        {
            "presets": [
                {
                    "name": "Alpha",
                    "model": "foo",
                    "loras": [],
                }
            ]
        },
    )
    monkeypatch.setattr(presets, "PRESET_FILE", str(preset_path))
    presets.load_presets(force_reload=True)

    first = presets.get_preset("Alpha")
    assert first is not None
    first["model"] = "bar"

    second = presets.get_preset("Alpha")
    assert second is not None
    assert second["model"] == "foo"


def test_load_presets_uses_cache_until_file_changes(tmp_path, monkeypatch):
    preset_path = tmp_path / "presets.json"
    payload = {"presets": [{"name": "Alpha", "model": "foo"}]}
    _write_presets(preset_path, payload)
    monkeypatch.setattr(presets, "PRESET_FILE", str(preset_path))
    presets.load_presets(force_reload=True)

    loaded = presets.load_presets()
    assert loaded == {"Alpha": {"name": "Alpha", "model": "foo"}}

    original_reader = presets._read_presets_from_disk

    def _boom(*_args, **_kwargs):
        raise AssertionError("Cache should prevent disk reload")

    monkeypatch.setattr(presets, "_read_presets_from_disk", _boom)
    cached = presets.load_presets()
    assert cached == loaded

    monkeypatch.setattr(presets, "_read_presets_from_disk", original_reader)
    payload["presets"].append({"name": "Beta", "model": "bar"})
    _write_presets(preset_path, payload)
    new_time = time.time() + 5
    os.utime(preset_path, (new_time, new_time))

    refreshed = presets.load_presets()
    assert set(refreshed) == {"Alpha", "Beta"}
