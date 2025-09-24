import pytest

from chargen import lora_blend


def test_save_and_load_roundtrip(tmp_path, monkeypatch):
    storage = tmp_path / "lora_sets.json"
    monkeypatch.setattr(lora_blend, "BLEND_FILE", storage)

    rows = [["loras/a.safetensors", 0.5], ["loras/b.safetensors", "0.25"]]
    lora_blend.save_set("combo", rows)

    loaded = lora_blend.blend_to_rows(lora_blend.get_set("combo"))
    assert loaded == [["loras/a.safetensors", 0.5], ["loras/b.safetensors", 0.25]]


def test_apply_blend_updates_weights(tmp_path, monkeypatch):
    storage = tmp_path / "lora_sets.json"
    monkeypatch.setattr(lora_blend, "BLEND_FILE", storage)

    preset = {
        "loras": [
            {"path": "loras/a.safetensors", "weight": 0.1},
            {"path": "loras/c.safetensors", "weight": 1.0},
        ]
    }

    rows = [["loras/a.safetensors", 0.7], ["loras/b.safetensors", 0.3]]
    lora_blend.apply_blend(preset, rows)

    weights = {entry["path"]: entry["weight"] for entry in preset["loras"]}
    assert weights["loras/a.safetensors"] == 0.7
    assert weights["loras/b.safetensors"] == 0.3
    assert "loras/c.safetensors" in weights


def test_guardrails_on_blend_size(tmp_path, monkeypatch):
    storage = tmp_path / "lora_sets.json"
    monkeypatch.setattr(lora_blend, "BLEND_FILE", storage)

    rows = [[f"loras/{idx}.safetensors", 0.1] for idx in range(lora_blend.MAX_LORAS + 1)]

    with pytest.raises(ValueError):
        lora_blend.save_set("too-many", rows)
