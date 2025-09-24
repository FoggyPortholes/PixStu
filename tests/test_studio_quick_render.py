import os

import gradio as gr
import pytest

from chargen import studio


class DummyGenerator:
    def __init__(self, preset):
        self.preset = preset

    def generate(self, prompt, seed):  # pragma: no cover - exercised indirectly
        return {"prompt": prompt, "seed": seed, "preset": self.preset}


@pytest.fixture(autouse=True)
def _patch_generator(monkeypatch):
    monkeypatch.setattr(studio, "BulletProofGenerator", DummyGenerator)


def _preset_with_loras(existing_path: str, missing_path: str) -> dict:
    return {
        "loras": [
            {
                "display_path": existing_path,
                "resolved_path": existing_path,
                "path": existing_path,
            },
            {
                "display_path": missing_path,
                "resolved_path": missing_path,
                "path": missing_path,
            },
        ]
    }


def test_quick_render_ignores_unselected_missing_loras(tmp_path, monkeypatch):
    available = tmp_path / "loras" / "available.safetensors"
    available.parent.mkdir(parents=True, exist_ok=True)
    available.write_text("stub")

    missing = tmp_path / "loras" / "missing.safetensors"

    preset = _preset_with_loras(str(available), str(missing))
    monkeypatch.setattr(studio, "get_preset", lambda name: preset)

    result = studio._quick_render("preset", str(available), 0.75)

    assert result["prompt"] == "LoRA quick preview"
    assert result["preset"] == preset


def test_quick_render_errors_for_selected_missing_lora(tmp_path, monkeypatch):
    selected_missing = tmp_path / "loras" / "missing.safetensors"
    preset = _preset_with_loras(str(tmp_path / "loras" / "available.safetensors"), str(selected_missing))
    monkeypatch.setattr(studio, "get_preset", lambda name: preset)

    with pytest.raises(gr.Error) as excinfo:
        studio._quick_render("preset", str(selected_missing), 1.0)

    message = str(excinfo.value)
    # The error should reference only the missing LoRA that was selected.
    assert "missing.safetensors" in message
    assert os.fspath(selected_missing) in message
