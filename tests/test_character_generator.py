import types

import pytest
from PIL import Image

import chargen.generator as generator


class DummyPipeline:
    def __init__(self, *_args, **_kwargs):
        self.loaded_loras = []
        self.device = None

    def to(self, device):
        self.device = device
        return self

    def load_lora_weights(self, path, weight=1.0):
        self.loaded_loras.append((path, weight))

    def __call__(self, *_, **__):
        return types.SimpleNamespace(images=[Image.new("RGB", (16, 16), (10, 20, 30))])


@pytest.fixture(autouse=True)
def patch_pipeline(monkeypatch):
    monkeypatch.setattr(generator, "StableDiffusionXLPipeline", DummyPipeline)
    monkeypatch.setattr(generator, "StableDiffusionXLControlNetPipeline", DummyPipeline)
    monkeypatch.setattr(generator, "torch", types.SimpleNamespace(
        cuda=types.SimpleNamespace(is_available=lambda: False),
        backends=types.SimpleNamespace(mps=types.SimpleNamespace(is_available=lambda: False)),
        Generator=lambda device=None: types.SimpleNamespace(manual_seed=lambda seed: seed),
    ))
    yield


def test_generate_returns_image(tmp_path, monkeypatch):
    preset = {
        "model": "dummy/model",
        "positive": ["sharp"],
        "negative": ["blur"],
        "loras": [{"path": "loras/example.safetensors", "weight": 0.5}],
        "steps": 5,
        "cfg": 4.5,
        "resolution": 128,
    }
    gen = generator.BulletProofGenerator(preset)
    image = gen.generate("sample prompt", seed=7)
    assert isinstance(image, Image.Image)
    assert image.size == (16, 16)
    assert gen.pipe.loaded_loras[0][0] == "loras/example.safetensors"
