from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image


def _make_dummy_pipeline(label: str):
    class _DummyPipeline:
        instances = []

        def __init__(self, base, torch_dtype=None):
            self.base = base
            self.torch_dtype = torch_dtype
            self.load_calls = []
            self.set_args = None
            self.fused = None
            self.calls = []
            self.__class__.instances.append(self)

        def load_lora_weights(self, directory, *, adapter_name=None, weight_name=None):
            self.load_calls.append(
                {
                    "directory": directory,
                    "adapter_name": adapter_name,
                    "weight_name": weight_name,
                }
            )

        def set_adapters(self, names, adapter_weights=None):
            self.set_args = {
                "names": list(names),
                "weights": list(adapter_weights) if adapter_weights is not None else None,
            }

        def fuse_lora(self, lora_scale=1.0):
            self.fused = lora_scale

        def __call__(self, **kwargs):
            self.calls.append(kwargs)
            width = kwargs.get("width", 64)
            height = kwargs.get("height", 64)
            img = Image.new("RGBA", (width, height), (10, 20, 30, 255))
            return SimpleNamespace(images=[img])

    _DummyPipeline.__name__ = f"DummyPipeline{label}"
    return _DummyPipeline


@pytest.fixture()
def generator_env(monkeypatch, tmp_path):
    import importlib

    generator = importlib.import_module("chargen.generator")
    model_setup = importlib.import_module("chargen.model_setup")

    outputs_dir = tmp_path / "outputs"
    outputs_dir.mkdir()
    models_dir = tmp_path / "models"
    lora_dir = models_dir / "lora"
    lora_dir.mkdir(parents=True)
    lora_file = lora_dir / "pixel.safetensors"
    lora_file.write_bytes(b"fake")

    torch_stub = SimpleNamespace(
        float16="float16",
        float32="float32",
        manual_seed=lambda seed: seed,
        cuda=SimpleNamespace(is_available=lambda: False),
    )

    monkeypatch.setattr(model_setup, "OUTPUTS", str(outputs_dir))
    monkeypatch.setattr(model_setup, "MODELS", str(models_dir))
    monkeypatch.setattr(model_setup, "LORAS", str(lora_dir))
    monkeypatch.setattr(model_setup, "ROOT", str(tmp_path))
    monkeypatch.setattr(generator, "model_setup", model_setup)
    monkeypatch.setattr(generator, "torch", torch_stub)

    txt2img_cls = _make_dummy_pipeline("Txt2Img")
    img2img_cls = _make_dummy_pipeline("Img2Img")
    monkeypatch.setattr(generator, "StableDiffusionXLPipeline", txt2img_cls)
    monkeypatch.setattr(generator, "StableDiffusionXLImg2ImgPipeline", img2img_cls)

    preset = {
        "base_model": "local/sdxl",
        "loras": [{"path": "models/lora/pixel.safetensors", "weight": 0.8}],
        "suggested": {"steps": 12, "guidance": 4.5},
    }

    return SimpleNamespace(
        generator_module=generator,
        preset=preset,
        outputs_dir=outputs_dir,
        lora_file=lora_file,
        txt2img_cls=txt2img_cls,
        img2img_cls=img2img_cls,
    )


def test_generate_writes_image_and_applies_lora(generator_env, tmp_path):
    gen = generator_env.generator_module.CharacterGenerator(generator_env.preset)
    out_path = Path(gen.generate("mystic swordsman", seed=7, size=128))

    assert out_path.exists()
    pipe = generator_env.txt2img_cls.instances[0]
    assert pipe.base == "local/sdxl"
    assert pipe.load_calls[0]["directory"] == str(generator_env.lora_file.parent)
    assert pipe.load_calls[0]["weight_name"] == generator_env.lora_file.name
    assert pipe.set_args == {"names": ["preset_lora_0"], "weights": [0.8]}

    call = pipe.calls[0]
    assert call["prompt"] == "mystic swordsman"
    assert call["generator"] == 7
    assert call["height"] == 128 and call["width"] == 128


def test_refine_uses_img2img_pipeline(generator_env, tmp_path):
    gen = generator_env.generator_module.CharacterGenerator(generator_env.preset)

    ref_image = tmp_path / "ref.png"
    Image.new("RGBA", (96, 156), (220, 100, 50, 255)).save(ref_image)

    out_path = Path(
        gen.refine(
            str(ref_image),
            "pose variant",
            strength=0.4,
            seed=11,
            size=192,
        )
    )

    assert out_path.exists()
    pipe = generator_env.img2img_cls.instances[0]
    assert pipe.load_calls[0]["directory"] == str(generator_env.lora_file.parent)

    call = pipe.calls[0]
    assert call["prompt"] == "pose variant"
    assert call["generator"] == 11
    assert call["num_inference_steps"] == 12
    assert call["guidance_scale"] == 4.5
    assert call["image"].size == (192, 192)
