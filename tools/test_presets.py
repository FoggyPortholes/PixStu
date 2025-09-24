import json
import os
import time

import torch
from diffusers import StableDiffusionXLPipeline

PRESET_FILE = os.path.join("configs", "curated_models.json")
SAMPLE_DIR = os.path.join("docs", "preset_samples")


def load_presets():
    with open(PRESET_FILE, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    if isinstance(data, dict):
        presets = data.get("presets", [])
    else:
        presets = data
    return presets


def run_one(preset: dict, prompt: str):
    model_id = preset.get("model", "stabilityai/stable-diffusion-xl-base-1.0")
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    pipe = StableDiffusionXLPipeline.from_pretrained(model_id, torch_dtype=dtype)
    pipe = pipe.to("cuda" if torch.cuda.is_available() else "cpu")
    image = pipe(
        prompt=f"{prompt}, " + ", ".join(preset.get("positive", [])),
        negative_prompt=", ".join(preset.get("negative", [])),
        num_inference_steps=preset.get("steps", 30),
        guidance_scale=preset.get("cfg", 7.0),
    ).images[0]
    folder = os.path.join(SAMPLE_DIR, preset["name"].replace(" ", "_"))
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, f"sample_{int(time.time())}.png")
    image.save(path)
    print("[OK]", path)


def main():
    presets = load_presets()
    base_prompt = "heroic character portrait"
    for preset in presets:
        run_one(preset, base_prompt)


if __name__ == "__main__":
    main()
