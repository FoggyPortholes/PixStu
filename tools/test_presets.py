import json
import os
import time

import torch
from diffusers import StableDiffusionXLPipeline

CFG = ("configs/curated_models.json", "docs/preset_samples")


def load():
    with open(CFG[0], "r", encoding="utf-8") as f:
        return json.load(f)


def _coerce_list(value):
    if not value:
        return []
    if isinstance(value, str):
        return [part.strip() for part in value.split(",") if part.strip()]
    if isinstance(value, (list, tuple, set)):
        result = []
        for entry in value:
            result.extend(_coerce_list(entry))
        return result
    return [str(value)]


def run_one(preset, prompt):
    model = preset.get("model", "stabilityai/stable-diffusion-xl-base-1.0")
    pipe = StableDiffusionXLPipeline.from_pretrained(
        model,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    ).to("cuda" if torch.cuda.is_available() else "cpu")

    suggested = preset.get("suggested", {})
    steps = int(preset.get("steps", suggested.get("steps", 30)))
    cfg = float(preset.get("cfg", suggested.get("guidance", 7.0)))
    positives = _coerce_list(preset.get("positive"))
    if not positives:
        positives = _coerce_list(preset.get("style_prompt"))
    negatives = _coerce_list(preset.get("negative"))
    if not negatives:
        negatives = _coerce_list(preset.get("negative_prompt"))

    img = pipe(
        prompt=prompt + (", " + ", ".join(positives) if positives else ""),
        negative_prompt=", ".join(negatives),
        num_inference_steps=steps,
        guidance_scale=cfg,
    ).images[0]
    outdir = os.path.join(CFG[1], preset.get("name", "preset").replace(" ", "_"))
    os.makedirs(outdir, exist_ok=True)
    p = os.path.join(outdir, f"sample_{int(time.time())}.png")
    img.save(p)
    print("[OK]", p)


if __name__ == "__main__":
    data = load()
    prompts = ["heroic character portrait", "action pose", "wizard with staff"]
    for preset in data:
        for pr in prompts:
            run_one(preset, pr)
