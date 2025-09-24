import os, json, torch, time
from diffusers import StableDiffusionXLPipeline

CFG = ("configs/curated_models.json", "docs/preset_samples")

def load():
    with open(CFG[0], "r", encoding="utf-8") as f:
        return json.load(f)

def run_one(preset, prompt):
    model = preset.get("model", "stabilityai/stable-diffusion-xl-base-1.0")
    pipe = StableDiffusionXLPipeline.from_pretrained(
        model,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    ).to("cuda" if torch.cuda.is_available() else "cpu")
    img = pipe(
        prompt=prompt+", "+", ".join(preset.get("positive", [])),
        negative_prompt=", ".join(preset.get("negative", [])),
        num_inference_steps=preset.get("steps", 30),
        guidance_scale=preset.get("cfg", 7.0)
    ).images[0]
    outdir = os.path.join(CFG[1], preset["name"].replace(" ", "_"))
    os.makedirs(outdir, exist_ok=True)
    p = os.path.join(outdir, f"sample_{int(time.time())}.png")
    img.save(p); print("[OK]", p)

if __name__ == "__main__":
    data = load()
    prompt = "heroic character portrait"
    for preset in data:
        run_one(preset, prompt)
