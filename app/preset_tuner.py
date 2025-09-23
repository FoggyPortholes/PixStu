import json
import os
import sys
from typing import List, Optional, Tuple

import gradio as gr
import torch
from diffusers import StableDiffusionXLPipeline

ROOT = os.path.abspath(os.path.dirname(__file__))
PROJ = os.path.abspath(os.path.join(ROOT, ".."))
EXE_DIR = os.path.dirname(sys.executable) if getattr(sys, "frozen", False) else PROJ
MODELS_ROOT = os.getenv("PCS_MODELS_ROOT", os.path.join(EXE_DIR, "models"))
PRESET_PATH = os.path.join(PROJ, "configs", "curated_models.json")


def _abs_under_models(path: Optional[str]) -> Optional[str]:
    if not path:
        return path

    expanded = os.path.expanduser(str(path))
    if os.path.isabs(expanded):
        return os.path.normpath(expanded)

    normalized = os.path.normpath(expanded)

    project_candidate = os.path.abspath(os.path.join(PROJ, normalized))
    if os.path.exists(project_candidate):
        return os.path.normpath(project_candidate)

    parts = normalized.split(os.sep)
    if parts and parts[0] == "models":
        normalized = os.path.join(*parts[1:]) if len(parts) > 1 else ""

    return os.path.normpath(os.path.abspath(os.path.join(MODELS_ROOT, normalized)))


def _device() -> Tuple[str, torch.dtype]:
    if torch.cuda.is_available():
        return "cuda", torch.float16
    mps = getattr(torch.backends, "mps", None)
    if mps and torch.backends.mps.is_available():  # pragma: no cover - environment specific
        return "mps", torch.float16
    return "cpu", torch.float32


DEV, DTYPE = _device()


def _load_presets() -> List[dict]:
    if not os.path.isfile(PRESET_PATH):
        return []
    with open(PRESET_PATH, "r", encoding="utf-8") as fh:
        payload = json.load(fh)
    return payload.get("presets", [])


PRESETS: List[dict] = _load_presets()


def save_presets(presets: List[dict]) -> None:
    os.makedirs(os.path.dirname(PRESET_PATH), exist_ok=True)
    with open(PRESET_PATH, "w", encoding="utf-8") as fh:
        json.dump({"presets": presets}, fh, indent=2)


def _resolve_model_id(model_id: str) -> str:
    if not model_id:
        return "stabilityai/stable-diffusion-xl-base-1.0"
    resolved = _abs_under_models(model_id)
    if resolved and os.path.isdir(resolved):
        return resolved
    return model_id


def update_preset(preset_name: str, steps: int, guidance: float, weight: float) -> str:
    for preset in PRESETS:
        if preset.get("name") == preset_name:
            suggested = preset.setdefault("suggested", {})
            suggested["steps"] = steps
            suggested["guidance"] = guidance
            if preset.get("loras"):
                preset["loras"][0]["weight"] = weight
            save_presets(PRESETS)
            return (
                f"Updated preset: {preset_name} with Steps={steps}, "
                f"Guidance={guidance}, LoRA Weight={weight}"
            )
    return f"Preset '{preset_name}' not found"


def _load_pipeline(preset: dict, weight: Optional[float] = None) -> StableDiffusionXLPipeline:
    base_model = _resolve_model_id(preset.get("base_model"))
    pipe = StableDiffusionXLPipeline.from_pretrained(
        base_model,
        torch_dtype=DTYPE,
    )
    if DEV != "cpu":
        pipe.to(DEV)

    loras = preset.get("loras") or []
    if loras:
        entry = loras[0]
        lora_path = _abs_under_models(entry.get("path"))
        if lora_path and os.path.exists(lora_path):
            weight_to_use = weight if weight is not None else entry.get("weight", 1.0)
            pipe.load_lora_weights(lora_path, weight=weight_to_use)
    return pipe


def preview_generation(
    prompt: str,
    preset_name: str,
    steps: int,
    guidance: float,
    weight: float,
    seed: int = 42,
):
    preset = next((p for p in PRESETS if p.get("name") == preset_name), None)
    if not preset:
        return None, None, "Preset not found"

    generator_device = "cuda" if DEV == "cuda" else "cpu"

    pipe_before = _load_pipeline(preset)
    generator = torch.Generator(device=generator_device).manual_seed(seed)
    before = pipe_before(
        prompt=prompt or "pixel art character sprite",
        num_inference_steps=preset.get("suggested", {}).get("steps", 20),
        guidance_scale=preset.get("suggested", {}).get("guidance", 7.5),
        generator=generator,
        height=64,
        width=64,
    ).images[0]

    pipe_after = _load_pipeline(preset, weight=weight)
    generator = torch.Generator(device=generator_device).manual_seed(seed)
    after = pipe_after(
        prompt=prompt or "pixel art character sprite",
        num_inference_steps=steps,
        guidance_scale=guidance,
        generator=generator,
        height=64,
        width=64,
    ).images[0]

    return before, after, "Preview before/after generated successfully"


with gr.Blocks(analytics_enabled=False) as demo:
    gr.Markdown("Curated Preset Tuner")
    with gr.Row():
        preset = gr.Dropdown([p.get("name") for p in PRESETS], label="Preset", info="Select which preset to tune.")
    with gr.Row():
        prompt = gr.Textbox(
            label="Prompt",
            info="Optional custom prompt for preview. Leave empty to use default.",
        )
    with gr.Row():
        steps = gr.Slider(
            5,
            50,
            value=20,
            step=1,
            label="Steps",
            info="Number of denoising steps. More steps = higher quality but slower.",
        )
        guidance = gr.Slider(
            1,
            15,
            value=7.5,
            step=0.1,
            label="Guidance",
            info="Prompt adherence. Higher = closer to prompt, lower = more variety.",
        )
        weight = gr.Slider(
            0.1,
            2.0,
            value=1.0,
            step=0.1,
            label="LoRA Weight",
            info="Controls style intensity from the LoRA model. Lower = subtle, Higher = stronger.",
        )
    with gr.Row():
        update_btn = gr.Button("Update Preset")
        preview_btn = gr.Button("Preview Generation")
    with gr.Row():
        output_status = gr.Textbox(label="Update Status")
    with gr.Row():
        before_img = gr.Image(label="Before (current preset)")
        after_img = gr.Image(label="After (with adjustments)")
        preview_status = gr.Textbox(label="Preview Status")

    update_btn.click(update_preset, inputs=[preset, steps, guidance, weight], outputs=output_status)
    preview_btn.click(
        preview_generation,
        inputs=[prompt, preset, steps, guidance, weight],
        outputs=[before_img, after_img, preview_status],
    )


if __name__ == "__main__":
    port = int(os.getenv("PCS_PRESET_PORT", "7861"))
    os.environ.setdefault("HF_HOME", os.path.join(PROJ, "hf_cache"))
    os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
    demo.launch(share=False, inbrowser=True, server_name="127.0.0.1", server_port=port, show_error=True)
