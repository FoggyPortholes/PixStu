"""Simple Gradio demo that drives curated PixStu presets."""
from __future__ import annotations

import json
import os
import sys
import time
from typing import Dict, Optional, Tuple

import gradio as gr
import torch
from diffusers import StableDiffusionXLImg2ImgPipeline, StableDiffusionXLPipeline
from PIL import Image

try:
    from .gif_creator import make_gif_from_sprite
    from .gif_tools import save_gif
    from .masking import (
        estimate_bg_color,
        make_alpha,
        mask_gif,
        mask_video_to_outputs,
        save_png_sequence,
    )
except ImportError:  # pragma: no cover - direct script execution fallback
    from gif_creator import make_gif_from_sprite
    from gif_tools import save_gif
    from masking import (
        estimate_bg_color,
        make_alpha,
        mask_gif,
        mask_video_to_outputs,
        save_png_sequence,
    )

ROOT = os.path.abspath(os.path.dirname(__file__))
PROJ = os.path.abspath(os.path.join(ROOT, ".."))

PRESET_PATH = os.path.join(PROJ, "configs", "curated_models.json")
with open(PRESET_PATH, "r", encoding="utf-8") as f:
    PRESETS = json.load(f).get("presets", [])

EXE_DIR = os.path.dirname(sys.executable) if getattr(sys, "frozen", False) else PROJ
MODELS_ROOT = os.environ.get("PCS_MODELS_ROOT", os.path.join(EXE_DIR, "models"))
OUTPUTS_DIR = os.environ.get("PCS_OUTPUTS_DIR", os.path.join(PROJ, "outputs"))
os.makedirs(OUTPUTS_DIR, exist_ok=True)

_DEVICE_KIND = "cuda" if torch.cuda.is_available() else "mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu"
_DTYPE = torch.float16 if _DEVICE_KIND in {"cuda", "mps"} else torch.float32
_DEVICE = torch.device("cuda" if _DEVICE_KIND == "cuda" else "mps" if _DEVICE_KIND == "mps" else "cpu")

_PIPELINES: Dict[str, Tuple[StableDiffusionXLPipeline, Optional[StableDiffusionXLImg2ImgPipeline]]] = {}


def _make_generator(seed: Optional[int]) -> Optional[torch.Generator]:
    if seed is None:
        return None
    try:
        return torch.Generator(device=_DEVICE).manual_seed(int(seed))
    except Exception:
        return torch.manual_seed(int(seed))


def _resolve_under_models(path: Optional[str]) -> Optional[str]:
    if not path:
        return path

    expanded = os.path.expanduser(str(path))
    if os.path.isabs(expanded):
        return os.path.normpath(expanded)

    normalized = os.path.normpath(expanded)

    project_candidate = os.path.abspath(os.path.join(PROJ, normalized))
    if os.path.exists(project_candidate):
        return project_candidate

    parts = normalized.split(os.sep)
    if parts and parts[0] == "models":
        normalized = os.path.join(*parts[1:]) if len(parts) > 1 else ""

    return os.path.abspath(os.path.join(MODELS_ROOT, normalized))


def _resolve_model_id(candidate: Optional[str]) -> Optional[str]:
    if not candidate:
        return candidate
    resolved = _resolve_under_models(candidate)
    if resolved and os.path.exists(resolved):
        return resolved
    return candidate


def _ensure_pipeline(preset: Dict) -> Tuple[StableDiffusionXLPipeline, Optional[StableDiffusionXLImg2ImgPipeline]]:
    cached = _PIPELINES.get(preset["name"])
    if cached:
        return cached

    base_id = _resolve_model_id(preset.get("base_model"))
    base_pipe = StableDiffusionXLPipeline.from_pretrained(
        base_id,
        torch_dtype=_DTYPE,
        use_safetensors=True,
    )
    base_pipe.to(_DEVICE)
    base_pipe.enable_vae_tiling()
    if _DEVICE_KIND == "cuda":
        try:
            base_pipe.enable_xformers_memory_efficient_attention()
        except Exception:
            pass
    base_pipe.set_progress_bar_config(disable=True)

    refiner_pipe = None
    refiner_id = _resolve_model_id(preset.get("refiner_model"))
    if refiner_id:
        refiner_pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
            refiner_id,
            torch_dtype=_DTYPE,
            use_safetensors=True,
        )
        refiner_pipe.to(_DEVICE)
        refiner_pipe.enable_vae_tiling()
        if _DEVICE_KIND == "cuda":
            try:
                refiner_pipe.enable_xformers_memory_efficient_attention()
            except Exception:
                pass
        refiner_pipe.set_progress_bar_config(disable=True)

    _PIPELINES[preset["name"]] = (base_pipe, refiner_pipe)
    return _PIPELINES[preset["name"]]


def _apply_loras(pipe, preset: Dict) -> None:
    try:
        pipe.unload_lora_weights()
    except Exception:
        pass

    loras = preset.get("loras") or []
    if not loras:
        return

    adapters = []
    weights = []
    for entry in loras:
        path = entry.get("path")
        if not path:
            continue
        abs_path = _resolve_under_models(path)
        if not abs_path or not os.path.exists(abs_path):
            continue
        adapter_name = os.path.splitext(os.path.basename(abs_path))[0]
        try:
            pipe.load_lora_weights(abs_path, adapter_name=adapter_name)
        except Exception:
            # Adapter may already be present from a previous call; keep going.
            pass
        adapters.append(adapter_name)
        weights.append(float(entry.get("weight", 1.0)))
    if adapters:
        pipe.set_adapters(adapters, adapter_weights=weights)


def load_pipeline(preset: Dict) -> Tuple[StableDiffusionXLPipeline, Optional[StableDiffusionXLImg2ImgPipeline]]:
    base, refiner = _ensure_pipeline(preset)
    _apply_loras(base, preset)
    if refiner is not None:
        _apply_loras(refiner, preset)
    return base, refiner


def generate(prompt: str, preset_name: str, seed: int = 42, steps: Optional[int] = None, guidance: Optional[float] = None):
    prompt_text = str(prompt or "").strip()
    if not prompt_text:
        return None, "Prompt is required"
    preset = next((p for p in PRESETS if p.get("name") == preset_name), None)
    if not preset:
        return None, f"Preset {preset_name} not found"

    base, refiner = load_pipeline(preset)
    try:
        seed_value = int(seed) if seed is not None else None
    except (TypeError, ValueError):
        seed_value = None
    generator = _make_generator(seed_value)

    suggested = preset.get("suggested", {})
    steps = int(steps or suggested.get("steps", 20))
    guidance = float(guidance or suggested.get("guidance", 7.5))
    height = int(suggested.get("height", 64))
    width = int(suggested.get("width", 64))

    base_result = base(
        prompt=prompt_text,
        num_inference_steps=steps,
        guidance_scale=guidance,
        generator=generator,
        height=height,
        width=width,
    )
    image = base_result.images[0]

    if refiner is not None:
        refiner_steps = max(1, steps // 4)
        image = refiner(
            prompt=prompt_text,
            image=image,
            num_inference_steps=refiner_steps,
            guidance_scale=guidance,
            generator=generator,
            strength=0.25,
        ).images[0]

    filename = f"pixstu_{int(time.time())}.png"
    save_path = os.path.join(OUTPUTS_DIR, filename)
    image.save(save_path)

    message = f"Saved to {save_path}"
    if seed_value is not None:
        message += f" (seed {seed_value})"

    return image, message


def build_ui():
    preset_choices = [p.get("name") for p in PRESETS if p.get("name")]
    with gr.Blocks(analytics_enabled=False) as demo:
        with gr.Row():
            prompt = gr.Textbox(label="Prompt", info="Describe the character or sprite you want to generate.")
            preset = gr.Dropdown(
                preset_choices,
                value=(preset_choices[0] if preset_choices else None),
                label="Preset",
                info="Choose from available curated presets",
            )
        with gr.Row():
            seed = gr.Number(value=42, label="Seed", info="Fix a random seed for reproducibility.")
            steps = gr.Slider(5, 50, value=20, label="Steps", info="Number of denoising steps.")
            guidance = gr.Slider(1, 15, value=7.5, label="Guidance", info="Prompt adherence strength.")
        with gr.Row():
            btn = gr.Button("Generate", interactive=bool(preset_choices))
        with gr.Row():
            output_img = gr.Image(label="Output")
            output_path = gr.Textbox(label="Output Path")
        btn.click(generate, inputs=[prompt, preset, seed, steps, guidance], outputs=[output_img, output_path])
        if not preset_choices:
            gr.Markdown("No curated presets found. Add entries to `configs/curated_models.json`.")

        gif_preset_choices = [p.get("name") for p in PRESETS if p.get("name")]
        default_gif_preset = gif_preset_choices[0] if gif_preset_choices else None
        with gr.Accordion("GIF Maker (Prompt + Uploaded Sprite)", open=True):
            with gr.Row():
                sprite = gr.File(label="Upload base sprite", file_types=[".png", ".webp"], file_count="single")
                preset_gif = gr.Dropdown(
                    gif_preset_choices,
                    value=default_gif_preset,
                    label="Preset",
                    interactive=bool(gif_preset_choices),
                )
            with gr.Row():
                prompt_gif = gr.Textbox(
                    label="Prompt",
                    info="Describe the animation. If empty and preset allows, a default prompt is used.",
                )
            with gr.Row():
                frames = gr.Slider(4, 24, value=8, step=1, label="Frames")
                frame_size = gr.Slider(32, 256, value=64, step=16, label="Frame Size")
                duration = gr.Slider(40, 300, value=90, step=5, label="Frame Duration (ms)")
            with gr.Row():
                motion = gr.Radio(["Idle", "Walk", "Run", "Attack"], value="Idle", label="Motion Preset")
                seed_lock = gr.Number(value=42, label="Seed")
                seed_jitter = gr.Slider(0, 25, value=0, step=1, label="Seed Jitter")
            with gr.Row():
                img_strength = gr.Slider(0.1, 0.8, value=0.35, step=0.05, label="Img2Img Strength")
                lock_palette = gr.Checkbox(value=True, label="Palette Lock")
                export_sheet = gr.Checkbox(value=True, label="Also Export Sprite Sheet")
            with gr.Row():
                make_btn = gr.Button("Create GIF", interactive=bool(gif_preset_choices))
            with gr.Row():
                gif_output = gr.File(label="Animated GIF Output")
                sheet_output = gr.File(label="Sprite Sheet Output (optional)")
                status = gr.Textbox(label="Status")

            def _run_make_gif(
                sprite_file,
                preset_name,
                prompt,
                frames,
                frame_size,
                duration,
                motion,
                seed_lock,
                seed_jitter,
                img_strength,
                lock_palette,
                export_sheet,
            ):
                if sprite_file is None:
                    return None, None, "Please upload a base sprite PNG/WEBP."
                if not preset_name:
                    return None, None, "Please select a preset."
                try:
                    gif_path, sheet_path = make_gif_from_sprite(
                        sprite_path=sprite_file.name,
                        preset_name=preset_name,
                        prompt=prompt,
                        frames=int(frames),
                        frame_size=int(frame_size),
                        duration_ms=int(duration),
                        seed=int(seed_lock) if seed_lock is not None else None,
                        seed_jitter=int(seed_jitter),
                        motion_mode=motion,
                        img_strength=float(img_strength),
                        lock_palette=bool(lock_palette),
                        export_sheet=bool(export_sheet),
                    )
                    return gif_path, sheet_path, "GIF created successfully."
                except Exception as e:  # pragma: no cover - surfaced to UI
                    return None, None, f"Error: {e}"

            make_btn.click(
                _run_make_gif,
                inputs=[
                    sprite,
                    preset_gif,
                    prompt_gif,
                    frames,
                    frame_size,
                    duration,
                    motion,
                    seed_lock,
                    seed_jitter,
                    img_strength,
                    lock_palette,
                    export_sheet,
                ],
                outputs=[gif_output, sheet_output, status],
            )

        with gr.Accordion("Auto Mask", open=False):
            gr.Markdown("#### Automatic GIF/Video Masking — remove background to transparent")
            with gr.Row():
                src_type = gr.Radio(
                    ["Sprite/GIF", "Video"],
                    value="Sprite/GIF",
                    label="Source Type",
                    info="Choose the type of file to mask.",
                )
                tol = gr.Slider(
                    0,
                    100,
                    value=20,
                    step=1,
                    label="Tolerance",
                    info="Higher removes more near-background pixels.",
                )
                size = gr.Slider(
                    16,
                    512,
                    value=64,
                    step=16,
                    label="Target Size (square)",
                    info="Resize with nearest-neighbor for pixel crispness.",
                )
            with gr.Row():
                file_in = gr.File(
                    label="Upload Sprite/GIF/Video",
                    file_types=[".png", ".webp", ".gif", ".mp4", ".mov", ".webm"],
                )
                do_png = gr.Checkbox(
                    value=True,
                    label="Export PNG Sequence",
                    info="Saves masked frames as PNGs.",
                )
                do_gif = gr.Checkbox(
                    value=True,
                    label="Export GIF",
                    info="Outputs a transparent GIF.",
                )
            with gr.Row():
                run_mask = gr.Button("Run Auto Mask")
            with gr.Row():
                masked_gif = gr.File(label="Masked GIF Output")
                png_seq_dir = gr.Textbox(label="PNG Sequence Folder")
                mask_status = gr.Textbox(label="Status")

            def _run_mask(src_type, tol, size, file_in, do_png, do_gif):
                if file_in is None:
                    return None, "", "Please upload a file."

                path = getattr(file_in, "name", None) or getattr(file_in, "path", None)
                if not path:
                    return None, "", "Unable to read uploaded file path."

                try:
                    tolerance = int(tol)
                    target = int(size)
                except Exception:
                    tolerance = 20
                    target = 64

                try:
                    if src_type == "Sprite/GIF" and path.lower().endswith(".gif"):
                        gif_path, png_dir = mask_gif(
                            path,
                            tolerance=tolerance,
                            lock_palette=bool(do_gif),
                            export_png_seq=bool(do_png),
                        )
                        status_msg = "Masked GIF generated."
                    elif src_type == "Sprite/GIF":
                        image = Image.open(path)
                        image = image.convert("RGBA")
                        image = image.resize((target, target), Image.NEAREST)
                        bg = estimate_bg_color(image)
                        masked = make_alpha(image, bg, tolerance)
                        gif_path = None
                        if do_gif:
                            gif_path = save_gif([masked], duration_ms=120, loop=0, lock_palette=True)
                        png_dir = save_png_sequence([masked]) if do_png else None
                        status_msg = "Masked sprite processed."
                    else:
                        gif_path, png_dir = mask_video_to_outputs(
                            path,
                            tolerance=tolerance,
                            target_size=(target, target),
                            export_gif=bool(do_gif),
                            export_png_seq=bool(do_png),
                        )
                        status_msg = "Masked video processed."
                    return gif_path, png_dir or "", status_msg
                except Exception as exc:  # pragma: no cover - surfaced to UI
                    return None, "", f"Error: {exc}"

            run_mask.click(
                _run_mask,
                inputs=[src_type, tol, size, file_in, do_png, do_gif],
                outputs=[masked_gif, png_seq_dir, mask_status],
            )
    return demo


if __name__ == "__main__":
    demo = build_ui()
    demo.launch(server_name="127.0.0.1", server_port=int(os.environ.get("PCS_PORT", 7860)))
