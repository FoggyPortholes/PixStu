import os
import random
from typing import Tuple

import gradio as gr

from .generator import CharacterGenerator
from .metadata import save_metadata
from .presets import Presets
from .reference_gallery import build_gallery
from . import model_setup, ui_guard, ui_theme

PRESETS = Presets()
CONSISTENCY_SUFFIX = ", highly coherent character design, consistent identity, sharp details, clean silhouette"
EXPECTED_TABS = ["Character Studio", "Reference Gallery"]


def _augment_prompt(prompt: str) -> str:
    prompt = (prompt or "").strip()
    return (prompt + CONSISTENCY_SUFFIX).strip(", ")


def _run_generation(
    prompt_txt: str,
    preset_name: str,
    seed_val: float,
    jitter_val: float,
    size_val: float,
    ref_path: str,
    ref_strength: float,
) -> Tuple[str | None, str, str]:
    try:
        preset = PRESETS.get(preset_name) or {}
        generator = CharacterGenerator(preset)
        seed_int = int(seed_val)
        if jitter_val:
            seed_int = seed_int + random.randint(0, int(jitter_val))
        augmented = _augment_prompt(prompt_txt)

        if ref_path:
            out_path = generator.refine(
                ref_path,
                augmented,
                strength=float(ref_strength),
                seed=seed_int,
                size=int(size_val),
            )
        else:
            out_path = generator.generate(augmented, seed=seed_int, size=int(size_val))

        metadata = {
            "prompt": augmented,
            "preset": preset_name,
            "seed": seed_int,
            "size": int(size_val),
            "ref": bool(ref_path),
        }
        meta_path = save_metadata(os.path.dirname(out_path), metadata)
        return out_path, meta_path, "Done"
    except Exception as exc:  # pragma: no cover - runtime safety
        return None, "", f"Error: {exc}"


def build_ui() -> gr.Blocks:
    model_setup.ensure_directories()
    css = ui_theme.theme_css()

    with gr.Blocks(css=css, analytics_enabled=False, title="CharGen Studio") as demo:
        gr.HTML("""
        <div style="text-align:center; margin-bottom: 12px;">
            <h1 style="color:#3cffd0; text-shadow:0 0 12px rgba(60,255,208,0.6);">CharGen Studio</h1>
            <p style="color:rgba(230,245,255,0.75); font-size: 12px;">High-fidelity character generation with consistent outputs.</p>
        </div>
        """)

        with gr.Tabs(elem_id="chargen-tabs"):
            with gr.TabItem("Character Studio"):
                with gr.Column(elem_classes=["chargen-panel"]):
                    gr.Markdown("### Prompt & Preset", elem_classes=["chargen-group-title"])
                    with gr.Row():
                        prompt = gr.Textbox(
                            label="Prompt",
                            placeholder="e.g., cyber ninja with neon katana",
                            info="Describe the character. Retro flair encouraged.",
                        )
                        preset = gr.Dropdown(
                            PRESETS.names(),
                            label="Preset",
                            info="Pick from curated styles for consistent outputs.",
                        )
                    gr.Markdown("### Seed Controls", elem_classes=["chargen-group-title"])
                    with gr.Row():
                        seed = gr.Number(
                            value=42,
                            label="Seed",
                            info="Re-use seeds for deterministic looks.",
                        )
                        jitter = gr.Slider(
                            minimum=0,
                            maximum=50,
                            value=0,
                            step=1,
                            label="Seed Jitter",
                            info="Randomly nudge the seed within this range.",
                        )
                        size = gr.Slider(
                            minimum=256,
                            maximum=1024,
                            value=512,
                            step=64,
                            label="Output Size",
                            info="Square render size in pixels.",
                        )
                    gr.Markdown("### Reference Guidance", elem_classes=["chargen-group-title"])
                    with gr.Row():
                        ref = gr.Image(
                            type="filepath",
                            label="Reference Image",
                            info="Drop a character reference to enable img2img.",
                        )
                        strength = gr.Slider(
                            minimum=0.1,
                            maximum=0.9,
                            value=0.35,
                            step=0.05,
                            label="Ref Strength",
                            info="Blend between prompt and reference (img2img).",
                        )
                    generate_btn = gr.Button("Generate", elem_classes=["chargen-primary"])
                    out_img = gr.Image(label="Output Character", interactive=False)
                    meta_box = gr.Textbox(
                        label="Metadata JSON",
                        interactive=False,
                        info="Saved metadata file path.",
                    )
                    status = gr.Textbox(
                        label="Status",
                        interactive=False,
                        info="Generation status and warnings.",
                    )

                generate_btn.click(
                    _run_generation,
                    inputs=[prompt, preset, seed, jitter, size, ref, strength],
                    outputs=[out_img, meta_box, status],
                )

            with gr.TabItem("Reference Gallery"):
                with gr.Column(elem_classes=["chargen-panel"]):
                    gr.Markdown("### Reference Gallery", elem_classes=["chargen-group-title"])
                    gr.Markdown(
                        "Upload assets to `reference_gallery/` to curate your inspiration deck.",
                        elem_id="gallery-hint",
                    )
                    build_gallery()

        ui_guard.assert_tabs(EXPECTED_TABS)

    return demo
