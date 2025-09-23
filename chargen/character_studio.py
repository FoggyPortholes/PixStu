import os
import queue
import random
import threading
from typing import Generator, Tuple

import gradio as gr

from .generator import CharacterGenerator
from .metadata import save_metadata
from .presets import Presets
from .reference_gallery import build_gallery
from . import model_setup, ui_guard, ui_theme

PRESETS = Presets()
CONSISTENCY_SUFFIX = ", highly coherent character design, consistent identity, sharp details, clean silhouette"
EXPECTED_TABS = ["Character Studio", "Reference Gallery"]
PREVIEW_INTERVAL = 4


def _augment_prompt(prompt: str) -> str:
    prompt = (prompt or "").strip()
    return (prompt + CONSISTENCY_SUFFIX).strip(", ")


def _stream_generation(
    prompt_txt: str,
    preset_name: str,
    seed_val: float,
    jitter_val: float,
    size_val: float,
    ref_path: str,
    ref_strength: float,
) -> Generator:
    try:
        preset = PRESETS.get(preset_name) or {}
        generator = CharacterGenerator(preset)
        seed_int = int(seed_val)
        if jitter_val:
            seed_int = seed_int + random.randint(0, int(jitter_val))
        augmented = _augment_prompt(prompt_txt)

        suggested_steps = int(preset.get("suggested", {}).get("steps", 30))
        progress_interval = max(1, suggested_steps // PREVIEW_INTERVAL)

        if ref_path:
            run_kwargs = dict(
                ref_image_path=ref_path,
                prompt=augmented,
                strength=float(ref_strength),
            )
            run_fn = generator.refine
        else:
            run_kwargs = dict(prompt=augmented)
            run_fn = generator.generate

        result_queue: "queue.Queue[tuple]" = queue.Queue()

        def _progress(step: int, total: int, image):
            result_queue.put(("preview", step, total, image))

        def _worker():
            try:
                out_path = run_fn(
                    seed=seed_int,
                    size=int(size_val),
                    progress_callback=_progress,
                    progress_interval=progress_interval,
                    **run_kwargs,
                )
                metadata = {
                    "prompt": augmented,
                    "preset": preset_name,
                    "seed": seed_int,
                    "size": int(size_val),
                    "ref": bool(ref_path),
                }
                meta_path = save_metadata(os.path.dirname(out_path), metadata)
                result_queue.put(("final", out_path, meta_path))
            except Exception as exc:
                result_queue.put(("error", str(exc)))
            finally:
                result_queue.put(("done", None))

        threading.Thread(target=_worker, daemon=True).start()

        preview_image = None
        while True:
            tag, *payload = result_queue.get()
            if tag == "preview":
                step_index, total, image = payload
                preview_image = image
                yield (
                    gr.update(value=preview_image),
                    gr.update(),
                    gr.update(),
                    gr.update(value=f"Rendering… step {step_index + 1}/{total}"),
                )
            elif tag == "final":
                out_path, meta_path = payload
                yield (
                    gr.update(value=preview_image or out_path),
                    gr.update(value=out_path),
                    gr.update(value=meta_path),
                    gr.update(value="Done"),
                )
            elif tag == "error":
                message = payload[0]
                yield (
                    gr.update(value=preview_image),
                    gr.update(value=None),
                    gr.update(value=""),
                    gr.update(value=f"Error: {message}"),
                )
            elif tag == "done":
                break
    except Exception as exc:  # pragma: no cover - runtime safety
        yield (
            gr.update(value=None),
            gr.update(value=None),
            gr.update(value=""),
            gr.update(value=f"Error: {exc}"),
        )


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
                        with gr.Column():
                            ref = gr.Image(
                                type="filepath",
                                label="Reference Image",
                            )
                            gr.Markdown(
                                "Drop a character reference to enable img2img.",
                                elem_classes=["chargen-hint"],
                            )
                        with gr.Column():
                            strength = gr.Slider(
                                minimum=0.1,
                                maximum=0.9,
                                value=0.35,
                                step=0.05,
                                label="Ref Strength",
                                info="Blend between prompt and reference (img2img).",
                            )
                            gr.Markdown(
                                "Higher values cling to the reference silhouette.",
                                elem_classes=["chargen-hint"],
                            )
                    generate_btn = gr.Button("Generate", elem_classes=["chargen-primary"])
                    preview_img = gr.Image(label="Live Preview", interactive=False)
                    out_img = gr.Image(label="Final Character", interactive=False)
                    gr.Markdown(
                        "Results land in the `outputs/` directory with metadata logs.",
                        elem_classes=["chargen-hint"],
                    )
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
                    _stream_generation,
                    inputs=[prompt, preset, seed, jitter, size, ref, strength],
                    outputs=[preview_img, out_img, meta_box, status],
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
