from __future__ import annotations

import os
import queue
import random
import threading
from typing import Generator

import gradio as gr
from PIL import Image

from . import model_setup, ui_guard, ui_theme
from .bulletproof import BulletProofGenerator
from .metadata import save_metadata
from .plugins.base import UIContext
from .plugins.manager import get_plugin_manager
from .presets import Presets
from .reference_gallery import build_gallery

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
    *plugin_values,
) -> Generator:
    manager = get_plugin_manager()
    preset = PRESETS.get(preset_name) or {}
    generator = BulletProofGenerator(preset)

    seed_int = int(seed_val)
    if jitter_val:
        seed_int = seed_int + random.randint(0, int(jitter_val))

    suggested_steps = int(preset.get("suggested", {}).get("steps", 30))
    progress_interval = max(1, suggested_steps // PREVIEW_INTERVAL)
    augmented_prompt = _augment_prompt(
        ", ".join(filter(None, [preset.get("style_prompt"), prompt_txt]))
    )
    negative_prompt = preset.get("negative_prompt")

    if ref_path:
        run_kwargs = dict(
            ref_image_path=ref_path,
            prompt=augmented_prompt,
            strength=float(ref_strength),
        )
        run_fn = generator.refine
    else:
        run_kwargs = dict(prompt=augmented_prompt)
        run_fn = generator.generate

    session = manager.create_session(
        preset_name=preset_name,
        seed=seed_int,
        size=int(size_val),
        ref_image=ref_path or None,
        ref_strength=float(ref_strength) if ref_path else None,
    )

    manager.prepare_session(session, list(plugin_values))

    result_queue: "queue.Queue" = queue.Queue()
    start_updates = manager.on_generation_start(session)
    result_queue.put(("start", start_updates))

    def _progress(step: int, total: int, image):
        plugin_updates = manager.on_preview(session, step, total, image)
        result_queue.put(("preview", step, total, image, plugin_updates))

    def _worker():
        try:
            controlnet_config = session.storage.get("controlnet")
            ip_adapter_config = session.storage.get("ip_adapter")

            out_path = run_fn(
                seed=seed_int,
                size=int(size_val),
                progress_callback=_progress,
                progress_interval=progress_interval,
                controlnet_config=controlnet_config,
                ip_adapter_config=ip_adapter_config,
                negative_prompt=negative_prompt,
                **run_kwargs,
            )
            warnings = (
                generator.consume_warnings() if hasattr(generator, "consume_warnings") else []
            )
            metadata = {
                "prompt": augmented_prompt,
                "preset": preset_name,
                "seed": seed_int,
                "size": int(size_val),
                "ref": bool(ref_path),
            }
            if negative_prompt:
                metadata["negative_prompt"] = negative_prompt
            meta_path = save_metadata(os.path.dirname(out_path), metadata)

            try:
                with Image.open(out_path) as img:
                    final_image = img.copy()
            except Exception:
                final_image = None

            plugin_updates = manager.on_generation_complete(session, final_image, out_path, meta_path)
            result_queue.put(("final", final_image or out_path, meta_path, plugin_updates, warnings))
        except Exception as exc:
            warnings = (
                generator.consume_warnings() if hasattr(generator, "consume_warnings") else []
            )
            plugin_updates = manager.on_error(session, str(exc))
            result_queue.put(("error", str(exc), plugin_updates, warnings))
        finally:
            result_queue.put(("done",))

    threading.Thread(target=_worker, daemon=True).start()

    while True:
        message = result_queue.get()
        tag = message[0]

        if tag == "start":
            plugin_updates = message[1]
            yield (
                gr.update(value=None),
                gr.update(value=""),
                gr.update(value="Preparing..."),
                *plugin_updates,
            )
        elif tag == "preview":
            _tag, step_index, total, image, plugin_updates = message
            yield (
                gr.update(),
                gr.update(),
                gr.update(value=f"Rendering... step {step_index + 1}/{total}"),
                *plugin_updates,
            )
        elif tag == "final":
            _tag, image_value, meta_path, plugin_updates, warnings = message
            status_value = "Done"
            if warnings:
                bullet_list = "\n".join(f"- {warning}" for warning in warnings)
                status_value = f"Done with warnings:\n{bullet_list}"
            yield (
                gr.update(value=image_value),
                gr.update(value=meta_path),
                gr.update(value=status_value),
                *plugin_updates,
            )
        elif tag == "error":
            _tag, error_message, plugin_updates, warnings = message
            status_value = f"Error: {error_message}"
            if warnings:
                bullet_list = "\n".join(f"- {warning}" for warning in warnings)
                status_value = f"{status_value}\nWarnings:\n{bullet_list}"
            yield (
                gr.update(value=None),
                gr.update(value=""),
                gr.update(value=status_value),
                *plugin_updates,
            )
        elif tag == "done":
            break


def build_ui() -> gr.Blocks:
    model_setup.ensure_directories()
    css = ui_theme.theme_css()

    plugin_manager = get_plugin_manager()

    with gr.Blocks(css=css, analytics_enabled=False, title="CharGen Studio") as demo:
        gr.HTML(
            """
        <div style=\"text-align:center; margin-bottom: 12px;\">
            <h1 style=\"color:#3cffd0; text-shadow:0 0 12px rgba(60,255,208,0.6);\">CharGen Studio</h1>
            <p style=\"color:rgba(230,245,255,0.75); font-size: 12px;\">High-fidelity character generation with consistent outputs.</p>
        </div>
        """
        )

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
                        seed = gr.Number(value=42, label="Seed", info="Re-use seeds for deterministic looks.")
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
                            ref = gr.Image(type="filepath", label="Reference Image")
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

                    with gr.Column() as plugin_container:
                        pass

                    plugin_outputs = plugin_manager.setup_ui(
                        UIContext(
                            gradio=gr,
                            container=plugin_container,
                            components={
                                "meta": meta_box,
                                "status": status,
                                "final_image": out_img,
                                "preset": preset,
                            },
                        )
                    )

            generate_btn.click(
                _stream_generation,
                inputs=[prompt, preset, seed, jitter, size, ref, strength, *plugin_manager.inputs],
                outputs=[out_img, meta_box, status, *plugin_outputs],
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
