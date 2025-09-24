import os
import gradio as gr
from .presets import Presets
from .generator import CharacterGenerator
from .metadata import save_metadata
from .reference_gallery import gallery_ui
from . import lora_catalog

PRESETS = Presets()

CONSISTENCY_SUFFIX = ", highly coherent character design, consistent identity, sharp details, clean silhouette"


def _augment_prompt(p: str) -> str:
    p = p.strip() if p else ""
    return (p + CONSISTENCY_SUFFIX).strip(", ")


def _preset_details(name: str) -> str:
    return PRESETS.describe(name)


def _lora_table_rows():
    rows = []
    for record in lora_catalog.list_records():
        status = "available" if record.exists else "downloadable" if record.repo_id else "missing"
        rows.append([record.name, record.path, status, record.description])
    if not rows:
        rows.append(["(none)", "", "", "No LoRAs detected."])
    return rows


def _lora_preview(name: str):
    record = lora_catalog.find_record(name) if name else None
    if not record:
        return None, "Select a LoRA from the catalog to view details."
    status = "available" if record.exists else "downloadable" if record.repo_id else "missing"
    lines = [
        f"**Name:** {record.name}",
        f"**Status:** {status}",
        f"**Path:** `{record.path}`",
    ]
    if record.repo_id:
        lines.append(f"**Repo:** `{record.repo_id}`")
    if record.description:
        lines.append(record.description)
    if record.tags:
        lines.append("**Tags:** " + ", ".join(record.tags))
    return record.preview_url, "\n".join(lines)


def _download_lora(name: str, preset_name: str):
    if not name:
        message = "Select a LoRA first."
    else:
        message = lora_catalog.download(name)
    PRESETS.refresh()
    records = lora_catalog.list_records()
    rows = [[rec.name, rec.path, "available" if rec.exists else "downloadable" if rec.repo_id else "missing", rec.description] for rec in records]
    if not rows:
        rows = [["(none)", "", "", "No LoRAs detected."]]
    choices = [rec.name for rec in records]
    dropdown = gr.update(choices=choices, value=name if name in choices else None)
    preview, desc = _lora_preview(name if name in choices else None)
    return (
        message,
        rows,
        dropdown,
        preview,
        desc,
        PRESETS.describe(preset_name),
    )


def build_ui():
    with gr.Blocks(title="PixStu - Character Generator", analytics_enabled=False) as demo:
        gr.Markdown("## Character Studio")

        with gr.Row():
            prompt = gr.Textbox(
                label="Prompt",
                placeholder="e.g., silver-haired mage with ornate staff, fantasy RPG",
            )
            preset = gr.Dropdown(
                PRESETS.names(),
                label="Preset",
            )
        preset_info = gr.Markdown(_preset_details(PRESETS.names()[0]) if PRESETS.names() else "No presets configured.")
        preset.change(_preset_details, inputs=preset, outputs=preset_info)

        with gr.Row():
            seed = gr.Number(value=42, label="Seed")
            jitter = gr.Slider(0, 50, value=0, step=1, label="Seed Jitter")
            size = gr.Slider(256, 1024, value=512, step=64, label="Output Size")
        with gr.Row():
            ref = gr.Image(type="filepath", label="Reference Image (optional)")
            strength = gr.Slider(0.1, 0.9, value=0.35, step=0.05, label="Ref Strength (img2img)")

        out_img = gr.Image(label="Output Character")
        meta_box = gr.Textbox(label="Metadata Path", interactive=False)
        status = gr.Textbox(label="Status", interactive=False)

        def _run(prompt_txt, preset_name, seed_val, jitter_val, size_val, ref_path, ref_strength):
            try:
                pconf = PRESETS.get(preset_name) or {}
                gen = CharacterGenerator(pconf)
                seed_val = int(seed_val)
                if jitter_val:
                    import random

                    seed_val = seed_val + random.randint(0, int(jitter_val))
                aug = _augment_prompt(prompt_txt)

                if ref_path:
                    out_path = gen.refine(
                        ref_path,
                        aug,
                        strength=float(ref_strength),
                        seed=seed_val,
                        size=int(size_val),
                    )
                else:
                    out_path = gen.generate(aug, seed=seed_val, size=int(size_val))

                meta = {
                    "prompt": aug,
                    "preset": preset_name,
                    "seed": seed_val,
                    "size": int(size_val),
                    "ref": bool(ref_path),
                }
                mpath = save_metadata(os.path.dirname(out_path), meta)
                return out_path, mpath, "Done"
            except Exception as e:
                import traceback

                traceback.print_exc()
                return None, "", f"Error: {e}"

        gr.Button("Generate Character").click(
            _run,
            inputs=[prompt, preset, seed, jitter, size, ref, strength],
            outputs=[out_img, meta_box, status],
        )

        with gr.Accordion("LoRA Library", open=False):
            lora_select = gr.Dropdown(
                [rec.name for rec in lora_catalog.list_records()],
                label="LoRA Catalog",
                allow_custom_value=False,
            )
            with gr.Row():
                lora_preview = gr.Image(label="Preview", interactive=False)
                lora_desc = gr.Markdown("Select a LoRA from the catalog to view details.")
            lora_table = gr.Dataframe(
                value=_lora_table_rows(),
                headers=["Name", "Path", "Status", "Description"],
                datatype=["str", "str", "str", "str"],
                interactive=False,
                label="Local LoRAs",
            )
            download_status = gr.Textbox(label="Download Status", interactive=False)
            download_btn = gr.Button("Download selected LoRA", variant="secondary")

            lora_select.change(_lora_preview, inputs=lora_select, outputs=[lora_preview, lora_desc])
            download_btn.click(
                _download_lora,
                inputs=[lora_select, preset],
                outputs=[download_status, lora_table, lora_select, lora_preview, lora_desc, preset_info],
            )

        gr.Markdown("---\n### Reference Gallery")
        gallery_ui()

    return demo
