import os
import gradio as gr
from .presets import Presets
from .generator import CharacterGenerator
from .metadata import save_metadata
from .reference_gallery import gallery_ui

PRESETS = Presets()

CONSISTENCY_SUFFIX = ", highly coherent character design, consistent identity, sharp details, clean silhouette"


def _augment_prompt(p: str) -> str:
    p = p.strip() if p else ""
    return (p + CONSISTENCY_SUFFIX).strip(", ")


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

        gr.Markdown("---\n### Reference Gallery")
        gallery_ui()

    return demo
