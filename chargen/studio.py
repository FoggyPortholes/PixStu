import os
import gradio as gr
from chargen.presets import get_preset_names, get_preset, missing_assets
from chargen.generator import BulletProofGenerator

# Optional: targeted/pin editor & substitution imports
from chargen.pin_editor import Pin, apply_pin_edits
from chargen.substitution import SubstitutionEngine

# Download manager
from tools.download_manager import DownloadManager

dl = DownloadManager()

RETRO_CSS = """
:root { --accent: #44e0ff; }
body { font-family: 'Press Start 2P', monospace; background: #0a0a0f; color: #e6e6f0; }
.gr-button { border-radius: 16px; }
"""

def build_ui():
    with gr.Blocks(css=RETRO_CSS, title="CharGen Studio") as demo:
        with gr.Tab("Character Studio"):
            preset = gr.Dropdown(label="Preset", choices=get_preset_names(), info="Select style/model preset")
            prompt = gr.Textbox(label="Prompt", info="Describe the character, pose, or action")
            seed = gr.Number(label="Seed", value=42, precision=0, info="Use same seed for reproducibility")
            go = gr.Button("Generate", info="Create character image")
            out = gr.Image(label="Output", type="pil")
            missing = gr.HTML(label="Missing Assets", visible=False)

            def _check_missing(preset_name):
                p = get_preset(preset_name)
                miss = missing_assets(p)
                if not miss:
                    return gr.update(visible=False, value="")
                items = [f"<li>{os.path.basename(m['path'])} — ~{m['size_gb']} GB</li>" for m in miss]
                html = "<b>Missing assets:</b><ul>" + "".join(items) + "</ul>Go to the Downloads tab to queue. Only one download runs at a time."
                return gr.update(visible=True, value=html)

            preset.change(_check_missing, [preset], [missing])

            def _run(preset_name, pr, sd):
                p = get_preset(preset_name)
                if missing_assets(p):
                    raise gr.Error("Preset assets missing. See Downloads tab.")
                gen = BulletProofGenerator(p)
                return gen.generate(pr, int(sd))

            go.click(_run, [preset, prompt, seed], [out])

        with gr.Tab("Substitution"):
            preset_dd = gr.Dropdown(label="Preset", choices=get_preset_names(), info="Select preset for substitution")
            char1 = gr.Image(label="Identity Image (char1)", type="pil", info="Upload reference image for identity")
            char2 = gr.Image(label="Pose Image (char2)", type="pil", info="Upload pose image (OpenPose auto-extract if available)")
            sprompt = gr.Textbox(label="Prompt", lines=2, info="Extra description for substitution run")
            id_strength = gr.Slider(0.0, 1.0, value=0.7, step=0.05, label="Identity Strength", info="Blend ratio of identity image")
            pose_strength = gr.Slider(0.0, 2.0, value=1.0, step=0.05, label="Pose Strength", info="Strength of pose conditioning")
            sseed = gr.Number(label="Seed", value=42, precision=0, info="Seed for deterministic substitution")
            go2 = gr.Button("Generate Substitution", info="Run identity→pose substitution")
            sub_out = gr.Image(label="Output")

            def _run_sub(preset_name, i1, i2, pr, ids, poses, sd):
                p = get_preset(preset_name)
                if missing_assets(p):
                    raise gr.Error("Preset assets missing. See Downloads tab.")
                eng = SubstitutionEngine(p)
                return eng.run(i1, i2, pr, identity_strength=ids, pose_strength=poses, seed=int(sd))

            go2.click(_run_sub, [preset_dd, char1, char2, sprompt, id_strength, pose_strength, sseed], [sub_out])

        with gr.Tab("Pin Editor"):
            pin_base = gr.Image(label="Base Image", type="pil", info="Image to edit with targeted pins")
            pin_table = gr.Dataframe(headers=["x","y","label","prompt"], row_count=(0, "dynamic"), label="Pins Table", interactive=True)
            ref_img = gr.Image(label="Optional Reference Image", type="pil")
            radius = gr.Slider(8, 128, value=32, step=1, label="Pin Radius", info="Mask radius around each pin")
            apply_btn = gr.Button("Apply Pin Edits", info="Run placeholder inpaint per pin")
            out_gallery = gr.Gallery(label="Pin Edit Results", columns=3)

            def _apply(base_img, rows, ref, r):
                if base_img is None or not rows:
                    return []
                pins = []
                for row in rows:
                    try:
                        x, y, label, pr = int(row[0]), int(row[1]), str(row[2]), str(row[3])
                        pins.append(Pin(x, y, label or "pin", pr or "", ref))
                    except Exception:
                        continue
                from chargen import pin_editor as pe
                def _editor_fn(img, mask, prompt, ref_img):
                    # TODO: integrate with real inpaint; placeholder returns img
                    return img
                return list(pe.apply_pin_edits(base_img, pins, _editor_fn).values())

            apply_btn.click(_apply, [pin_base, pin_table, ref_img, radius], [out_gallery])

        with gr.Tab("Reference Gallery"):
            gr.Markdown("(Placeholder) Thumbnails grid. Click to load as reference.")

        with gr.Tab("Downloads"):
            url_in = gr.Textbox(label="Model/LoRA URL", info="Direct download link")
            file_in = gr.Textbox(label="Save As (filename)", info="File name to save under loras/")
            size_in = gr.Number(label="Size (GB)", value=1.0, info="Approximate size for info only")
            add_btn = gr.Button("Queue Download", info="Add to background download queue")
            status = gr.Textbox(label="Status", interactive=False)

            def _queue(url, fname, size):
                if not url or not fname:
                    return "Invalid input"
                dl.add(url, fname, float(size))
                dl.run_async()
                return f"Queued {fname} ({size} GB)"

            add_btn.click(_queue, [url_in, file_in, size_in], [status])

    return demo

if __name__ == "__main__":
    demo = build_ui()
    try:
        from chargen.ui_guard import check_ui
        for m in check_ui(demo):
            print(m)
    except Exception as e:
        print("[UI] Drift check skipped:", e)
    demo.launch(server_name="127.0.0.1", server_port=int(os.getenv("PCS_PORT", "7860")))
