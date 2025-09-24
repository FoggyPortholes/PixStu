import os, gradio as gr
from chargen.presets import get_preset_names, get_preset, missing_assets
from chargen.generator import BulletProofGenerator
from chargen.substitution import SubstitutionEngine
from chargen.pin_editor import Pin
from chargen.inpaint import inpaint_region

# Animation imports
from chargen.txt2gif import txt2gif
from chargen.img2gif import img2gif
from chargen.txt2vid import txt2vid
from chargen.txt2vid_diffusers import txt2vid_diffusers

RETRO_CSS = ":root { --accent: #44e0ff; } body { font-family: 'Press Start 2P', monospace; background: #0a0a0f; color: #e6e6f0; } .gr-button{border-radius:16px;}"

# Utility: build LoRA table + previews
from PIL import Image

def _preset_to_lora_rows(preset):
    return [[l.get("path",""), l.get("weight",1.0), l.get("download",""), l.get("size_gb","")] for l in preset.get("loras", [])]

def _quick_render(preset_name, lora_path, weight):
    p = get_preset(preset_name)
    for l in p.get("loras", []):
        l["weight"] = float(weight) if l.get("path") == lora_path else 0.0
    gen = BulletProofGenerator(p)
    return gen.generate("LoRA quick preview", seed=42)

def build_ui():
    with gr.Blocks(css=RETRO_CSS, title="PixStu Studio") as demo:
        # Character Studio
        with gr.Tab("Character Studio"):
            preset = gr.Dropdown(label="Preset", choices=get_preset_names(), info="Choose a style/model preset.")
            prompt = gr.Textbox(label="Prompt", info="Describe the single character.")
            seed = gr.Number(label="Seed", value=42, precision=0, info="Use the same seed to reproduce a look.")
            gr.Markdown("Adjust LoRA weights to influence style.")
            lora_info = gr.Dataframe(headers=["LoRA Path","Weight","Download URL","Size (GB)"], interactive=True, label="LoRAs")
            lora_quick_btn = gr.Button("Quick Render Selected LoRA")
            lora_quick_out = gr.Image(label="LoRA Quick Preview")
            go = gr.Button("Generate", variant="primary")
            out = gr.Image(label="Output")

            def _on_preset(preset_name):
                return _preset_to_lora_rows(get_preset(preset_name))
            preset.change(_on_preset, [preset], [lora_info])

            def _run(preset_name, pr, sd, loras_override):
                p = get_preset(preset_name)
                if missing_assets(p):
                    raise gr.Error("Preset assets missing.")
                for row, l in zip(loras_override, p.get("loras", [])):
                    l["weight"] = float(row[1]) if row[1] else l.get("weight",1.0)
                gen = BulletProofGenerator(p)
                return gen.generate(pr, int(sd))

            def _run_quick(preset_name, loras_override):
                if not loras_override:
                    raise gr.Error("No LoRA selected")
                row = loras_override[0]
                return _quick_render(preset_name, row[0], row[1])

            go.click(_run, [preset, prompt, seed, lora_info], [out])
            lora_quick_btn.click(_run_quick, [preset, lora_info], [lora_quick_out])

        # Substitution
        with gr.Tab("Substitution"):
            preset_dd = gr.Dropdown(label="Preset", choices=get_preset_names())
            char1 = gr.Image(label="Identity Image", type="pil")
            char2 = gr.Image(label="Pose Image", type="pil")
            sprompt = gr.Textbox(label="Prompt")
            lora_info_sub = gr.Dataframe(headers=["LoRA Path","Weight","Download URL","Size (GB)"], interactive=True)
            lora_quick_btn_sub = gr.Button("Quick Render Selected LoRA")
            lora_quick_out_sub = gr.Image(label="LoRA Quick Preview")
            go2 = gr.Button("Generate Substitution")
            sub_out = gr.Image(label="Output")

            def _on_preset_sub(preset_name):
                return _preset_to_lora_rows(get_preset(preset_name))
            preset_dd.change(_on_preset_sub, [preset_dd], [lora_info_sub])

            def _run_sub(preset_name, i1, i2, pr, loras_override):
                p = get_preset(preset_name)
                if missing_assets(p):
                    raise gr.Error("Preset assets missing.")
                for row, l in zip(loras_override, p.get("loras", [])):
                    l["weight"] = float(row[1]) if row[1] else l.get("weight",1.0)
                eng = SubstitutionEngine(p)
                return eng.run(i1, i2, pr)

            def _run_quick_sub(preset_name, loras_override):
                if not loras_override:
                    raise gr.Error("No LoRA selected")
                row = loras_override[0]
                return _quick_render(preset_name, row[0], row[1])

            go2.click(_run_sub, [preset_dd, char1, char2, sprompt, lora_info_sub], [sub_out])
            lora_quick_btn_sub.click(_run_quick_sub, [preset_dd, lora_info_sub], [lora_quick_out_sub])

        # Pin Editor
        with gr.Tab("Pin Editor"):
            preset_pin = gr.Dropdown(label="Preset (optional)", choices=get_preset_names())
            pin_base = gr.Image(label="Base Image", type="pil")
            pin_table = gr.Dataframe(headers=["x","y","label","prompt"], row_count=(0, "dynamic"))
            lora_info_pin = gr.Dataframe(headers=["LoRA Path","Weight","Download URL","Size (GB)"], interactive=True)
            lora_quick_btn_pin = gr.Button("Quick Render Selected LoRA")
            lora_quick_out_pin = gr.Image(label="LoRA Quick Preview")
            apply_btn = gr.Button("Apply Pin Edits")
            out_gallery = gr.Gallery(label="Results", columns=3)

            def _on_preset_pin(preset_name):
                return _preset_to_lora_rows(get_preset(preset_name))
            preset_pin.change(_on_preset_pin, [preset_pin], [lora_info_pin])

            def _apply(base_img, rows):
                return [base_img] if base_img else []

            def _run_quick_pin(preset_name, loras_override):
                if not loras_override:
                    raise gr.Error("No LoRA selected")
                row = loras_override[0]
                return _quick_render(preset_name, row[0], row[1])

            apply_btn.click(_apply, [pin_base, pin_table], [out_gallery])
            lora_quick_btn_pin.click(_run_quick_pin, [preset_pin, lora_info_pin], [lora_quick_out_pin])

        # GIF/Video
        with gr.Tab("GIF/Video"):
            preset_av = gr.Dropdown(label="Preset", choices=get_preset_names())
            prompt_av = gr.Textbox(label="Prompt")
            seed_av = gr.Number(label="Seed", value=42, precision=0)
            n_frames = gr.Slider(2, 16, value=6, step=1, label="Frames")
            mode = gr.Radio(["txt2gif","img2gif","txt2vid (gif2mp4)","txt2vid (diffusers)"])
            in_img = gr.Image(label="Input Image (img2gif only)", type="pil", visible=False)

            out_file = gr.File(label="Output File")
            gif_preview = gr.Image(label="GIF Preview", visible=False)
            vid_preview = gr.Video(label="MP4 Preview", visible=False)

            def _toggle(selected):
                return gr.update(visible=(selected=="img2gif"))
            mode.change(_toggle, [mode], [in_img])

            def _run(selected, preset_name, pr, sd, nf, img=None):
                path = None
                if selected=="txt2gif":
                    path = txt2gif(preset_name, pr, n_frames=nf, seed=sd)
                elif selected=="img2gif":
                    path = img2gif(preset_name, img, pr, n_frames=nf, seed=sd)
                elif selected=="txt2vid (gif2mp4)":
                    path = txt2vid(preset_name, pr, n_frames=nf, fps=4, seed=sd)
                elif selected=="txt2vid (diffusers)":
                    path = txt2vid_diffusers(pr, n_frames=nf, seed=sd)
                gif_upd = gr.update(visible=False, value=None)
                vid_upd = gr.update(visible=False, value=None)
                if isinstance(path, str):
                    if path.lower().endswith('.mp4'):
                        vid_upd = gr.update(visible=True, value=path)
                    elif path.lower().endswith('.gif'):
                        gif_upd = gr.update(visible=True, value=path)
                return path, gif_upd, vid_upd

            go_anim = gr.Button("Generate Animation")
            go_anim.click(_run, [mode, preset_av, prompt_av, seed_av, n_frames, in_img], [out_file, gif_preview, vid_preview])

    return demo

if __name__ == "__main__":
    demo = build_ui()
    demo.launch(server_name="127.0.0.1", server_port=int(os.getenv("PCS_PORT","7860")))
