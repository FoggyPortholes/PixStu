import os
import subprocess
import gradio as gr
from huggingface_hub import login as hf_login, hf_hub_download
from chargen.presets import get_preset_names, get_preset, missing_assets
from chargen.generator import BulletProofGenerator
from chargen.substitution import SubstitutionEngine
from chargen.pin_editor import Pin
# Animation imports
from chargen.txt2gif import txt2gif
from chargen.img2gif import img2gif
from chargen.txt2vid import txt2vid
from chargen.txt2vid_diffusers import txt2vid_diffusers
from chargen.txt2vid_wan import txt2vid_wan_guarded

RETRO_CSS = ":root { --accent: #44e0ff; } body { font-family: 'Press Start 2P', monospace; background: #0a0a0f; color: #e6e6f0; } .gr-button{border-radius:16px;}"


def _preset_to_lora_rows(preset):
    if not preset:
        return []
    return [
        [l.get("path", ""), l.get("weight", 1.0), l.get("download", ""), l.get("size_gb", "")]
        for l in preset.get("loras", [])
    ]


def _quick_render(preset_name, lora_path, weight):
    p = get_preset(preset_name)
    if not p:
        raise gr.Error("Preset not found")
    for l in p.get("loras", []):
        l["weight"] = float(weight) if l.get("path") == lora_path else 0.0
    gen = BulletProofGenerator(p)
    return gen.generate("LoRA quick preview", seed=42)


def _hf_auth(token: str):
    if token:
        try:
            hf_login(token=token)
            return "[HF] Authenticated with Hugging Face Hub"
        except Exception as e:  # pragma: no cover - hub errors vary
            return f"[HF] Authentication failed: {e}"
    return "[HF] No token provided. Public models only."


def _install_wan22():
    try:
        subprocess.check_call(
            [
                "pip",
                "install",
                "--prefer-binary",
                "git+https://github.com/Wan-Video/Wan2.2.git#egg=wan22",
            ]
        )
        return "[Wan2.2] Installed successfully. Restart may be required."
    except Exception as e:  # pragma: no cover - installer side effects
        return f"[Wan2.2] Install failed: {e}"


def _auto_download_assets(preset_name):
    if not preset_name:
        return "[HF] No preset selected."
    preset = get_preset(preset_name)
    if not preset:
        return f"[HF] Unknown preset: {preset_name}"
    missing = missing_assets(preset)
    if not missing:
        return "[HF] All assets available."
    downloaded = []
    for asset in missing:
        path = asset.get("path")
        repo = asset.get("download")
        if not path or not repo:
            downloaded.append(f"Missing download info for {path}")
            continue
        filename = os.path.basename(path)
        try:
            tmp_path = hf_hub_download(repo_id=repo, filename=filename)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            os.replace(tmp_path, path)
            downloaded.append(path)
        except Exception as exc:  # pragma: no cover - external download
            downloaded.append(f"Failed: {path} ({exc})")
    prefix = "[HF] Downloaded/checked:"
    return f"{prefix} {downloaded}" if downloaded else "[HF] No downloads triggered."


def _auto_download_on_select(preset_name):
    preset = get_preset(preset_name) if preset_name else None
    rows = _preset_to_lora_rows(preset)
    status = _auto_download_assets(preset_name) if preset_name else "[HF] Select a preset."
    return rows, status


def _update_lora_weights(loras_override, preset):
    if not loras_override:
        return
    for row, lora in zip(loras_override, preset.get("loras", [])):
        if not row:
            continue
        try:
            weight = float(row[1]) if row[1] not in (None, "") else lora.get("weight", 1.0)
        except (TypeError, ValueError):
            weight = lora.get("weight", 1.0)
        lora["weight"] = weight


def build_ui():
    with gr.Blocks(css=RETRO_CSS, title="PixStu Studio") as demo:
        with gr.Tab("Hugging Face"):
            hf_token = gr.Textbox(
                label="Hugging Face Token",
                type="password",
                info="Paste your HF token for gated/private models.",
            )
            hf_btn = gr.Button("Login to Hugging Face")
            hf_status = gr.Textbox(label="Status", interactive=False)
            hf_btn.click(_hf_auth, [hf_token], [hf_status])

            wan_btn = gr.Button("Install Wan2.2 (Video Generator)")
            wan_install_status = gr.Textbox(label="Wan2.2 Install Status", interactive=False)
            wan_btn.click(_install_wan22, [], [wan_install_status])

        with gr.Tab("Character Studio"):
            preset = gr.Dropdown(
                label="Preset",
                choices=get_preset_names(),
                info="Choose a style/model preset.",
            )
            prompt = gr.Textbox(label="Prompt", info="Describe the single character.")
            seed = gr.Number(label="Seed", value=42, precision=0, info="Use the same seed to reproduce a look.")
            asset_status = gr.Textbox(label="Asset Status", interactive=False)
            gr.Markdown("Adjust LoRA weights to influence style.")
            lora_info = gr.Dataframe(
                headers=["LoRA Path", "Weight", "Download URL", "Size (GB)"],
                interactive=True,
                label="LoRAs",
            )
            download_btn = gr.Button("Download Missing Assets")
            lora_quick_btn = gr.Button("Quick Render Selected LoRA")
            lora_quick_out = gr.Image(label="LoRA Quick Preview")
            go = gr.Button("Generate", variant="primary")
            out = gr.Image(label="Output")

            preset.change(_auto_download_on_select, [preset], [lora_info, asset_status])
            download_btn.click(_auto_download_assets, [preset], [asset_status])

            def _run(preset_name, pr, sd, loras_override):
                preset_cfg = get_preset(preset_name)
                if not preset_cfg:
                    raise gr.Error("Preset not found")
                if missing_assets(preset_cfg):
                    raise gr.Error("Preset assets missing.")
                _update_lora_weights(loras_override, preset_cfg)
                gen = BulletProofGenerator(preset_cfg)
                return gen.generate(pr, int(sd))

            def _run_quick(preset_name, loras_override):
                if not loras_override:
                    raise gr.Error("No LoRA selected")
                row = loras_override[0]
                return _quick_render(preset_name, row[0], row[1])

            go.click(_run, [preset, prompt, seed, lora_info], [out])
            lora_quick_btn.click(_run_quick, [preset, lora_info], [lora_quick_out])

        with gr.Tab("Substitution"):
            preset_dd = gr.Dropdown(label="Preset", choices=get_preset_names())
            char1 = gr.Image(label="Identity Image", type="pil")
            char2 = gr.Image(label="Pose Image", type="pil")
            sprompt = gr.Textbox(label="Prompt")
            asset_status_sub = gr.Textbox(label="Asset Status", interactive=False)
            lora_info_sub = gr.Dataframe(
                headers=["LoRA Path", "Weight", "Download URL", "Size (GB)"],
                interactive=True,
            )
            download_btn_sub = gr.Button("Download Missing Assets")
            lora_quick_btn_sub = gr.Button("Quick Render Selected LoRA")
            lora_quick_out_sub = gr.Image(label="LoRA Quick Preview")
            go2 = gr.Button("Generate Substitution")
            sub_out = gr.Image(label="Output")

            preset_dd.change(_auto_download_on_select, [preset_dd], [lora_info_sub, asset_status_sub])
            download_btn_sub.click(_auto_download_assets, [preset_dd], [asset_status_sub])

            def _run_sub(preset_name, i1, i2, pr, loras_override):
                preset_cfg = get_preset(preset_name)
                if not preset_cfg:
                    raise gr.Error("Preset not found")
                if missing_assets(preset_cfg):
                    raise gr.Error("Preset assets missing.")
                _update_lora_weights(loras_override, preset_cfg)
                eng = SubstitutionEngine(preset_cfg)
                return eng.run(i1, i2, pr)

            def _run_quick_sub(preset_name, loras_override):
                if not loras_override:
                    raise gr.Error("No LoRA selected")
                row = loras_override[0]
                return _quick_render(preset_name, row[0], row[1])

            go2.click(_run_sub, [preset_dd, char1, char2, sprompt, lora_info_sub], [sub_out])
            lora_quick_btn_sub.click(_run_quick_sub, [preset_dd, lora_info_sub], [lora_quick_out_sub])

        with gr.Tab("Pin Editor"):
            preset_pin = gr.Dropdown(label="Preset (optional)", choices=get_preset_names())
            pin_base = gr.Image(label="Base Image", type="pil")
            pin_table = gr.Dataframe(headers=["x", "y", "label", "prompt"], row_count=(0, "dynamic"))
            asset_status_pin = gr.Textbox(label="Asset Status", interactive=False)
            lora_info_pin = gr.Dataframe(
                headers=["LoRA Path", "Weight", "Download URL", "Size (GB)"],
                interactive=True,
            )
            download_btn_pin = gr.Button("Download Missing Assets")
            lora_quick_btn_pin = gr.Button("Quick Render Selected LoRA")
            lora_quick_out_pin = gr.Image(label="LoRA Quick Preview")
            apply_btn = gr.Button("Apply Pin Edits")
            out_gallery = gr.Gallery(label="Results", columns=3)

            preset_pin.change(_auto_download_on_select, [preset_pin], [lora_info_pin, asset_status_pin])
            download_btn_pin.click(_auto_download_assets, [preset_pin], [asset_status_pin])

            def _apply(base_img, rows):
                if base_img is None:
                    raise gr.Error("Provide a base image for pin edits.")
                pins = []
                for row in rows or []:
                    if len(row) < 3:
                        continue
                    prompt_val = row[3] if len(row) > 3 else ""
                    pins.append(Pin(row[0], row[1], row[2], prompt_val))
                _ = pins  # Placeholder until pin editing implemented
                return [base_img]

            def _run_quick_pin(preset_name, loras_override):
                if not loras_override:
                    raise gr.Error("No LoRA selected")
                row = loras_override[0]
                return _quick_render(preset_name, row[0], row[1])

            apply_btn.click(_apply, [pin_base, pin_table], [out_gallery])
            lora_quick_btn_pin.click(_run_quick_pin, [preset_pin, lora_info_pin], [lora_quick_out_pin])

        with gr.Tab("GIF/Video"):
            preset_av = gr.Dropdown(label="Preset", choices=get_preset_names())
            prompt_av = gr.Textbox(label="Prompt")
            seed_av = gr.Number(label="Seed", value=42, precision=0)
            n_frames = gr.Slider(2, 24, value=6, step=1, label="Frames")
            fps = gr.Slider(1, 30, value=4, step=1, label="FPS")
            width = gr.Slider(256, 1024, value=512, step=64, label="Width")
            height = gr.Slider(256, 1024, value=512, step=64, label="Height")
            mode = gr.Radio(
                [
                    "txt2gif",
                    "img2gif",
                    "txt2vid (gif2mp4)",
                    "txt2vid (diffusers)",
                    "txt2vid (Wan2.2)",
                ],
                value="txt2gif",
            )
            asset_status_av = gr.Textbox(label="Asset Status", interactive=False)
            in_img = gr.Image(label="Input Image (img2gif only)", type="pil", visible=False)

            out_file = gr.File(label="Output File")
            gif_preview = gr.Image(label="GIF Preview", visible=False)
            vid_preview = gr.Video(label="MP4 Preview", visible=False)
            wan_status = gr.Textbox(label="Wan2.2 Status", interactive=False)

            preset_av.change(_auto_download_assets, [preset_av], [asset_status_av])

            def _toggle(selected):
                return gr.update(visible=(selected == "img2gif"))

            mode.change(_toggle, [mode], [in_img])

            def _run(selected, preset_name, pr, sd, nf, fps_val, width_val, height_val, img=None):
                gif_upd = gr.update(visible=False, value=None)
                vid_upd = gr.update(visible=False, value=None)
                wan_msg = ""
                path = None
                if selected in {"txt2gif", "img2gif", "txt2vid (gif2mp4)"}:
                    preset_cfg = get_preset(preset_name)
                    if not preset_cfg:
                        raise gr.Error("Preset not found")
                    if missing_assets(preset_cfg):
                        raise gr.Error("Preset assets missing.")
                if selected == "txt2gif":
                    path = txt2gif(preset_name, pr, n_frames=nf, seed=sd)
                elif selected == "img2gif":
                    path = img2gif(preset_name, img, pr, n_frames=nf, seed=sd)
                elif selected == "txt2vid (gif2mp4)":
                    path = txt2vid(preset_name, pr, n_frames=nf, fps=int(fps_val), seed=sd)
                elif selected == "txt2vid (diffusers)":
                    path = txt2vid_diffusers(pr, n_frames=nf, seed=sd)
                elif selected == "txt2vid (Wan2.2)":
                    path, wan_msg = txt2vid_wan_guarded(
                        pr,
                        n_frames=int(nf),
                        seed=int(sd) if sd is not None else None,
                        fps=int(fps_val),
                        width=int(width_val),
                        height=int(height_val),
                    )
                if isinstance(path, str):
                    if path.lower().endswith(".mp4"):
                        vid_upd = gr.update(visible=True, value=path)
                    elif path.lower().endswith(".gif"):
                        gif_upd = gr.update(visible=True, value=path)
                return path, gif_upd, vid_upd, wan_msg

            go_anim = gr.Button("Generate Animation")
            go_anim.click(
                _run,
                [mode, preset_av, prompt_av, seed_av, n_frames, fps, width, height, in_img],
                [out_file, gif_preview, vid_preview, wan_status],
            )

    return demo


if __name__ == "__main__":
    demo = build_ui()
    demo.launch(server_name="127.0.0.1", server_port=int(os.getenv("PCS_PORT", "7860")))
