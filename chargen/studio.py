import os
from typing import Iterable, List

import gradio as gr
from huggingface_hub import hf_hub_download
from huggingface_hub import login as hf_login

from chargen.generator import BulletProofGenerator
from chargen.pin_editor import Pin
from chargen.presets import get_preset, get_preset_names, missing_assets
from chargen.substitution import SubstitutionEngine
# Animation imports
from chargen.txt2gif import txt2gif
from chargen.img2gif import img2gif
from chargen.txt2vid import txt2vid
from chargen.txt2vid_diffusers import txt2vid_diffusers
from chargen.txt2vid_wan import txt2vid_wan_guarded
from chargen.wan_install import ensure_wan22_installed

RETRO_CSS = ":root { --accent: #44e0ff; } body { font-family: 'Press Start 2P', monospace; background: #0a0a0f; color: #e6e6f0; } .gr-button{border-radius:16px;}"


def _preset_to_lora_rows(preset: dict | None) -> list[list[object]]:
    """Convert preset LoRA entries into rows consumable by ``gr.Dataframe``."""

    if not preset:
        return []

    rows: list[list[object]] = []
    for entry in preset.get("loras", []):
        display = entry.get("display_path") or entry.get("path") or ""
        weight = float(entry.get("weight", 0.0) or 0.0)
        download = entry.get("download", "")
        size = entry.get("size_gb")
        try:
            size_val: object = float(size) if size is not None else ""
        except (TypeError, ValueError):
            size_val = ""
        rows.append([display, weight, download, size_val])
    return rows


def _coerce_override_rows(overrides: object) -> list[list[object]]:
    """Normalize override data from ``gr.Dataframe`` callbacks."""

    if overrides is None:
        return []

    # Handle pandas DataFrame-like objects without relying on the import.
    if hasattr(overrides, "empty"):
        try:
            if overrides.empty:  # type: ignore[attr-defined]
                return []
        except Exception:  # pragma: no cover - defensive
            pass

    if hasattr(overrides, "to_numpy"):
        try:
            return overrides.to_numpy().tolist()  # type: ignore[call-arg,attr-defined]
        except Exception:  # pragma: no cover - defensive
            pass

    if hasattr(overrides, "tolist"):
        try:
            rows = overrides.tolist()  # type: ignore[attr-defined]
        except Exception:  # pragma: no cover - defensive
            pass
        else:
            if isinstance(rows, list):
                return rows

    try:
        rows = list(overrides)  # type: ignore[arg-type]
    except TypeError:
        return []

    if rows and not isinstance(rows[0], (list, tuple)):
        return [list(rows)]
    return [list(row) for row in rows]


def _apply_lora_overrides(preset: dict, overrides: Iterable[Iterable[object]] | object) -> None:
    """Apply LoRA weight overrides coming from the interactive table."""

    if not preset:
        return

    rows = _coerce_override_rows(overrides)
    if not rows:
        return

    override_map: dict[str, float] = {}
    for row in rows:
        if not row:
            continue
        path = row[0]
        if not path:
            continue
        try:
            weight = float(row[1])
        except (TypeError, ValueError):
            continue
        override_map[str(path)] = weight

    if not override_map:
        return

    for entry in preset.get("loras", []):
        display = entry.get("display_path") or entry.get("path")
        if display in override_map:
            entry["weight"] = override_map[display]


def _asset_status_message(missing: List[dict]) -> str:
    if not missing:
        return "All preset assets are available."
    parts = [f"• {item.get('display_path', 'unknown')}" for item in missing]
    return "Missing assets:\n" + "\n".join(parts)


def _auto_download_on_select(preset_name: str):
    preset = get_preset(preset_name)
    if not preset:
        return gr.update(value=[]), "Preset not found."

    rows = _preset_to_lora_rows(preset)
    missing = missing_assets(preset)
    status = _asset_status_message(missing)
    return gr.update(value=rows), status


def _download_via_hub(entry: dict) -> str:
    download = entry.get("download") or ""
    if not download:
        return f"{entry.get('display_path')}: no download metadata."

    repo_id = None
    filename = None
    if "::" in download:
        repo_id, filename = download.split("::", 1)
    elif download.count("/") > 1 and not download.startswith("http"):
        repo_id, filename = download.split("/", 1)

    if not repo_id or not filename or hf_hub_download is None:
        return f"{entry.get('display_path')}: unsupported download format."

    target = entry.get("resolved_path") or entry.get("path") or ""
    target_dir = os.path.dirname(target) or os.getcwd()
    os.makedirs(target_dir, exist_ok=True)
    try:
        hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=target_dir,
            local_dir_use_symlinks=False,
        )
    except Exception as exc:  # pragma: no cover - network/runtime failures
        return f"{entry.get('display_path')}: download failed ({exc})."
    return f"{entry.get('display_path')}: downloaded."


def _auto_download_assets(preset_name: str) -> str:
    preset = get_preset(preset_name)
    if not preset:
        return "Preset not found."

    missing = missing_assets(preset)
    if not missing:
        return "All preset assets are already available."

    reports: list[str] = []
    for entry in missing:
        resolved = entry.get("resolved_path") or entry.get("path")
        if resolved and os.path.exists(resolved):
            continue
        download_info = entry.get("download") or ""
        if download_info and not download_info.startswith("http"):
            reports.append(_download_via_hub(entry))
        else:
            reports.append(
                f"{entry.get('display_path')}: manual download required ({download_info or 'no URL'})."
            )
    return "\n".join(reports)


def _hf_auth(token: str) -> str:
    token = (token or "").strip()
    if not token:
        return "Provide a Hugging Face token."
    try:
        hf_login(token=token, add_to_git_credential=False)
    except Exception as exc:  # pragma: no cover - runtime failures
        return f"Login failed: {exc}"
    return "Authentication successful."


def _install_wan22() -> str:
    _, message, _ = ensure_wan22_installed()
    return f"Wan2.2 {message}"


def _quick_render(preset_name, lora_path, weight):
    p = get_preset(preset_name)
    if not p:
        raise gr.Error("Preset not found")
    missing = missing_assets(p)
    if missing:
        raise gr.Error(_asset_status_message(missing))
    try:
        target_weight = float(weight)
    except (TypeError, ValueError):
        target_weight = 0.0
    for l in p.get("loras", []):
        display = l.get("display_path") or l.get("path")
        l["weight"] = target_weight if display == lora_path else 0.0
    gen = BulletProofGenerator(p)
    return gen.generate("LoRA quick preview", seed=42)




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

                _apply_lora_overrides(preset_cfg, loras_override)

                missing = missing_assets(preset_cfg)
                if missing:
                    raise gr.Error(_asset_status_message(missing))

                gen = BulletProofGenerator(preset_cfg)
                try:
                    seed_val = int(sd) if sd is not None else 42
                except (TypeError, ValueError):
                    seed_val = 42
                return gen.generate(pr, seed=seed_val)

            def _run_quick(preset_name, loras_override):
                rows = _coerce_override_rows(loras_override)
                if not rows:
                    raise gr.Error("No LoRA selected")
                row = rows[0]
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

                _apply_lora_overrides(preset_cfg, loras_override)

                missing = missing_assets(preset_cfg)
                if missing:
                    raise gr.Error(_asset_status_message(missing))

                eng = SubstitutionEngine(preset_cfg)
                return eng.run(i1, i2, pr)

            def _run_quick_sub(preset_name, loras_override):
                rows = _coerce_override_rows(loras_override)
                if not rows:
                    raise gr.Error("No LoRA selected")
                row = rows[0]
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
                rows = _coerce_override_rows(loras_override)
                if not rows:
                    raise gr.Error("No LoRA selected")
                row = rows[0]
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
