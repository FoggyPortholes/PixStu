import os
import socket

import gradio as gr

from chargen.presets import get_preset_names, get_preset, missing_assets
from chargen.generator import BulletProofGenerator
from chargen.pin_editor import Pin, apply_pin_edits
from chargen.substitution import SubstitutionEngine
from chargen.logging_config import configure_logging
from tools.download_manager import DownloadManager

dl = DownloadManager()

RETRO_CSS = """
:root { --accent: #44e0ff; }
body { font-family: 'Press Start 2P', monospace; background: #0a0a0f; color: #e6e6f0; }
.gr-button { border-radius: 16px; }
"""


def _env_port() -> int | None:
    for key in ("PCS_PORT", "GRADIO_SERVER_PORT"):
        value = os.environ.get(key)
        if value:
            try:
                return int(value)
            except ValueError:
                continue
    return None


def _port_available(host: str, port: int) -> bool:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind((host, port))
        return True
    except OSError:
        return False


def _pick_port(host: str) -> int | None:
    env_port = _env_port()
    if env_port:
        return env_port
    default_port = 7860
    return default_port if _port_available(host, default_port) else None


def build_ui():
    with gr.Blocks(css=RETRO_CSS, title="CharGen Studio") as demo:
        with gr.Tab("Character Studio"):
            preset = gr.Dropdown(label="Preset", choices=get_preset_names(), info="Select style/model preset")
            prompt = gr.Textbox(label="Prompt", info="Describe the character, pose, or action")
            seed = gr.Number(label="Seed", value=42, precision=0, info="Use same seed for reproducibility")
            generate_btn = gr.Button("Generate", info="Create character image")
            out_img = gr.Image(label="Output", type="pil")
            missing_html = gr.HTML(label="Missing Assets", visible=False)

            def _check_missing(preset_name):
                preset_data = get_preset(preset_name) or {}
                miss = missing_assets(preset_data)
                if not miss:
                    return gr.update(visible=False, value="")
                items = [
                    f"<li>{os.path.basename(m['path'])} - ~{m['size_gb']} GB</li>"
                    for m in miss
                    if m.get("path")
                ]
                html = (
                    "<b>Missing assets:</b><ul>"
                    + "".join(items)
                    + "</ul>Go to the Downloads tab to queue. Only one download runs at a time."
                )
                return gr.update(visible=True, value=html)

            preset.change(_check_missing, [preset], [missing_html])

            def _run(preset_name, text_prompt, seed_value):
                preset_data = get_preset(preset_name) or {}
                if missing_assets(preset_data):
                    raise gr.Error("Preset assets missing. See Downloads tab.")
                generator = BulletProofGenerator(preset_data)
                return generator.generate(text_prompt, int(seed_value))

            generate_btn.click(_run, [preset, prompt, seed], [out_img])

        with gr.Tab("Substitution"):
            preset_sub = gr.Dropdown(label="Preset", choices=get_preset_names(), info="Select preset for substitution")
            identity_img = gr.Image(label="Identity Image (char1)", type="pil", info="Upload reference image for identity")
            pose_img = gr.Image(label="Pose Image (char2)", type="pil", info="Upload pose image (OpenPose auto-extract if available)")
            sub_prompt = gr.Textbox(label="Prompt", lines=2, info="Extra description for substitution run")
            id_strength = gr.Slider(0.0, 1.0, value=0.7, step=0.05, label="Identity Strength", info="Blend ratio of identity image")
            pose_strength = gr.Slider(0.0, 2.0, value=1.0, step=0.05, label="Pose Strength", info="Strength of pose conditioning")
            sub_seed = gr.Number(label="Seed", value=42, precision=0, info="Seed for deterministic substitution")
            sub_btn = gr.Button("Generate Substitution", info="Run identity?pose substitution")
            sub_output = gr.Image(label="Output")

            def _run_sub(preset_name, identity, pose, text_prompt, ids, poses, seed_value):
                preset_data = get_preset(preset_name) or {}
                if missing_assets(preset_data):
                    raise gr.Error("Preset assets missing. See Downloads tab.")
                engine = SubstitutionEngine(preset_data)
                return engine.run(
                    char1_identity=identity,
                    char2_pose=pose,
                    prompt=text_prompt,
                    identity_strength=float(ids or 0.0),
                    pose_strength=float(poses or 0.0),
                    seed=int(seed_value or 0),
                )

            sub_btn.click(
                _run_sub,
                [preset_sub, identity_img, pose_img, sub_prompt, id_strength, pose_strength, sub_seed],
                [sub_output],
            )

        with gr.Tab("Pin Editor"):
            preset_pin = gr.Dropdown(label="Preset (optional)", choices=get_preset_names(), info="Use preset's base model for inpaint")
            pin_base = gr.Image(label="Base Image", type="pil", info="Image to edit with targeted pins")
            pin_table = gr.Dataframe(headers=["x", "y", "label", "prompt"], row_count=(0, "dynamic"), label="Pins Table", interactive=True)
            ref_img = gr.Image(label="Optional Reference Image", type="pil")
            radius = gr.Slider(8, 128, value=32, step=1, label="Pin Radius", info="Mask radius around each pin")
            apply_btn = gr.Button("Apply Pin Edits", info="Run placeholder inpaint per pin")
            gallery = gr.Gallery(label="Pin Edit Results", columns=3)

            def _apply(preset_name, base_img, rows, ref_image, radius_value):
                if base_img is None or not rows:
                    return []
                pins = []
                for row in rows:
                    try:
                        x, y, label, prompt_text = int(row[0]), int(row[1]), str(row[2]), str(row[3])
                        pins.append(Pin(x, y, label or "pin", prompt_text or "", ref_image))
                    except Exception:
                        continue

                def _editor_fn(img, mask, prompt_text, ref):  # placeholder
                    return img

                return list(apply_pin_edits(base_img, pins, _editor_fn).values())

            apply_btn.click(_apply, [preset_pin, pin_base, pin_table, ref_img, radius], [gallery])

        with gr.Tab("Reference Gallery"):
            gr.Markdown("(Placeholder) Thumbnails grid. Click to load as reference.")

        with gr.Tab("Downloads"):
            url = gr.Textbox(label="Model/LoRA URL", info="Direct download link")
            filename = gr.Textbox(label="Save As (filename)", info="File name to save under loras/")
            size = gr.Number(label="Size (GB)", value=1.0, info="Approximate size for info only")
            queue_btn = gr.Button("Queue Download", info="Add to background download queue")
            status = gr.Textbox(label="Status", interactive=False)

            def _queue(url_value, filename_value, size_value):
                if not url_value or not filename_value:
                    return "Invalid input"
                dl.add(url_value, filename_value, float(size_value or 0.0))
                dl.run_async()
                return f"Queued {filename_value} ({size_value} GB)"

            queue_btn.click(_queue, [url, filename, size], [status])

    return demo


def build_app() -> gr.Blocks:
    configure_logging()
    demo = build_ui()
    try:
        from chargen.ui_guard import check_ui

        for issue in check_ui(demo):
            print(issue)
    except Exception as exc:  # pragma: no cover
        print("[UI] Drift check skipped:", exc)
    return demo


if __name__ == "__main__":
    app = build_app()
    server_name = os.getenv("PCS_SERVER_NAME", "127.0.0.1")
    port = _pick_port(server_name)
    open_browser = os.getenv("PCS_OPEN_BROWSER", "0").lower() in {"1", "true", "yes", "on"}
    app.launch(
        share=False,
        inbrowser=open_browser,
        server_name=server_name,
        server_port=port,
        show_error=True,
    )
