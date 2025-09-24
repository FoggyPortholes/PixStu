"""Gradio application entry-point for PixStu."""

from __future__ import annotations

import json
import os
import inspect
from pathlib import Path
from typing import Iterable

import gradio as gr

from chargen import lora_blend
from chargen.generator import BulletProofGenerator
from chargen.pin_editor import Pin, apply_pin_edits
from chargen.presets import get_preset, get_preset_names, missing_assets
from chargen.substitution import SubstitutionEngine

# Optional imports for downloads
try:  # pragma: no cover - optional dependency
    from tools.downloads import KNOWN_LORAS, download_lora, resolve_missing_loras
except Exception:  # pragma: no cover - keep studio usable without huggingface_hub
    download_lora = None  # type: ignore[assignment]
    resolve_missing_loras = None  # type: ignore[assignment]
    KNOWN_LORAS: dict[str, tuple[str, str]] = {}

PIXSTU_AUTO_DOWNLOAD = os.environ.get("PIXSTU_AUTO_DOWNLOAD", "0").lower() in (
    "1",
    "true",
    "yes",
    "on",
)


_INFO_SUPPORT_CACHE: dict[type, bool] = {}


def _info_kwargs(component_cls: type, text: str) -> dict[str, str]:
    try:
        supported = _INFO_SUPPORT_CACHE.get(component_cls)
        if supported is None:
            sig = inspect.signature(component_cls.__init__)
            supported = "info" in sig.parameters
            _INFO_SUPPORT_CACHE[component_cls] = supported
        if supported:
            return {"info": text}
    except Exception:
        pass
    return {}


def _missing_assets_message(missing: list[str]) -> str:
    lines = ["Missing assets:"]
    for name in missing:
        lines.append(f"• loras/{name}")
    lines.append("\nOptions:")
    lines.append("1) Use the Downloads tab to fetch assets automatically.")
    lines.append("2) Or set PIXSTU_AUTO_DOWNLOAD=1 to fetch on-demand.")
    lines.append("3) Or remove/disable the preset referencing these LoRAs.")
    return "\n".join(lines)


def _try_auto_download(missing: list[str]) -> list[str]:
    if not PIXSTU_AUTO_DOWNLOAD or download_lora is None:
        return missing
    still_missing: list[str] = []
    for fname in missing:
        try:
            download_lora(fname)
        except Exception:  # pragma: no cover - network/optional dependency
            still_missing.append(fname)
    return still_missing


def _scan_presets_for_loras() -> list[str]:
    preset_path = Path(".pixstu/lora_sets.json")
    out: list[str] = []
    if preset_path.exists():
        try:
            data = json.loads(preset_path.read_text())
            if isinstance(data, list):
                for item in data:
                    if isinstance(item, dict) and "lora" in item:
                        out.append(str(item["lora"]))
                    elif isinstance(item, list):
                        for sub in item:
                            if isinstance(sub, dict) and "lora" in sub:
                                out.append(str(sub["lora"]))
        except Exception:  # pragma: no cover - malformed custom config should not crash UI
            pass
    return sorted(set(out))


def _missing_lora_filenames(entries: Iterable[dict]) -> list[str]:
    names: set[str] = set()
    for entry in entries:
        path = (
            entry.get("display_path")
            or entry.get("resolved_path")
            or entry.get("path")
            or ""
        )
        if not path:
            continue


    if selected_entry is None:
        selected_entry = {
            "display_path": lora_path,
            "resolved_path": str(expected_path),
            "path": lora_path,
        }

    selected_entry = dict(selected_entry)
    selected_entry["weight"] = float(strength)
    preset["loras"] = [selected_entry]

    generator = BulletProofGenerator(preset)
    return generator.generate("LoRA quick preview", int(strength or 0))


RETRO_CSS = """
:root { --accent: #44e0ff; }
body { font-family: 'Press Start 2P', monospace; background: #0a0a0f; color: #e6e6f0; }
.gr-button { border-radius: 16px; }
"""


def build_ui() -> gr.Blocks:
    with gr.Blocks(css=RETRO_CSS, title="CharGen Studio") as demo:
        with gr.Tab("Character Studio"):
            preset = gr.Dropdown(
                label="Preset",
                choices=get_preset_names(),
                **_info_kwargs(gr.Dropdown, "Select style/model preset"),
            )
            prompt = gr.Textbox(
                label="Prompt",
                **_info_kwargs(gr.Textbox, "Describe the character, pose, or action"),
            )


        with gr.Tab("Substitution"):
            preset_sub = gr.Dropdown(
                label="Preset",
                choices=get_preset_names(),
                **_info_kwargs(gr.Dropdown, "Select preset for substitution"),
            )
            identity_img = gr.Image(
                label="Identity Image (char1)",
                type="pil",
                **_info_kwargs(gr.Image, "Upload reference image for identity"),
            )
            pose_img = gr.Image(
                label="Pose Image (char2)",
                type="pil",
                **_info_kwargs(gr.Image, "Upload pose image (OpenPose auto-extract if available)"),
            )
            sub_prompt = gr.Textbox(
                label="Prompt",
                lines=2,
                **_info_kwargs(gr.Textbox, "Extra description for substitution run"),
            )
            id_strength = gr.Slider(
                0.0,
                1.0,
                value=0.7,
                step=0.05,
                label="Identity Strength",
                **_info_kwargs(gr.Slider, "Blend ratio of identity image"),
            )
            pose_strength = gr.Slider(
                0.0,
                2.0,
                value=1.0,
                step=0.05,
                label="Pose Strength",
                **_info_kwargs(gr.Slider, "Strength of pose conditioning"),
            )
            sub_seed = gr.Number(
                label="Seed",
                value=42,
                precision=0,
                **_info_kwargs(gr.Number, "Seed for deterministic substitution"),
            )
            sub_btn = gr.Button(
                "Generate Substitution",
                **_info_kwargs(gr.Button, "Run identity→pose substitution"),
            )
            sub_output = gr.Image(label="Output")

            def _run_sub(
                preset_name: str,
                identity,
                pose,
                text_prompt: str,
                ids,
                poses,
                seed_value,
            ):
                preset_data = get_preset(preset_name) or {}
                miss = missing_assets(preset_data)
                missing_names = _try_auto_download(_missing_lora_filenames(miss))
                if missing_names:
                    raise gr.Error(_missing_assets_message(missing_names))
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
                [
                    preset_sub,
                    identity_img,
                    pose_img,
                    sub_prompt,
                    id_strength,
                    pose_strength,
                    sub_seed,
                ],
                [sub_output],
            )

        with gr.Tab("Pin Editor"):
            preset_pin = gr.Dropdown(
                label="Preset (optional)",
                choices=get_preset_names(),
                **_info_kwargs(gr.Dropdown, "Use preset's base model for inpaint"),
            )
            pin_base = gr.Image(
                label="Base Image",
                type="pil",
                **_info_kwargs(gr.Image, "Image to edit with targeted pins"),
            )
            pin_table = gr.Dataframe(
                headers=["x", "y", "label", "prompt"],
                row_count=(0, "dynamic"),
                label="Pins Table",
                interactive=True,
            )
            ref_img = gr.Image(label="Optional Reference Image", type="pil")
            radius = gr.Slider(
                8,
                128,
                value=32,
                step=1,
                label="Pin Radius",
                **_info_kwargs(gr.Slider, "Mask radius around each pin"),
            )
            apply_btn = gr.Button(
                "Apply Pin Edits",
                **_info_kwargs(gr.Button, "Run placeholder inpaint per pin"),
            )
            gallery = gr.Gallery(label="Pin Edit Results", columns=3)

            def _apply(preset_name, base_img, rows, ref_image, radius_value):
                if base_img is None or not rows:
                    return []
                pins: list[Pin] = []
                for row in rows:
                    try:
                        x, y, label, prompt_text = (
                            int(row[0]),
                            int(row[1]),
                            str(row[2]),
                            str(row[3]),
                        )
                        pins.append(
                            Pin(
                                x,
                                y,
                                label or "pin",
                                prompt_text or "",
                                ref_image,
                            )
                        )
                    except Exception:
                        continue

                def _editor_fn(img, mask, prompt_text, ref):  # placeholder implementation
                    return img

                return list(apply_pin_edits(base_img, pins, _editor_fn).values())

            apply_btn.click(
                _apply, [preset_pin, pin_base, pin_table, ref_img, radius], [gallery]
            )

        with gr.Tab("Reference Gallery"):
            gr.Markdown("(Placeholder) Thumbnails grid. Click to load as reference.")

        with gr.Tab("Downloads"):
            gr.Markdown("### Asset Downloads")
            if KNOWN_LORAS:
                known_list = "\n".join(
                    f"- **{name}** ← `{repo}`" for name, (repo, _file) in sorted(KNOWN_LORAS.items())
                )
                gr.Markdown("Known LoRA registry:\n" + known_list)
            missing_box = gr.Textbox(label="Detected Missing LoRAs", lines=6)
            scan_btn = gr.Button("Scan Presets for Missing LoRAs")
            dl_btn = gr.Button("Download All Missing")

            def _scan():
                loras = _scan_presets_for_loras()
                if not loras:
                    return "No LoRAs referenced in presets."
                if resolve_missing_loras is None:
                    return "Install huggingface_hub to scan/download."
                missing = resolve_missing_loras(loras)
                if not missing:
                    return "All referenced LoRAs are present."
                return "\n".join(missing)

            def _download_all(text: str):
                if not text or not text.strip():
                    return "Nothing to download."
                if download_lora is None:
                    return (
                        "Install huggingface_hub to download assets: pip install huggingface_hub"
                    )
                missing = [line.strip() for line in text.splitlines() if line.strip()]
                failed: list[str] = []
                for fname in missing:
                    try:
                        download_lora(fname)
                    except Exception as exc:  # pragma: no cover - depends on network availability
                        failed.append(f"{fname} — {exc}")
                if failed:
                    return "Some downloads failed:\n" + "\n".join(failed)
                return "All missing LoRAs downloaded."

            scan_btn.click(_scan, outputs=missing_box)
            dl_btn.click(_download_all, inputs=missing_box, outputs=missing_box)

    return demo


def studio() -> gr.Blocks:
    """Backwards compatible factory."""

    return build_ui()


if __name__ == "__main__":
    app = studio()
    try:  # pragma: no cover - guard is optional in production builds
        from chargen.ui_guard import check_ui

        for warning in check_ui(app):
            print(warning)
    except Exception as exc:  # pragma: no cover - optional dependency / environment guard
        print("[UI] Drift check skipped:", exc)
    app.launch(
        server_name=os.getenv("PCS_SERVER_NAME", "127.0.0.1"),
        server_port=int(os.getenv("PCS_PORT", "7860")),
    )

