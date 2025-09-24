"""Simplified Gradio studio with retro-modern styling and downloads tab."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Callable, Iterable

import gradio as gr

from chargen.generator import BulletProofGenerator
from chargen.presets import get_preset, missing_assets

try:  # pragma: no cover - optional dependency for downloads
    from tools.downloads import download_lora, resolve_missing_loras, KNOWN_LORAS
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


def _missing_assets_message(missing: list[str]) -> str:
    lines = ["⚠️ Missing assets:"]
    for name in missing:
        lines.append(f"• loras/{name}")
    lines.append("\n➡️ Fix:")
    lines.append("1. Use the *Downloads* tab to fetch assets.")
    lines.append("2. Or set PIXSTU_AUTO_DOWNLOAD=1 to fetch automatically.")
    lines.append("3. Or remove the preset referencing these assets.")
    return "\n".join(lines)


def _try_auto_download(missing: list[str]) -> list[str]:
    if not PIXSTU_AUTO_DOWNLOAD or download_lora is None:
        return missing
    still_missing: list[str] = []
    for fname in missing:
        try:
            download_lora(fname)
        except Exception:  # pragma: no cover - depends on network availability
            still_missing.append(fname)
    return still_missing


def _scan_presets_for_loras() -> list[str]:
    preset_path = Path(".pixstu/lora_sets.json")
    found: set[str] = set()
    if preset_path.exists():
        try:
            data = json.loads(preset_path.read_text())
        except Exception:  # pragma: no cover - malformed custom config shouldn't crash UI
            data = []
        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict) and "lora" in item:
                    found.add(str(item["lora"]))
                elif isinstance(item, list):
                    for sub in item:
                        if isinstance(sub, dict) and "lora" in sub:
                            found.add(str(sub["lora"]))
    return sorted(found)


RETRO_THEME = {"primary_hue": "purple", "secondary_hue": "cyan", "neutral_hue": "gray"}


def _stub(*_args, **_kwargs):  # pragma: no cover - UI placeholder
    return None


def _missing_lora_filenames(entries: Iterable) -> list[str]:
    names: set[str] = set()
    for entry in entries:
        if isinstance(entry, dict):
            path = (
                entry.get("resolved_path")
                or entry.get("display_path")
                or entry.get("path")
                or ""
            )
        else:
            path = str(entry or "")
        if not path:
            continue
        names.add(Path(path).name)
    return sorted(names)


def _quick_render(preset_name: str, lora_path: str, strength: float):
    preset = get_preset(preset_name) or {}
    preset.setdefault("loras", [])

    expected_path = Path(lora_path)
    missing = missing_assets(preset)
    missing_paths = {
        Path(asset.get("resolved_path") or asset.get("path") or "")
        for asset in missing
        if asset.get("resolved_path") or asset.get("path")
    }

    if expected_path in missing_paths or not expected_path.exists():
        missing_name = expected_path.name
        message = (
            "Selected LoRA is missing: "
            f"{missing_name} (expected at {expected_path.as_posix()})"
        )
        raise gr.Error(message)

    selected_entry = None
    for entry in preset.get("loras", []):
        candidate = entry.get("resolved_path") or entry.get("path")
        if candidate and Path(candidate) == expected_path:
            selected_entry = entry
            break

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


def studio(on_inpaint: Callable | None = None) -> gr.Blocks:
    """Build the retro-modern PixStu studio."""

    inpaint_cb = on_inpaint or _stub

    with gr.Blocks(
        theme=RETRO_THEME,
        css="""
        .retro-text textarea { font-family: monospace; background: #1a1a1a; color: #f5f5f5; }
        .retro-img { border: 2px solid #6c63ff; }
        .retro-slider input { accent-color: #ff66cc; }
        .retro-num input { font-family: monospace; background: #222; color: #0ff; }
        .retro-check input { accent-color: #ffcc00; }
        .retro-btn { font-family: monospace; background: #333; color: #fff; border: 2px solid #ff66cc; }
        .retro-btn:hover { background: #ff66cc; color: #000; }
        .retro-output img { border: 4px solid #0ff; }
        .retro-box textarea { background: #111; color: #ffcc00; font-family: monospace; }
    """,
    ) as demo:
        with gr.Tab("🎨 Inpainting"):
            gr.Markdown("## 🕹️ Pixel Inpainting")
            prompt = gr.Textbox(label="Prompt", elem_classes="retro-text")
            init = gr.Image(type="filepath", label="Init Image", elem_classes="retro-img")
            mask = gr.Image(
                type="filepath",
                label="Mask (white=inpaint)",
                elem_classes="retro-img",
            )
            steps = gr.Slider(
                1,
                100,
                value=50,
                step=1,
                label="Steps",
                elem_classes="retro-slider",
            )
            scale = gr.Slider(
                0.0,
                20.0,
                value=7.5,
                step=0.1,
                label="CFG Scale",
                elem_classes="retro-slider",
            )
            seed = gr.Number(value=None, precision=0, label="Seed", elem_classes="retro-num")
            no_safety = gr.Checkbox(label="Disable Safety Checker", elem_classes="retro-check")
            run_btn = gr.Button("▶️ Run Inpaint", elem_classes="retro-btn")
            out = gr.Image(label="Output", elem_classes="retro-output")
            gallery = gr.Gallery(label="Reference Gallery").style(grid=[4], height="auto")
            run_btn.click(
                inpaint_cb,
                inputs=[prompt, init, mask, steps, scale, seed, no_safety],
                outputs=[out, gallery],
            )

        with gr.Tab("🖼️ Gallery"):
            gr.Markdown("## 🕹️ Reference Gallery")
            gallery = gr.Gallery(value=[]).style(grid=[6], height="auto")
            refresh = gr.Button("🔄 Refresh", elem_classes="retro-btn")
            refresh.click(lambda: [], outputs=gallery)

        with gr.Tab("📥 Downloads"):
            gr.Markdown("## 🕹️ Manage Assets")
            scan_btn = gr.Button("🔍 Scan Presets", elem_classes="retro-btn")
            dl_btn = gr.Button("⬇️ Download Missing", elem_classes="retro-btn")
            missing_box = gr.Textbox(
                label="Missing LoRAs",
                lines=6,
                interactive=False,
                elem_classes="retro-box",
            )

            def _scan() -> str:
                loras = _scan_presets_for_loras()
                if not loras:
                    return "No LoRAs referenced in presets."
                if resolve_missing_loras is None:
                    return "Install huggingface_hub to scan/download."
                missing = resolve_missing_loras(loras)
                if not missing:
                    return "All referenced LoRAs are present."
                names = _missing_lora_filenames(missing)
                remaining = _try_auto_download(list(missing))
                if remaining != list(missing):
                    if not remaining:
                        return "All referenced LoRAs downloaded automatically."
                    auto_msg = "Auto-downloaded some assets. Still missing:\n"
                    return auto_msg + "\n".join(_missing_lora_filenames(remaining))
                if names:
                    return "\n".join(names)
                return "\n".join(_missing_lora_filenames(remaining))

            def _download_all(text: str) -> str:
                if not text.strip():
                    return "Nothing to download."
                if download_lora is None:
                    return "Install huggingface_hub to download assets."
                missing = [line.strip() for line in text.splitlines() if line.strip()]
                failed: list[str] = []
                for fname in missing:
                    try:
                        download_lora(fname)
                    except Exception as exc:  # pragma: no cover - depends on network availability
                        failed.append(f"{fname} — {exc}")
                return (
                    "All missing LoRAs downloaded."
                    if not failed
                    else "Failed:\n" + "\n".join(failed)
                )

            scan_btn.click(_scan, outputs=missing_box)
            dl_btn.click(_download_all, inputs=missing_box, outputs=missing_box)

    return demo


__all__ = [
    "studio",
    "_missing_assets_message",
    "_try_auto_download",
    "_scan_presets_for_loras",
    "_missing_lora_filenames",
    "_quick_render",
]

