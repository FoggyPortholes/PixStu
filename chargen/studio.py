from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import gradio as gr
from PIL import Image

from .generator import BulletProofGenerator
from .inpaint import inpaint
from .model_setup import LORAS
from .presets import get_preset, load_presets, missing_assets
from .reference_gallery import list_gallery, save_to_gallery
from tools.download_manager import DownloadManager

RETRO_THEME = {
    "primary_hue": "purple",
    "secondary_hue": "cyan",
    "neutral_hue": "gray",
}


def retro_markdown(title: str) -> str:
    return f"## 🕹️ {title}"


_DOWNLOAD_MANAGER = DownloadManager(target_dir=LORAS)


def _collect_missing_assets() -> Dict[str, Dict[str, Any]]:
    """Return aggregated missing LoRA metadata keyed by resolved path."""

    catalog: Dict[str, Dict[str, Any]] = {}
    for preset_name, preset in load_presets().items():
        for asset in missing_assets(preset):
            key = asset.get("resolved_path") or asset.get("display_path") or asset.get("download")
            if not key:
                key = f"{preset_name}:{len(catalog)}"
            record = catalog.setdefault(
                key,
                {
                    "display_path": asset.get("display_path", ""),
                    "resolved_path": asset.get("resolved_path", ""),
                    "download": asset.get("download"),
                    "size_gb": asset.get("size_gb"),
                    "presets": set(),
                },
            )
            record["presets"].add(preset_name)
            if not record["display_path"]:
                record["display_path"] = asset.get("display_path", "")
            if not record["resolved_path"]:
                record["resolved_path"] = asset.get("resolved_path", "")
    return catalog


def _format_missing_report(records: Dict[str, Dict[str, Any]]) -> str:
    if not records:
        return "All presets ready. No missing LoRAs."

    lines: list[str] = []
    for payload in records.values():
        display = payload.get("display_path") or payload.get("resolved_path") or "Unknown LoRA"
        size = payload.get("size_gb")
        size_text = ""
        if isinstance(size, (int, float)) and size:
            size_text = f" (~{float(size):.2f} GB)"
        presets = ", ".join(sorted(payload.get("presets", [])))
        lines.append(f"{display}{size_text}")
        if presets:
            lines.append(f"  ↳ Presets: {presets}")
        url = payload.get("download")
        if url:
            lines.append(f"  ↳ {url}")
        lines.append("")
    return "\n".join(lines).strip()


def _scan() -> str:
    """Scan presets and return a retro-styled missing asset report."""

    records = _collect_missing_assets()
    return _format_missing_report(records)


def _download_all(current_report: str) -> str:
    """Queue downloads for all missing LoRAs with known URLs and refresh report."""

    records = _collect_missing_assets()
    if not records:
        return "All presets ready. No missing LoRAs."

    queued: list[str] = []
    for payload in records.values():
        url = payload.get("download")
        if not url:
            continue
        resolved = payload.get("resolved_path") or ""
        display = payload.get("display_path") or resolved
        filename = Path(resolved or display).name
        if not filename:
            continue
        size = payload.get("size_gb")
        try:
            size_value = float(size) if size is not None else 0.0
        except (TypeError, ValueError):
            size_value = 0.0
        _DOWNLOAD_MANAGER.add(url, filename, size_value)
        queued.append(filename)
    if queued:
        _DOWNLOAD_MANAGER.run_async()
    report = _format_missing_report(records)
    if queued:
        report = f"{report}\n\nQueued downloads: {', '.join(queued)}"
    elif not any(payload.get("download") for payload in records.values()):
        report = f"{report}\n\nNo downloadable LoRAs found."
    return report


def _quick_render(preset_name: str, selected_lora: str, strength: float):
    """Generate a quick preview for a preset while tolerating missing LoRAs."""

    preset = get_preset(preset_name) or {}
    if selected_lora:
        selected_path = Path(selected_lora)
        if not selected_path.exists():
            name = selected_path.name or selected_lora
            raise gr.Error(f"Missing LoRA: {name} ({selected_path})")
    generator = BulletProofGenerator(preset)
    return generator.generate("LoRA quick preview", 0)


def run_inpaint(
    prompt: str,
    init_path: str | None,
    mask_path: str | None,
    steps: int | float,
    scale: float,
    seed: float | int | None,
    disable_safety: bool,
):
    """Execute the inpainting pipeline and refresh the gallery."""

    if not prompt or not prompt.strip():
        raise gr.Error("Describe the patch before running inpaint.")
    if not init_path:
        raise gr.Error("Please provide an init image.")
    if not mask_path:
        raise gr.Error("Please provide a mask image.")

    try:
        step_value = int(steps)
    except (TypeError, ValueError):
        raise gr.Error("Steps must be an integer value.") from None
    try:
        scale_value = float(scale)
    except (TypeError, ValueError):
        raise gr.Error("CFG scale must be numeric.") from None

    seed_value: int | None
    if seed in (None, ""):
        seed_value = None
    else:
        try:
            seed_value = int(seed)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            raise gr.Error("Seed must be an integer.") from None

    try:
        result: Image.Image = inpaint(
            prompt=prompt.strip(),
            init_image=init_path,
            mask_image=mask_path,
            guidance_scale=scale_value,
            steps=step_value,
            seed=seed_value,
            disable_safety_checker=bool(disable_safety),
        )
    except FileNotFoundError as exc:
        raise gr.Error(str(exc)) from exc
    except Exception as exc:  # pragma: no cover - runtime safeguard
        raise gr.Error(f"Inpaint failed: {exc}") from exc

    try:
        save_to_gallery(result, label=prompt)
    except Exception:  # pragma: no cover - gallery IO best-effort
        pass

    return result, list_gallery()


def studio() -> gr.Blocks:
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
            gr.Markdown(retro_markdown("Pixel Inpainting"))
            gr.Markdown("Fill areas of an image with new pixel art, keeping backgrounds blank.")

            with gr.Row():
                prompt = gr.Textbox(label="Prompt", placeholder="Describe patch…", elem_classes="retro-text")
            with gr.Row():
                init = gr.Image(type="filepath", label="Init Image", elem_classes="retro-img")
                mask = gr.Image(type="filepath", label="Mask (white=inpaint)", elem_classes="retro-img")
            with gr.Row():
                steps = gr.Slider(1, 100, value=50, step=1, label="Steps", elem_classes="retro-slider")
                scale = gr.Slider(0.0, 20.0, value=7.5, step=0.1, label="CFG Scale", elem_classes="retro-slider")
                seed = gr.Number(value=None, precision=0, label="Seed", elem_classes="retro-num")
            with gr.Row():
                no_safety = gr.Checkbox(label="Disable Safety Checker", elem_classes="retro-check")
            with gr.Row():
                run_btn = gr.Button("▶️ Run Inpaint", elem_classes="retro-btn")
            out = gr.Image(label="Output", elem_classes="retro-output")
            gallery = gr.Gallery(label="Reference Gallery").style(grid=[4], height="auto")
            run_btn.click(
                run_inpaint,
                inputs=[prompt, init, mask, steps, scale, seed, no_safety],
                outputs=[out, gallery],
            )

        with gr.Tab("🖼️ Gallery"):
            gr.Markdown(retro_markdown("Reference Gallery"))
            gr.Markdown("Browse your saved pixel characters. Latest first.")
            g = gr.Gallery(value=list_gallery()).style(grid=[6], height="auto")
            refresh = gr.Button("🔄 Refresh", elem_classes="retro-btn")
            refresh.click(lambda: list_gallery(), inputs=None, outputs=g)

        with gr.Tab("📥 Downloads"):
            gr.Markdown(retro_markdown("Manage Assets"))
            gr.Markdown("Scan your presets for missing LoRAs and fetch them automatically.")

            with gr.Row():
                scan_btn = gr.Button("🔍 Scan Presets", elem_classes="retro-btn")
                dl_btn = gr.Button("⬇️ Download Missing", elem_classes="retro-btn")
            missing_box = gr.Textbox(label="Missing LoRAs", lines=6, interactive=False, elem_classes="retro-box")

            scan_btn.click(_scan, inputs=None, outputs=missing_box)
            dl_btn.click(_download_all, inputs=missing_box, outputs=missing_box)

    return demo


if __name__ == "__main__":
    studio().launch()
