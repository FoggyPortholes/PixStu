"""
Retro UI with Inpainting, Gallery, Downloads tabs.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import List

import gradio as gr

from ..core.inpaint import inpaint
from ..tools.device import pick_device
from ..tools.downloads import download_lora, have_lora
from ..tools.self_heal import self_heal
from ..tools.version import VERSION

GALLERY = Path("outputs/gallery")
GALLERY.mkdir(parents=True, exist_ok=True)


def _safe_stem(text: str | None) -> str:
    base = "" if text is None else "".join(c if c.isalnum() else "_" for c in text)
    base = base[:40]
    return base or datetime.now().strftime("%Y%m%d%H%M%S")


def _gallery() -> List[str]:
    return [
        str(path)
        for path in sorted(GALLERY.glob("*.png"), key=lambda p: p.stat().st_mtime, reverse=True)[:50]
    ]


@self_heal("scan_presets")
def _scan_presets() -> List[str]:
    cfg = Path(".pixstu/lora_sets.json")
    if not cfg.exists():
        return []
    try:
        data = json.loads(cfg.read_text(encoding="utf-8"))
        return [item["lora"] for item in data if isinstance(item, dict) and "lora" in item]
    except Exception:
        return []


def _run_inpaint(prompt, init, mask, steps, scale, seed):
    image, meta = inpaint(prompt, init, mask, steps, scale, seed)
    if image:
        stem = _safe_stem(prompt)
        dest = GALLERY / f"{stem}.png"
        image.save(dest)
    return image, _gallery()


def studio():
    with gr.Blocks(css=".retro-btn{font-family:monospace;background:#333;color:#fff;border:2px solid #ff66cc}") as demo:
        gr.Markdown(f"## 🕹️ PixStu v{VERSION} — Device: **{pick_device()}**")
        with gr.Tab("🎨 Inpainting"):
            prompt = gr.Textbox(label="Prompt")
            init = gr.Image(type="filepath", label="Init")
            mask = gr.Image(type="filepath", label="Mask")
            steps = gr.Slider(1, 100, 40, label="Steps")
            scale = gr.Slider(0, 20, 7.5, label="Guidance Scale")
            seed = gr.Number(label="Seed")
            run = gr.Button("▶️ Run", elem_classes="retro-btn")
            out = gr.Image(label="Output")
            gal = gr.Gallery(value=_gallery(), label="Recent")
            run.click(_run_inpaint, [prompt, init, mask, steps, scale, seed], [out, gal])
        with gr.Tab("🖼️ Gallery"):
            gallery = gr.Gallery(value=_gallery(), label="Saved")
            gr.Button("🔄 Refresh").click(_gallery, outputs=gallery)
        with gr.Tab("📥 Downloads"):
            scan = gr.Button("🔍 Scan Presets")
            download = gr.Button("⬇️ Download Missing")
            box = gr.Textbox(label="Missing LoRAs")

            def _missing():
                missing = [name for name in _scan_presets() if not have_lora(name)]
                return "\n".join(missing) or "All present"

            def _download(text: str):
                targets = [line.strip() for line in text.splitlines() if line.strip()]
                if not targets:
                    return "Nothing to download"
                for target in targets:
                    download_lora(target)
                return "Done"

            scan.click(_missing, outputs=box)
            download.click(_download, inputs=box, outputs=box)

    return demo


if __name__ == "__main__":  # pragma: no cover - manual launch
    studio().launch()
