"""
Retro-modern UI with tabs: Inpainting, Img2Img, Txt2Img, Txt2GIF, Gallery, Downloads.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

try:  # optional dependency guardrail
    import gradio as gr
except ModuleNotFoundError as exc:  # pragma: no cover - exercised when gradio missing
    gr = None  # type: ignore[assignment]
    _GRADIO_IMPORT_ERROR = exc
else:
    _GRADIO_IMPORT_ERROR = None

from ..core.img2img import img2img
from ..core.inpaint import inpaint
from ..core.txt2img import txt2img
from ..core.txt2vid import txt2gif
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


def _save(image, prompt: str | None) -> None:
    if image is None:
        return
    stem = _safe_stem(prompt)
    target = GALLERY / f"{stem}.png"
    try:
        image.save(target)
    except Exception:
        pass


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
        payload = json.loads(cfg.read_text(encoding="utf-8"))
    except Exception:
        return []
    return [item["lora"] for item in payload if isinstance(item, dict) and "lora" in item]


def _run_txt2img(prompt: str):
    image, _ = txt2img(prompt)
    _save(image, prompt)
    return image


def _run_img2img(prompt: str, init: str):
    image, _ = img2img(prompt, init)
    _save(image, prompt)
    return image


def _run_inpaint(prompt: str, init: str, mask: str):
    image, _ = inpaint(prompt, init, mask)
    _save(image, prompt)
    return image


def _run_txt2gif(prompt: str) -> Tuple[object, str]:
    image, meta = txt2gif(prompt)
    _save(image, prompt)
    return image, meta.get("gif_b64", "")


def studio():
    if gr is None:  # pragma: no cover - requires missing dependency
        raise RuntimeError(
            "PixStu studio requires the optional dependency 'gradio'. Install it with "
            "`pip install -r requirements.txt` (or `pip install gradio`) before running the UI."
        ) from _GRADIO_IMPORT_ERROR

    with gr.Blocks(
        css=".retro-btn{font-family:monospace;background:#333;color:#fff;border:2px solid #ff66cc}"
    ) as demo:
        gr.Markdown(f"## 🕹️ PixStu v{VERSION} — Device: **{pick_device()}**")

        with gr.Tab("✍️ Txt2Img"):
            prompt = gr.Textbox(label="Prompt")
            run = gr.Button("▶️ Txt2Img", elem_classes="retro-btn")
            out = gr.Image(label="Output")
            run.click(_run_txt2img, inputs=prompt, outputs=out)

        with gr.Tab("🖌️ Img2Img"):
            prompt = gr.Textbox(label="Prompt")
            init = gr.Image(type="filepath", label="Init Image")
            run = gr.Button("▶️ Img2Img", elem_classes="retro-btn")
            out = gr.Image(label="Output")
            run.click(_run_img2img, inputs=[prompt, init], outputs=out)

        with gr.Tab("🎨 Inpainting"):
            prompt = gr.Textbox(label="Prompt")
            init = gr.Image(type="filepath", label="Init Image")
            mask = gr.Image(type="filepath", label="Mask")
            run = gr.Button("▶️ Inpaint", elem_classes="retro-btn")
            out = gr.Image(label="Output")
            run.click(_run_inpaint, inputs=[prompt, init, mask], outputs=out)

        with gr.Tab("🎞️ Txt2GIF"):
            prompt = gr.Textbox(label="Prompt")
            run = gr.Button("▶️ Gif", elem_classes="retro-btn")
            preview = gr.Image(label="Preview")
            payload = gr.Textbox(label="GIF (base64)")
            run.click(_run_txt2gif, inputs=prompt, outputs=[preview, payload])

        with gr.Tab("🖼️ Gallery"):
            gallery = gr.Gallery(value=_gallery(), label="Recent")
            gr.Button("🔄 Refresh", elem_classes="retro-btn").click(_gallery, outputs=gallery)

        with gr.Tab("📥 Downloads"):
            scan = gr.Button("🔍 Scan Presets", elem_classes="retro-btn")
            download = gr.Button("⬇️ Download Missing", elem_classes="retro-btn")
            box = gr.Textbox(label="Missing LoRAs", lines=8)

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
