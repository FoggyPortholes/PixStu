"""Retro-styled Gradio studio focused on PixStu inpainting."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Callable

import gradio as gr
from PIL import Image

from .inpaint import inpaint
from .metadata import save_metadata
from .reference_gallery import list_gallery as _list_gallery, save_to_gallery

try:  # pragma: no cover - optional dependency for downloads
    from tools.downloads import download_lora, resolve_missing_loras
except Exception:  # pragma: no cover - keep UI usable without huggingface_hub
    download_lora = None  # type: ignore[assignment]
    resolve_missing_loras = None  # type: ignore[assignment]

try:  # pragma: no cover - optional self-heal helper
    from tools.self_heal import self_heal
except Exception:  # pragma: no cover - fall back to no-op decorator
    def self_heal(_name: str) -> Callable:
        def deco(fn: Callable) -> Callable:
            return fn

        return deco


PIXSTU_AUTO_DOWNLOAD = os.environ.get("PIXSTU_AUTO_DOWNLOAD", "0").lower() in (
    "1",
    "true",
    "yes",
    "on",
)


def _save_with_meta(image: Image.Image, metadata: dict, label: str) -> str:
    """Persist ``image`` and metadata to the reference gallery."""

    saved_path = Path(save_to_gallery(image, label=label or "untitled"))
    meta_path = saved_path.with_suffix(".json")
    save_metadata(str(meta_path), metadata)
    return str(saved_path)


def list_gallery(limit: int = 64) -> list[str]:
    """Return a limited list of gallery images sorted by recency."""

    items = _list_gallery()
    if limit is None or limit <= 0:
        return items
    return items[:limit]


@self_heal("scan_presets")
def _scan_presets_for_loras() -> list[str]:
    preset_path = Path(".pixstu/lora_sets.json")
    discovered: set[str] = set()
    if not preset_path.exists():
        return []
    try:
        payload = json.loads(preset_path.read_text(encoding="utf-8"))
    except Exception:
        return []

    if isinstance(payload, list):
        for item in payload:
            if isinstance(item, dict) and "lora" in item:
                discovered.add(str(item["lora"]))
            elif isinstance(item, list):
                for sub in item:
                    if isinstance(sub, dict) and "lora" in sub:
                        discovered.add(str(sub["lora"]))
    return sorted(discovered)


def _try_auto_download(missing: list[str]) -> list[str]:
    if not PIXSTU_AUTO_DOWNLOAD or download_lora is None:
        return missing
    remaining: list[str] = []
    for name in missing:
        try:
            download_lora(name)
        except Exception:
            remaining.append(name)
    return remaining


@self_heal("inpaint_run")
def _run_inpaint(
    prompt: str,
    init_img: str,
    mask_img: str,
    steps: int,
    scale: float,
    seed: str | int | None,
    no_safety: bool,
):
    if not prompt or not init_img or not mask_img:
        raise gr.Error("Prompt, Init, and Mask required.")

    image, metadata = inpaint(
        prompt=prompt,
        init_image=init_img,
        mask_image=mask_img,
        steps=int(steps or 50),
        guidance_scale=float(scale or 7.5),
        seed=int(seed) if seed not in (None, "") else None,
        disable_safety_checker=bool(no_safety),
    )
    _save_with_meta(image, metadata, label=prompt)
    return image, list_gallery()


RETRO_THEME = {"primary_hue": "purple", "secondary_hue": "cyan", "neutral_hue": "gray"}


def studio() -> gr.Blocks:
    with gr.Blocks(
        theme=RETRO_THEME,
        css="""
        .retro-text textarea { font-family: monospace; background: #111; color: #f5f5f5; }
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
            prompt = gr.Textbox(label="Prompt", placeholder="Describe patch…", elem_classes="retro-text")
            with gr.Row():
                init = gr.Image(type="filepath", label="Init Image", elem_classes="retro-img")
                mask = gr.Image(type="filepath", label="Mask (white=inpaint)", elem_classes="retro-img")
            steps = gr.Slider(1, 100, value=50, step=1, label="Steps", elem_classes="retro-slider")
            scale = gr.Slider(0, 20, value=7.5, step=0.1, label="CFG Scale", elem_classes="retro-slider")
            seed = gr.Number(value=None, precision=0, label="Seed", elem_classes="retro-num")
            no_safety = gr.Checkbox(label="Disable Safety Checker", elem_classes="retro-check")
            run_btn = gr.Button("▶️ Run Inpaint", elem_classes="retro-btn")
            out = gr.Image(label="Output", elem_classes="retro-output")
            gallery = gr.Gallery(label="Reference Gallery", columns=[4], height="auto")
            run_btn.click(
                _run_inpaint,
                [prompt, init, mask, steps, scale, seed, no_safety],
                [out, gallery],
            )

        with gr.Tab("🖼️ Gallery"):
            gr.Markdown("## 🕹️ Reference Gallery")
            gallery_view = gr.Gallery(value=list_gallery(), columns=[6], height="auto")
            refresh = gr.Button("🔄 Refresh", elem_classes="retro-btn")
            refresh.click(lambda: list_gallery(), outputs=gallery_view)

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
                if resolve_missing_loras is None:
                    return "Install huggingface_hub to scan."
                loras = _scan_presets_for_loras()
                if not loras:
                    return "No LoRAs referenced."
                missing = resolve_missing_loras(loras)
                if missing:
                    missing = _try_auto_download(missing)
                return "All present." if not missing else "\n".join(missing)

            def _download_all(text: str) -> str:
                if not text or not text.strip():
                    return "Nothing to download."
                if download_lora is None:
                    return "Install huggingface_hub to download."
                missing = [line.strip() for line in text.splitlines() if line.strip()]
                failures: list[str] = []
                for filename in missing:
                    try:
                        download_lora(filename)
                    except Exception as exc:  # pragma: no cover - network dependent
                        failures.append(f"{filename}: {exc}")
                return "Downloaded." if not failures else "Failed:\n" + "\n".join(failures)

            scan_btn.click(_scan, outputs=missing_box)
            dl_btn.click(_download_all, inputs=missing_box, outputs=missing_box)

    return demo


if __name__ == "__main__":  # pragma: no cover - manual launch helper
    studio().launch()
