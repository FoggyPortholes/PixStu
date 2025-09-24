import gradio as gr
from pathlib import Path
from datetime import datetime
from PIL import Image

from .inpaint import inpaint

GALLERY_DIR = Path("outputs/gallery")
META = GALLERY_DIR / "index.json"
GALLERY_DIR.mkdir(parents=True, exist_ok=True)

# Lightweight metadata index (no heavy deps)
import json

def _load_index():
    if META.exists():
        try:
            return json.loads(META.read_text())
        except Exception:
            return []
    return []

def _save_index(items):
    META.write_text(json.dumps(items, indent=2))


def save_to_gallery(img: Image.Image, label: str, prompt: str) -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in label)[:40]
    path = GALLERY_DIR / f"{ts}_{safe}.png"
    img.save(path)

    items = _load_index()
    items.insert(0, {
        "file": str(path),
        "label": label,
        "prompt": prompt,
        "created": ts,
        "type": "inpaint"
    })
    _save_index(items)
    return str(path)


def list_gallery():
    items = _load_index()
    return [i["file"] for i in items]


# Gradio App

def studio():
    with gr.Blocks(title="PixStu Studio") as demo:
        gr.Markdown("# 🎨 PixStu Studio — Generation + Reference Gallery")

        with gr.Tab("Inpainting"):
            with gr.Row():
                prompt = gr.Textbox(label="Prompt", info="Describe the patch to generate")
                steps = gr.Slider(4, 100, value=50, step=1, label="Steps")
                guidance = gr.Slider(1.0, 12.0, value=7.5, step=0.5, label="CFG Scale")
                threshold = gr.Slider(0, 255, value=128, step=1, label="Mask Threshold", info="Binarize mask; leave as-is if you want soft edges")
                seed = gr.Number(value=None, precision=0, label="Seed (optional)")
            with gr.Row():
                init = gr.Image(type="filepath", label="Init Image")
                mask = gr.Image(type="filepath", label="Mask Image (white = fill)")
            with gr.Row():
                run_btn = gr.Button("Run Inpainting")
                out = gr.Image(label="Output")
            with gr.Row():
                gallery = gr.Gallery(label="Reference Gallery", show_label=True).style(grid=[4], height="auto")

            def _run(p, i, m, s, g, t, sd):
                img = inpaint(p, i, m, steps=int(s), guidance_scale=float(g), threshold=int(t) if t is not None else None, seed=int(sd) if sd else None)
                save_to_gallery(img, label=p, prompt=p)
                return img, list_gallery()

            run_btn.click(_run, inputs=[prompt, init, mask, steps, guidance, threshold, seed], outputs=[out, gallery])

        with gr.Tab("Reference Gallery"):
            gallery2 = gr.Gallery(label="All Items").style(grid=[6], height="auto")
            refresh = gr.Button("Refresh Gallery")
            refresh.click(lambda: list_gallery(), outputs=[gallery2])

    return demo


if __name__ == "__main__":
    studio().launch()
