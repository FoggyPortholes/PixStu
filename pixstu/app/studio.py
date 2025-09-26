import json, base64
from pathlib import Path
from datetime import datetime
import gradio as gr

from ..core.inpaint import inpaint
from ..core.img2img import img2img
from ..core.txt2img import txt2img
from ..core.txt2vid import txt2gif
from ..tools.device import pick_device
from ..tools.downloads import download_lora, have_lora
from ..tools.version import VERSION

GALLERY = Path("outputs/gallery"); GALLERY.mkdir(parents=True, exist_ok=True)

def _save_output(img, meta: dict):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    fname = f"{ts}.png"
    fpath = GALLERY / fname
    img.save(fpath)
    (GALLERY / f"{ts}.json").write_text(json.dumps(meta, indent=2))
    return str(fpath)

def _gallery():
    return [str(p) for p in sorted(GALLERY.glob("*.png"), key=lambda x: x.stat().st_mtime, reverse=True)[:50]]

def studio():
    css = ".retro-btn{font-family:monospace;background:#333;color:#fff;border:2px solid #ff66cc}" \
          ".retro-img{border:2px solid #6c63ff}"
    with gr.Blocks(css=css) as demo:
        gr.Markdown(f"## 🕹️ PixStu v{VERSION} — Device: **{pick_device()}**")

        def safe_call(fn, fallback):
            def _wrap(*args, **kwargs):
                try:
                    return fn(*args, **kwargs)
                except Exception as e:
                    return fallback(e)

            return _wrap

        # Txt2Img
        with gr.Tab("✍️ Txt2Img"):
            t_prompt = gr.Textbox(label="Prompt")
            t_run = gr.Button("▶️ Run", elem_classes="retro-btn")
            t_out = gr.Image(label="Output", elem_classes="retro-img")
            t_gallery = gr.Gallery(value=_gallery())
            def _txt2img(p):
                img, meta = txt2img(p)
                _save_output(img, meta)
                return img, _gallery()

            def _txt2img_error(e):
                gr.Warning(f"Error: {e}")
                return None, _gallery()

            t_run.click(safe_call(_txt2img, _txt2img_error), inputs=t_prompt, outputs=[t_out, t_gallery])

        # Img2Img
        with gr.Tab("🖌️ Img2Img"):
            i_prompt = gr.Textbox(label="Prompt")
            i_init = gr.Image(type="filepath", label="Init")
            i_run = gr.Button("▶️ Run", elem_classes="retro-btn")
            i_out = gr.Image(label="Output", elem_classes="retro-img")
            i_gallery = gr.Gallery(value=_gallery())
            def _img2img(p, init):
                img, meta = img2img(p, init)
                _save_output(img, meta)
                return img, _gallery()

            def _img2img_error(e):
                gr.Warning(f"Error: {e}")
                return None, _gallery()

            i_run.click(
                safe_call(_img2img, _img2img_error), inputs=[i_prompt, i_init], outputs=[i_out, i_gallery]
            )

        # Inpainting
        with gr.Tab("🎨 Inpainting"):
            p_prompt = gr.Textbox(label="Prompt")
            p_init = gr.Image(type="filepath", label="Init")
            p_mask = gr.Image(type="filepath", label="Mask")
            p_run = gr.Button("▶️ Run", elem_classes="retro-btn")
            p_out = gr.Image(label="Output", elem_classes="retro-img")
            p_gallery = gr.Gallery(value=_gallery())
            def _inpaint(pr, ii, mi):
                img, meta = inpaint(pr, ii, mi)
                _save_output(img, meta)
                return img, _gallery()

            def _inpaint_error(e):
                gr.Warning(f"Error: {e}")
                return None, _gallery()

            p_run.click(
                safe_call(_inpaint, _inpaint_error), inputs=[p_prompt, p_init, p_mask], outputs=[p_out, p_gallery]
            )

        # Txt2GIF
        with gr.Tab("🎞️ Txt2GIF"):
            v_prompt = gr.Textbox(label="Prompt")
            v_run = gr.Button("▶️ Run", elem_classes="retro-btn")
            v_prev = gr.Image(label="Preview")
            v_b64 = gr.Textbox(label="GIF (base64)", lines=4)
            def _gif(p):
                img, meta = txt2gif(p)
                return img, meta.get("gif_b64", "")

            def _gif_error(e):
                gr.Warning(f"Error: {e}")
                return None, f"Error: {e}"

            v_run.click(safe_call(_gif, _gif_error), inputs=v_prompt, outputs=[v_prev, v_b64])

        # Gallery
        with gr.Tab("🖼️ Gallery"):
            g = gr.Gallery(value=_gallery())
            gr.Button("🔄 Refresh", elem_classes="retro-btn").click(lambda: _gallery(), outputs=g)

        # Downloads
        with gr.Tab("📥 Downloads"):
            d_file = gr.Textbox(label="LoRA filename")
            d_btn = gr.Button("⬇️ Download", elem_classes="retro-btn")
            d_status = gr.Textbox(label="Status")
            def _dl(x):
                if not x: return "Enter a filename."
                if have_lora(x): return f"Already present: {x}"
                try:
                    download_lora(x)
                    return f"Downloaded: {x}"
                except Exception as e:
                    return f"Error: {e}"
            d_btn.click(_dl, inputs=d_file, outputs=d_status)

    return demo

if __name__ == "__main__":
    studio().launch()
