import json, os
from pathlib import Path
from datetime import datetime
import gradio as gr

from ..core.inpaint import inpaint
from ..core.img2img import img2img
from ..core.txt2img import txt2img
from ..core.txt2vid import txt2gif
from ..core.presets import (
    load_presets,
    save_user_presets,
    synthesize_prompt,
    apply_preset_to_params,
    preset_trait_options,
)
from ..tools.device import pick_device
from ..tools.downloads import download_lora, have_lora
from ..tools.version import VERSION

GALLERY = Path("outputs/gallery"); GALLERY.mkdir(parents=True, exist_ok=True)

STRICT = os.environ.get("PIXSTU_STRICT", "0") == "1"


def _save_output(img, meta: dict):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    fname = f"{ts}.png"
    fpath = GALLERY / fname
    img.save(fpath)
    (GALLERY / f"{ts}.json").write_text(json.dumps(meta, indent=2))
    return str(fpath)


def _gallery():
    return [str(p) for p in sorted(GALLERY.glob("*.png"), key=lambda x: x.stat().st_mtime, reverse=True)[:100]]


def _default_trait_setup(preset):
    trait_opts = preset_trait_options(preset) if preset else {}
    keys = sorted(trait_opts.keys())
    def pick(idx: int) -> tuple[str, list[str]]:
        if idx < len(keys):
            key = keys[idx]
            return key, trait_opts.get(key, [])
        return "", []
    k1, c1 = pick(0)
    k2, c2 = pick(1)
    k3, c3 = pick(2)
    v1 = c1[0] if c1 else None
    v2 = c2[0] if c2 else None
    v3 = c3[0] if c3 else None
    traits = {}
    if k1 and v1:
        traits[k1] = v1
    if k2 and v2:
        traits[k2] = v2
    if k3 and v3:
        traits[k3] = v3
    return (k1, c1, v1, k2, c2, v2, k3, c3, v3, traits)


def studio():
    presets = load_presets()
    preset_names = [p["name"] for p in presets] or ["<no presets found>"]
    first_preset = presets[0] if presets else None
    params = apply_preset_to_params(first_preset) if first_preset else {
        "prompt": "",
        "negative": "",
        "steps": 28,
        "cfg_scale": 7.0,
        "width": 640,
        "height": 640,
    }
    style_default = first_preset.get("style", "") if first_preset else ""
    k1, c1, v1, k2, c2, v2, k3, c3, v3, trait_defaults = _default_trait_setup(first_preset)
    synth_default = synthesize_prompt(params["prompt"], style_default, trait_defaults)
    user_path = Path(".pixstu/presets.json")
    if user_path.exists():
        user_json_default = user_path.read_text(encoding="utf-8")
    else:
        user_json_default = "[]"

    css = ".retro-btn{font-family:monospace;background:#333;color:#fff;border:2px solid #ff66cc}" \
          ".retro-img{border:2px solid #6c63ff} .retro-box textarea{font-family:monospace;background:#0f0f0f;color:#ffcc00}"
    with gr.Blocks(css=css) as demo:
        gr.Markdown(
            f"## 🕹️ PixStu v{VERSION} — Device: **{pick_device()}**  |  Strict Guardrails: {'ON' if STRICT else 'OFF'}"
        )

        with gr.Tab("🧩 Presets & Generator"):
            dd = gr.Dropdown(
                choices=preset_names,
                label="Preset",
                value=(preset_names[0] if preset_names else None),
            )
            style = gr.Textbox(label="Style (tag)", value=style_default)
            trait1_key = gr.Textbox(label="Trait Key 1", value=(k1 or "role"))
            trait1_val = gr.Dropdown(label="Trait 1 Value", choices=c1, value=v1)
            trait2_key = gr.Textbox(label="Trait Key 2", value=(k2 or "palette"))
            trait2_val = gr.Dropdown(label="Trait 2 Value", choices=c2, value=v2)
            trait3_key = gr.Textbox(label="Trait Key 3", value=(k3 or "pose"))
            trait3_val = gr.Dropdown(label="Trait 3 Value", choices=c3, value=v3)

            base_prompt = gr.Textbox(label="Base Prompt", lines=3, value=params["prompt"])
            neg_prompt = gr.Textbox(label="Negative Prompt", lines=2, value=params.get("negative", ""))
            steps = gr.Slider(1, 100, params["steps"], step=1, label="Steps")
            cfg = gr.Slider(0.0, 20.0, params["cfg_scale"], step=0.1, label="CFG Scale")
            w = gr.Slider(256, 1024, params["width"], step=64, label="Width")
            h = gr.Slider(256, 1024, params["height"], step=64, label="Height")

            live_prompt = gr.Textbox(
                label="Synthesized Prompt (read-only)",
                interactive=False,
                value=synth_default,
                elem_classes=["retro-box"],
            )

            apply_btn = gr.Button("📥 Apply to Generators", elem_classes=["retro-btn"])
            quick = gr.Button("⚡ Quick Preview (low steps)", elem_classes=["retro-btn"])
            preview = gr.Image(label="Preview", elem_classes=["retro-img"])

            preset_editor = gr.Textbox(
                label="User Presets JSON",
                lines=8,
                value=user_json_default,
            )
            save_btn = gr.Button("💾 Save User Presets", elem_classes=["retro-btn"])
            save_status = gr.Textbox(label="Preset Save Status", interactive=False)

        with gr.Tab("✍️ Txt2Img"):
            t_prompt = gr.Textbox(label="Prompt", value=synth_default)
            t_steps = gr.Slider(1, 100, params["steps"], label="Steps")
            t_scale = gr.Slider(0.0, 20.0, params["cfg_scale"], step=0.1, label="CFG Scale")
            t_w = gr.Slider(256, 1024, params["width"], step=64, label="Width")
            t_h = gr.Slider(256, 1024, params["height"], step=64, label="Height")
            t_seed = gr.Number(value=None, precision=0, label="Seed")
            t_run = gr.Button("▶️ Run", elem_classes="retro-btn")
            t_out = gr.Image(label="Output", elem_classes="retro-img")
            t_gallery = gr.Gallery(value=_gallery())

        with gr.Tab("🖌️ Img2Img"):
            i_prompt = gr.Textbox(label="Prompt", value=synth_default)
            i_init = gr.Image(type="filepath", label="Init Image", elem_classes=["retro-img"])
            i_strength = gr.Slider(0.1, 1.0, 0.65, step=0.05, label="Strength")
            i_steps = gr.Slider(1, 100, params["steps"], label="Steps")
            i_scale = gr.Slider(0.0, 20.0, params["cfg_scale"], step=0.1, label="CFG Scale")
            i_seed = gr.Number(value=None, precision=0, label="Seed")
            i_run = gr.Button("▶️ Run", elem_classes="retro-btn")
            i_out = gr.Image(label="Output", elem_classes=["retro-img"])

        with gr.Tab("🎨 Inpainting"):
            p_prompt = gr.Textbox(label="Prompt", value=synth_default)
            p_init = gr.Image(type="filepath", label="Init Image", elem_classes=["retro-img"])
            p_mask = gr.Image(type="filepath", label="Mask (white=inpaint)", elem_classes=["retro-img"])
            p_steps = gr.Slider(1, 100, params["steps"], label="Steps")
            p_scale = gr.Slider(0.0, 20.0, params["cfg_scale"], step=0.1, label="CFG Scale")
            p_seed = gr.Number(value=None, precision=0, label="Seed")
            p_run = gr.Button("▶️ Run", elem_classes="retro-btn")
            p_out = gr.Image(label="Output", elem_classes=["retro-img"])

        with gr.Tab("🎞️ Txt2GIF"):
            v_prompt = gr.Textbox(label="Prompt")
            v_frames = gr.Slider(4, 48, 12, step=1, label="Frames")
            v_ms = gr.Slider(50, 400, 100, step=10, label="Frame Duration (ms)")
            v_seed = gr.Number(value=None, precision=0, label="Seed")
            v_run = gr.Button("▶️ Generate GIF", elem_classes="retro-btn")
            v_preview = gr.Image(label="Preview (first frame)", elem_classes=["retro-img"])
            v_b64 = gr.Textbox(label="GIF (base64)", lines=4, interactive=False)

        with gr.Tab("🖼️ Gallery"):
            g = gr.Gallery(value=_gallery()).style(grid=[6], height="auto")
            gr.Button("🔄 Refresh", elem_classes="retro-btn").click(lambda: _gallery(), outputs=g)

        with gr.Tab("📥 Downloads"):
            d_file = gr.Textbox(label="LoRA filename")
            d_btn = gr.Button("⬇️ Download", elem_classes="retro-btn")
            d_status = gr.Textbox(label="Status")

        def on_preset_change(name: str):
            ps = next((p for p in load_presets() if p["name"] == name), None)
            if not ps:
                return (
                    "",
                    "",
                    28,
                    7.0,
                    640,
                    640,
                    "",
                    "role",
                    gr.update(choices=[]),
                    "palette",
                    gr.update(choices=[]),
                    "pose",
                    gr.update(choices=[]),
                    "",
                    "",
                    28,
                    7.0,
                    640,
                    640,
                    "",
                    28,
                    7.0,
                    "",
                    28,
                    7.0,
                )
            params_local = apply_preset_to_params(ps)
            style_local = ps.get("style", "")
            k1l, c1l, v1l, k2l, c2l, v2l, k3l, c3l, v3l, trait_map = _default_trait_setup(ps)
            synth = synthesize_prompt(params_local["prompt"], style_local, trait_map)
            return (
                params_local["prompt"],
                params_local.get("negative", ""),
                params_local["steps"],
                params_local["cfg_scale"],
                params_local["width"],
                params_local["height"],
                style_local,
                k1l or "role",
                gr.update(choices=c1l, value=v1l),
                k2l or "palette",
                gr.update(choices=c2l, value=v2l),
                k3l or "pose",
                gr.update(choices=c3l, value=v3l),
                synth,
                synth,
                params_local["steps"],
                params_local["cfg_scale"],
                params_local["width"],
                params_local["height"],
                synth,
                params_local["steps"],
                params_local["cfg_scale"],
                synth,
                params_local["steps"],
                params_local["cfg_scale"],
            )

        dd.change(
            on_preset_change,
            inputs=dd,
            outputs=[
                base_prompt,
                neg_prompt,
                steps,
                cfg,
                w,
                h,
                style,
                trait1_key,
                trait1_val,
                trait2_key,
                trait2_val,
                trait3_key,
                trait3_val,
                live_prompt,
                t_prompt,
                t_steps,
                t_scale,
                t_w,
                t_h,
                i_prompt,
                i_steps,
                i_scale,
                p_prompt,
                p_steps,
                p_scale,
            ],
        )

        def trait_key_change(preset_name: str, key: str):
            ps = next((p for p in load_presets() if p["name"] == preset_name), None)
            if not ps:
                return gr.update(choices=[], value=None)
            options = preset_trait_options(ps).get(key, [])
            return gr.update(choices=options, value=(options[0] if options else None))

        trait1_key.change(trait_key_change, inputs=[dd, trait1_key], outputs=trait1_val)
        trait2_key.change(trait_key_change, inputs=[dd, trait2_key], outputs=trait2_val)
        trait3_key.change(trait_key_change, inputs=[dd, trait3_key], outputs=trait3_val)

        def on_trait_update(base: str, style_tag: str, k1p: str, v1p: str, k2p: str, v2p: str, k3p: str, v3p: str):
            traits = {}
            if k1p and v1p:
                traits[k1p] = v1p
            if k2p and v2p:
                traits[k2p] = v2p
            if k3p and v3p:
                traits[k3p] = v3p
            syn = synthesize_prompt(base, style_tag, traits)
            return syn, syn, syn, syn

        for ctrl in [
            base_prompt,
            style,
            trait1_key,
            trait1_val,
            trait2_key,
            trait2_val,
            trait3_key,
            trait3_val,
        ]:
            ctrl.change(
                on_trait_update,
                inputs=[
                    base_prompt,
                    style,
                    trait1_key,
                    trait1_val,
                    trait2_key,
                    trait2_val,
                    trait3_key,
                    trait3_val,
                ],
                outputs=[live_prompt, t_prompt, i_prompt, p_prompt],
            )

        def sync_steps(val: float):
            return int(val), int(val), int(val)

        steps.change(sync_steps, inputs=steps, outputs=[t_steps, i_steps, p_steps])

        def sync_cfg(val: float):
            return float(val), float(val), float(val)

        cfg.change(sync_cfg, inputs=cfg, outputs=[t_scale, i_scale, p_scale])

        def sync_width(val: float):
            return int(val)

        def sync_height(val: float):
            return int(val)

        w.change(sync_width, inputs=w, outputs=t_w)
        h.change(sync_height, inputs=h, outputs=t_h)

        def apply_to_generators(synth: str, st: float, cg_val: float, ww: float, hh: float):
            return (
                synth,
                int(st),
                float(cg_val),
                int(ww),
                int(hh),
                synth,
                int(st),
                float(cg_val),
                synth,
                int(st),
                float(cg_val),
            )

        apply_btn.click(
            apply_to_generators,
            inputs=[live_prompt, steps, cfg, w, h],
            outputs=[
                t_prompt,
                t_steps,
                t_scale,
                t_w,
                t_h,
                i_prompt,
                i_steps,
                i_scale,
                p_prompt,
                p_steps,
                p_scale,
            ],
        )

        def do_quick_preview(synth: str, neg: str, st: float, cg_val: float, ww: float, hh: float):
            img, _ = txt2img(
                synth,
                negative=neg or "",
                steps=int(max(8, min(24, st))),
                guidance_scale=float(cg_val),
                width=int(max(256, min(512, ww))),
                height=int(max(256, min(512, hh))),
                seed=42,
            )
            return img

        quick.click(
            do_quick_preview,
            inputs=[live_prompt, neg_prompt, steps, cfg, w, h],
            outputs=preview,
        )

        def _txt2img(p, s, sc, wv, hv, se):
            img, meta = txt2img(
                p,
                steps=int(s),
                guidance_scale=float(sc),
                width=int(wv),
                height=int(hv),
                seed=(None if se in ("", None) else int(se)),
            )
            _save_output(img, meta)
            return img, _gallery()

        t_run.click(
            _txt2img,
            inputs=[t_prompt, t_steps, t_scale, t_w, t_h, t_seed],
            outputs=[t_out, t_gallery],
        )

        def _img2img(p, ii, st, sp, sc, se):
            img, meta = img2img(
                p,
                ii,
                strength=float(st),
                steps=int(sp),
                guidance_scale=float(sc),
                seed=(None if se in ("", None) else int(se)),
            )
            _save_output(img, meta)
            return img

        i_run.click(
            _img2img,
            inputs=[i_prompt, i_init, i_strength, i_steps, i_scale, i_seed],
            outputs=i_out,
        )

        def _inpaint(pr, ii, mi, st, sc, se):
            img, meta = inpaint(
                pr,
                ii,
                mi,
                steps=int(st),
                guidance_scale=float(sc),
                seed=(None if se in ("", None) else int(se)),
            )
            _save_output(img, meta)
            return img

        p_run.click(
            _inpaint,
            inputs=[p_prompt, p_init, p_mask, p_steps, p_scale, p_seed],
            outputs=p_out,
        )

        def _gif(p, f, ms, se):
            img, meta = txt2gif(p, int(f), int(ms), (None if se in ("", None) else int(se)))
            return img, meta.get("gif_b64", "")

        v_run.click(_gif, inputs=[v_prompt, v_frames, v_ms, v_seed], outputs=[v_preview, v_b64])

        def _dl(x):
            if not x:
                return "Enter a filename."
            if have_lora(x):
                return f"Already present: {x}"
            try:
                download_lora(x)
                return f"Downloaded: {x}"
            except Exception as e:  # pragma: no cover - network dependent
                return f"Error: {e}"

        d_btn.click(_dl, inputs=d_file, outputs=d_status)

        def save_presets(json_text: str):
            try:
                payload = json.loads(json_text) if json_text.strip() else []
                if isinstance(payload, dict):
                    payload = [payload]
                if not isinstance(payload, list):
                    raise ValueError("Presets JSON must be a list of objects.")
                for entry in payload:
                    if not isinstance(entry, dict) or not entry.get("name"):
                        raise ValueError("Each preset must be an object with a name.")
                save_user_presets(payload)
                merged = load_presets()
                names = [p["name"] for p in merged] or ["<no presets found>"]
                normalized = json.dumps(payload, indent=2)
                return (
                    gr.update(value=normalized),
                    gr.update(choices=names, value=names[0] if names else None),
                    f"Saved {len(payload)} preset(s).",
                )
            except Exception as exc:
                return gr.update(), gr.update(), f"Error: {exc}"

        save_btn.click(save_presets, inputs=preset_editor, outputs=[preset_editor, dd, save_status])

    return demo


if __name__ == "__main__":
    studio().launch()
