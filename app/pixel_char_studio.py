import os, sys, json, glob, uuid, datetime, contextlib
from typing import List, Optional
import gradio as gr

# ``gradio_client`` currently assumes that every JSON schema object is a
# dictionary.  When the schema contains ``additionalProperties: false`` the
# library receives a boolean value instead which causes a ``TypeError`` when it
# tries to look for keys on the boolean.  We gently patch the helper so Gradio
# can render the interface without crashing.  The shim can be dropped once the
# upstream issue is fixed.
try:  # pragma: no cover - defensive patching for a runtime dependency
    from gradio_client import utils as _grc_utils
except Exception:  # pragma: no cover - gradio client may not be present
    _grc_utils = None
else:
    _orig_json_schema_to_python_type = _grc_utils._json_schema_to_python_type

    def _pcs_json_schema_to_python_type(schema, defs):
        if isinstance(schema, bool):
            return "Any" if schema else "None"
        return _orig_json_schema_to_python_type(schema, defs)

    _grc_utils._json_schema_to_python_type = _pcs_json_schema_to_python_type
import torch
from PIL import Image, ImageColor, ImageFilter
from diffusers import DiffusionPipeline, LCMScheduler, DPMSolverMultistepScheduler

ROOT = os.path.abspath(os.path.dirname(__file__))
PROJ = os.path.abspath(os.path.join(ROOT, ".."))
EXE_DIR = (os.path.dirname(sys.executable) if getattr(sys, "frozen", False) else PROJ)
MODELS_ROOT = os.getenv("PCS_MODELS_ROOT", os.path.join(EXE_DIR, "models"))

def _abs_under_models(p):
    if not p: return p
    expanded = os.path.expanduser(str(p))
    if os.path.isabs(expanded): return os.path.normpath(expanded)
    normalized = os.path.normpath(expanded)
    project_candidate = os.path.abspath(os.path.join(PROJ, normalized))
    if os.path.exists(project_candidate):
        return os.path.normpath(project_candidate)
    parts = normalized.split(os.sep)
    if parts and parts[0] == "models":
        normalized = os.path.join(*parts[1:]) if len(parts) > 1 else ""
    return os.path.normpath(os.path.abspath(os.path.join(MODELS_ROOT, normalized)))

def _device():
    if torch.cuda.is_available(): return "cuda", torch.float16, "cuda"
    if getattr(torch.backends,"mps",None) and torch.backends.mps.is_available(): return "mps", torch.float16, "mps"
    return "cpu", torch.float32, "cpu"

_DEV_KIND, _DTYPE, _DEVICE = _device()
_PIPE = None; _CUR_BASE = None; _ACTIVE_ADAPTERS = []

PALETTES = {
  "None": [],
  "DB16 (DawnBringer 16)": ["#140c1c","#442434","#30346d","#4e4a4e","#854c30","#346524","#d04648","#757161",
                            "#597dce","#d27d2c","#8595a1","#6daa2c","#d2aa99","#6dc2ca","#dad45e","#deeed6"]
}

def _make_palette_img(cols):
    pal_img = Image.new("P",(16,16)); pal=[]
    for hx in cols: pal.extend(ImageColor.getrgb(hx))
    pal += [0]*(768-len(pal)); pal_img.putpalette(pal); return pal_img

def to_pixel_art(img, scale=8, palette="DB16 (DawnBringer 16)", dither=False, crisp=True, sharpen=0.25):
    if sharpen>0: img = img.filter(ImageFilter.UnsharpMask(radius=1.0, percent=int(sharpen*150)))
    w,h = img.size; dw=max(1,w//max(1,scale)); dh=max(1,h//max(1,scale))
    small = img.resize((dw,dh), Image.BOX if crisp else Image.BILINEAR)
    if palette!="None":
        pal = _make_palette_img(PALETTES[palette]); small = small.convert("RGB").quantize(palette=pal, dither=Image.FLOYDSTEINBERG if dither else Image.NONE).convert("RGB")
    return small.resize((w,h), Image.NEAREST)

def scan_loras():
    ldir = os.path.join(MODELS_ROOT,"lora")
    files = sorted(glob.glob(os.path.join(ldir,"*.safetensors")) + glob.glob(os.path.join(ldir,"*.ckpt")))
    return [os.path.abspath(p) for p in files]

def load_curated():
    try:
        with open(os.path.join(PROJ,"configs","curated_models.json"),"r",encoding="utf-8") as f:
            return (json.load(f).get("presets") or [])
    except: return []

DEFAULT_MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"

def load_base(model_id, quality, local_dir=None):
    global _PIPE, _CUR_BASE

    # Resolve the requested model id.
    mid = model_id or DEFAULT_MODEL_ID
    # Allow callers to explicitly provide a local directory that should take
    # precedence over the identifier.
    if local_dir:
        candidate = _abs_under_models(local_dir)
        if os.path.isdir(candidate):
            mid = candidate
    # If the identifier maps to a folder inside the models directory make sure
    # we pass the absolute path to the pipeline loader.  This allows users to
    # select entries from the UI returned by ``scan_base_models``.
    resolved = _abs_under_models(mid)
    if os.path.isdir(resolved):
        mid = resolved

    need_reload = (_PIPE is None) or (_CUR_BASE != mid)
    if need_reload:
        pipe = DiffusionPipeline.from_pretrained(
            mid,
            use_safetensors=True,
            torch_dtype=(_DTYPE if _DEV_KIND != "dml" else torch.float32),
            variant=("fp16" if _DTYPE == torch.float16 else None),
        )
        pipe.to(_DEVICE)
        pipe.enable_vae_tiling()
        if _DEV_KIND == "cuda":
            with contextlib.suppress(Exception):
                pipe.enable_xformers_memory_efficient_attention()
        _PIPE = pipe
        _CUR_BASE = mid

    if quality == "Fast (LCM)":
        _PIPE.scheduler = LCMScheduler.from_config(_PIPE.scheduler.config)
    else:
        _PIPE.scheduler = DPMSolverMultistepScheduler.from_config(
            _PIPE.scheduler.config
        )

    return _PIPE

def configure_adapters(pipe, lcm_dir, loras, lora_weights, quality_mode):
    global _ACTIVE_ADAPTERS
    adapters, weights = [], []
    if quality_mode=="Fast (LCM)" and lcm_dir and os.path.isdir(lcm_dir):
        try: pipe.load_lora_weights(lcm_dir, adapter_name="lcm"); adapters.append("lcm"); weights.append(1.0)
        except Exception as e: print("[WARN] LCM:", e)
    for p in (loras or []):
        try:
            an = os.path.splitext(os.path.basename(p))[0]
            try: pipe.load_lora_into_unet(p, adapter_name=an)
            except: pipe.load_lora_weights(p, adapter_name=an)
            weights.append(float(lora_weights.get(p,1.0))); adapters.append(an)
        except Exception as e: print("[WARN] LoRA:", p, e)
    if adapters:
        try: pipe.set_adapters(adapters, adapter_weights=weights)
        except Exception as e: print("[WARN] set_adapters:", e)
    elif _ACTIVE_ADAPTERS:
        try: pipe.set_adapters([])
        except Exception as e: print("[WARN] clear adapters:", e)
        try: pipe.unload_lora_weights()
        except Exception as e: print("[WARN] unload adapters:", e)
    _ACTIVE_ADAPTERS = adapters[:]

def _sanitize(s):
    if s is None: return ""
    s=str(s).strip().strip('"').strip("'")
    return ("p"+s if s.lower().startswith("ixel art") else s)

def generate(prompt, negative, seed, steps, cfg, w, h, quality, pixel_scale, palette, dither, crisp, sharpen,
             base_model, lcm_dir, lora_files, lora_weights_json):
    prompt=_sanitize(prompt); negative=_sanitize(negative)
    if not prompt: return None, None, {"error":"empty prompt"}
    pipe = load_base(base_model, quality)
    try: lw=json.loads(lora_weights_json) if lora_weights_json else {}
    except: lw={}
    configure_adapters(pipe, lcm_dir, lora_files or [], lw, quality)

    steps=int(steps); cfg=float(cfg)
    if quality=="Fast (LCM)": steps=max(4,min(steps,12)); cfg=max(1.0,min(cfg,2.5))
    elif quality.startswith("Pro"): steps=max(34,steps); cfg=6.0
    else: steps=max(28,steps); cfg=max(5.0,min(cfg,7.0))

    if int(seed)<0: seed=int.from_bytes(os.urandom(4),"little")
    gen = torch.Generator(device=("cuda" if _DEV_KIND=="cuda" else "cpu")).manual_seed(int(seed))
    ac = (torch.autocast("cuda",dtype=torch.float16) if _DEV_KIND=="cuda" else
          torch.autocast("mps",dtype=torch.float16) if _DEV_KIND=="mps" else contextlib.nullcontext())
    with ac:
        img = pipe(prompt=prompt, negative_prompt=(negative or None),
                   width=int(w), height=int(h),
                   num_inference_steps=int(steps), guidance_scale=float(cfg),
                   generator=gen).images[0]
    pix = to_pixel_art(img, scale=int(pixel_scale), palette=str(palette),
                       dither=bool(dither), crisp=bool(crisp), sharpen=float(sharpen))
    meta = {"prompt":prompt,"negative_prompt":negative,"seed":seed,"steps":steps,"guidance":cfg,
            "width":int(w),"height":int(h),"palette":palette,"pixel_scale":int(pixel_scale),
            "model":base_model,"adapters":_ACTIVE_ADAPTERS[:], "created_at":datetime.datetime.utcnow().isoformat()+"Z",
            "loras": lora_files or []}
    return img, pix, meta

def scan_base_models():
    items=[]
    for d in sorted(glob.glob(os.path.join(MODELS_ROOT,"*"))):
        if os.path.isdir(d) and (os.path.isfile(os.path.join(d,"model_index.json")) or os.path.isfile(os.path.join(d,"config.json"))):
            items.append(d)
    items += ["stabilityai/stable-diffusion-xl-base-1.0"]
    return items

cur_presets = load_curated()
cur_names = [p["name"] for p in cur_presets] if cur_presets else []

def _apply_curated(name):
    if not name: return [gr.update()]*6
    p = next((x for x in cur_presets if x.get("name")==name), None)
    if not p: return [gr.update()]*6
    base = p.get("base_model") or "stabilityai/stable-diffusion-xl-base-1.0"
    lcm  = _abs_under_models(p.get("lcm_dir") or "")
    loras=[]; wts={}
    for e in p.get("loras",[]):
        q=e.get("path"); 
        if not q: continue
        ap = _abs_under_models(q)
        if os.path.isfile(ap): loras.append(ap); wts[ap]=float(e.get("weight",1.0))
    sugg=p.get("suggested",{})
    return [gr.update(value=base), gr.update(value=lcm), gr.update(value=loras),
            gr.update(value=json.dumps(wts)), gr.update(value=sugg.get("steps")), gr.update(value=sugg.get("guidance"))]

with gr.Blocks(analytics_enabled=False) as demo:
    gr.Markdown("Pixel Character Studio (Portable)")
    with gr.Row():
        prompt = gr.Textbox(label="Prompt", lines=3, value="pixel art hero, clean 1px outline, solid fills, two-tone shading, full body, facing camera")
        negative = gr.Textbox(label="Negative", lines=2, value="lowres, blur, gradient background, artifacts, watermark, text")
    with gr.Row():
        steps = gr.Slider(2,80,30,1,label="Steps")
        cfg   = gr.Slider(0,12,7.0,0.1,label="Guidance")
        width = gr.Dropdown([384,512,640,768], value=512, label="W")
        height= gr.Dropdown([384,512,640,768], value=512, label="H")
        quality = gr.Radio(["Fast (LCM)","Quality (Base)","Pro (Base + Refiner)"], value="Quality (Base)", label="Mode")
    with gr.Row():
        pixel_scale = gr.Slider(2,16,8,1,label="Pixel scale")
        palette = gr.Dropdown(list(PALETTES.keys()), value="DB16 (DawnBringer 16)", label="Palette")
        dither = gr.Checkbox(False, label="Dither"); crisp = gr.Checkbox(True, label="Crisp")
        sharpen = gr.Slider(0.0,2.0,0.25,0.05,label="Sharpen")

    gr.Markdown("Models and Adapters")
    with gr.Row():
        base_model = gr.Dropdown(choices=scan_base_models(), value="stabilityai/stable-diffusion-xl-base-1.0", label="Base (local snapshot or HF id)")
        lcm_dir = gr.Textbox(value=os.path.join(MODELS_ROOT,"lcm"), label="LCM dir (optional)")
    lora_files = gr.CheckboxGroup(choices=scan_loras(), value=[], label="LoRAs (multi)")
    lora_weights_json = gr.Textbox(value="{}", lines=2, label="LoRA weights JSON {path: weight}")
    with gr.Row():
        curated = gr.Dropdown(choices=cur_names, value=(cur_names[0] if cur_names else None), label="Curated")
        apply_cur = gr.Button("Apply curated")

    gen = gr.Button("Generate")
    out = gr.Image(label="Output"); pix = gr.Image(label="Pixelated"); meta = gr.Code(label="Meta", language="json"); status = gr.Textbox(label="Status")

    def _on_gen(prompt,negative,steps,cfg,width,height,quality,pixel_scale,palette,dither,crisp,sharpen,base_model,lcm_dir,lora_files,lw_json):
        img, pimg, m = generate(prompt,negative,-1,steps,cfg,width,height,quality,pixel_scale,palette,dither,crisp,sharpen,base_model,lcm_dir,lora_files,lw_json)
        meta_json = json.dumps(m, indent=2, sort_keys=True) if isinstance(m, dict) else str(m)
        return img, pimg, meta_json, (m.get("error") if isinstance(m,dict) and "error" in m else "")

    gen.click(_on_gen,
        inputs=[prompt,negative,steps,cfg,width,height,quality,pixel_scale,palette,dither,crisp,sharpen,base_model,lcm_dir,lora_files,lora_weights_json],
        outputs=[out,pix,meta,status])

    apply_cur.click(_apply_curated, inputs=[curated], outputs=[base_model,lcm_dir,lora_files,lora_weights_json,steps,cfg])

if __name__ == "__main__":
    port = int(os.getenv("PCS_PORT","7860"))
    os.environ["HF_HOME"] = os.path.join(PROJ, "hf_cache")
    os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
    demo.launch(share=False, inbrowser=True, server_name="127.0.0.1", server_port=port, show_error=True)
