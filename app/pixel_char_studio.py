import os, json, glob, uuid, datetime, contextlib

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

from . import pipeline_cache

ROOT = os.path.abspath(os.path.dirname(__file__))
PROJ = os.path.abspath(os.path.join(ROOT, ".."))
MODELS_ROOT = pipeline_cache.MODELS_ROOT
_abs_under_models = pipeline_cache.resolve_under_models

_DEV_KIND = pipeline_cache.DEV_KIND
_DTYPE = pipeline_cache.DTYPE
_DEVICE = pipeline_cache.DEVICE

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

DEFAULT_MODEL_ID = pipeline_cache.DEFAULT_MODEL_ID


def load_base(model_id, quality, local_dir=None):
    return pipeline_cache.get_txt2img(model_id, quality=quality, local_dir=local_dir)

def configure_adapters(pipe, lcm_dir, loras, lora_weights, quality_mode):
    return pipeline_cache.apply_loras(
        pipe,
        quality_mode=quality_mode,
        lcm_dir=lcm_dir,
        loras=loras,
        lora_weights=lora_weights,
    )

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
    active_adapters = pipeline_cache.get_active_adapters(pipe)
    meta = {"prompt":prompt,"negative_prompt":negative,"seed":seed,"steps":steps,"guidance":cfg,
            "width":int(w),"height":int(h),"palette":palette,"pixel_scale":int(pixel_scale),
            "model":base_model,"adapters":active_adapters[:], "created_at":datetime.datetime.utcnow().isoformat()+"Z",
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
    with gr.Tabs():
        with gr.Tab("Results"):
            out = gr.Image(label="Output")
            pix = gr.Image(label="Pixelated")
            meta = gr.Code(label="Meta", language="json")
        with gr.Tab("Timeline Preview"):
            gr.Markdown("#### Scrub through frames with onion-skinning and FPS control")
            refresh_btn = gr.Button("Refresh Preview", variant="secondary")
            with gr.Row():
                timeline_html = gr.HTML("""
                <div id='px-timeline' style='display:flex;gap:12px;align-items:center'>
                  <canvas id='px-canvas' width='128' height='128' style='image-rendering: pixelated;border:1px solid #ccc'></canvas>
                  <div style='display:flex;flex-direction:column;gap:8px'>
                    <input id='px-scrub' type='range' min='0' max='0' value='0'>
                    <label>FPS <input id='px-fps' type='number' min='1' max='60' value='8'></label>
                    <label>Onion α <input id='px-onion' type='range' min='0' max='0.6' step='0.05' value='0.25'></label>
                    <button id='px-play'>Play/Pause</button>
                  </div>
                </div>
                <script>
                (async function(){
                  const ctx = document.getElementById('px-canvas').getContext('2d');
                  const scrub = document.getElementById('px-scrub');
                  const fpsEl = document.getElementById('px-fps');
                  const onionEl = document.getElementById('px-onion');
                  const playBtn = document.getElementById('px-play');
                  // This assumes PixStu writes frames.json alongside frames/ outputs
                  const manifest = await fetch('./frames/frames.json?_=' + Date.now()).then(r=>r.json());
                  const imgs = await Promise.all(manifest.frames.map(async (p)=> {
                    const img = new Image(); img.src = './frames/' + p + '?_=' + Date.now();
                    await img.decode(); return img;
                  }));
                  scrub.max = String(imgs.length - 1);
                  let i=0, timer=null;
                  function draw(index){
                    ctx.clearRect(0,0,ctx.canvas.width,ctx.canvas.height);
                    const prev = imgs[(index-1+imgs.length)%imgs.length];
                    const curr = imgs[index];
                    const next = imgs[(index+1)%imgs.length];
                    const α = parseFloat(onionEl.value);
                    ctx.globalAlpha = α; ctx.drawImage(prev,0,0,128,128);
                    ctx.globalAlpha = 1.0; ctx.drawImage(curr,0,0,128,128);
                    ctx.globalAlpha = α; ctx.drawImage(next,0,0,128,128);
                    ctx.globalAlpha = 1.0;
                  }
                  function play(){
                    const fps = Math.max(1, Math.min(60, parseInt(fpsEl.value||'8',10)));
                    if(timer) { clearInterval(timer); timer=null; return; }
                    timer = setInterval(()=>{ i=(i+1)%imgs.length; scrub.value=String(i); draw(i); }, 1000/fps);
                  }
                  scrub.oninput = e => { i = parseInt(e.target.value,10); draw(i); };
                  onionEl.oninput = ()=> draw(i);
                  fpsEl.oninput = ()=> { if(timer){ clearInterval(timer); timer=null; play(); } };
                  playBtn.onclick = play;
                  draw(0);
                })();
                </script>
                """)
            refresh_btn.click(fn=None, inputs=None, outputs=None, _js="() => { location.reload(); }")
    status = gr.Textbox(label="Status")

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
