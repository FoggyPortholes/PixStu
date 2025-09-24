#!/usr/bin/env bash
set -euo pipefail

# Deployment + Production automation script for PixStu
# Installs requirements, verifies repo, checks conformance, downloads assets, and launches PixStu.

# ===== Utility =====
log() { echo "$(date '+%Y-%m-%d %H:%M:%S') $*"; }
ART_DIR="deploy_artifacts/$(date +%Y%m%d_%H%M%S)"
PORT="${PCS_PORT:-7860}"
mkdir -p "$ART_DIR"

cleanup() {
  log "[Deploy] Caught shutdown signal. Stopping PixStu gracefully..."
  # Gradio will exit on SIGTERM to the python process launched with exec
}
trap cleanup INT TERM

# ========== CHECKLIST ==========
# 1. GPU Drivers:
#    - NVIDIA: CUDA 11.8+ or CUDA 12.x with matching PyTorch build.
#    - AMD: ZLUDA installed.
#    - Intel: zkluda installed.
#    - Apple Silicon: macOS 13+, MPS enabled.
#
# 2. Python Environment:
#    - Python 3.10+ (recommended 3.11)
#    - Virtualenv or conda environment activated.
#
# 3. Hugging Face Authentication (for private LoRAs):
#    - Run `huggingface-cli login` if required.
#    - Ensure HF_TOKEN is set if running headless.
#
# 4. Network & Ports:
#    - Ensure port $PORT is open (override with PCS_PORT).
#    - Use reverse proxy (nginx/traefik) for HTTPS in production.
#
# 5. VRAM Sizing:
#    - 12 GB VRAM recommended for SDXL.
#    - Set PIPELINE_CACHE_MAX to balance performance vs memory.
#
# 6. Monitoring:
#    - Monitor logs for [UI] drift, [WARN] multiple characters, [FAIL] migration.
# =================================

# 0. Device/Backend probe
log "[Deploy] Probing compute backend..."
python - <<'PY'
import torch, json, os
backend = {
  'cuda_available': torch.cuda.is_available(),
  'cuda_device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
  'mps_available': getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available(),
  'device_preference': 'cuda' if torch.cuda.is_available() else ('mps' if (getattr(torch.backends,'mps',None) and torch.backends.mps.is_available()) else 'cpu')
}
print(json.dumps(backend))
PY

# 0b. HF token hint (do not expose token value)
if grep -q "huggingface.co" configs/curated_models.json 2>/dev/null; then
  if ! command -v huggingface-cli >/dev/null 2>&1; then
    log "[WARN] huggingface-cli not installed. Private models may fail to download."
  fi
  if [ -z "${HF_TOKEN:-}" ]; then
    log "[INFO] HF_TOKEN not set. Public models will work; private models require auth."
  else
    log "[OK] HF_TOKEN present (value hidden)."
  fi
fi

# 1. Environment setup
log "[Deploy] Installing requirements..."
pip install --upgrade pip
# Prefer wheels to avoid slow source builds
pip install --prefer-binary -r requirements-linux.txt

# 2. Verify repo structure
log "[Deploy] Running repo verification..."
python tools/verify_repo.py | tee "$ART_DIR/verify_repo.log"

# 3. Preset conformance check
log "[Deploy] Checking preset conformance..."
python tools/preset_conformance.py | tee "$ART_DIR/preset_conformance.log"

# 4. Smoke test presets (parallel optional)
log "[Deploy] Running preset smoke tests..."
if [ "${PIXSTU_PARALLEL_SMOKE:-1}" = "1" ]; then
  python - <<'PY' || true
import os, json, time, concurrent.futures as cf
from diffusers import StableDiffusionXLPipeline
import torch
CFG=("configs/curated_models.json","docs/preset_samples")
ps=json.load(open(CFG[0],encoding='utf-8'))
os.makedirs(CFG[1],exist_ok=True)

def run(preset):
    model=preset.get('model','stabilityai/stable-diffusion-xl-base-1.0')
    pipe=StableDiffusionXLPipeline.from_pretrained(model, torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32).to('cuda' if torch.cuda.is_available() else ('mps' if getattr(torch.backends,'mps',None) and torch.backends.mps.is_available() else 'cpu'))
    img=pipe(prompt='heroic character portrait, ' + ', '.join(preset.get('positive',[])), negative_prompt=', '.join(preset.get('negative',[])), num_inference_steps=preset.get('steps',20), guidance_scale=preset.get('cfg',7.0)).images[0]
    outdir=os.path.join(CFG[1], preset['name'].replace(' ','_'))
    os.makedirs(outdir, exist_ok=True)
    p=os.path.join(outdir, f"sample_{int(time.time()*1000)}.png")
    img.save(p)
    return p

with cf.ThreadPoolExecutor(max_workers=min(4, len(ps))) as ex:
    for pth in ex.map(run, ps):
        print('[OK] sample', pth)
PY
else
  python tools/test_presets.py || true
fi

# 5. Run single-character guard (if sample outputs exist)
if [ -d docs/preset_samples ]; then
  log "[Deploy] Running single-character guard on samples..."
  python tools/single_character_guard.py docs/preset_samples/*/*.png | tee "$ART_DIR/single_character_guard.log" || true
fi

# 6. Save curated presets & logs as artifacts
cp -f configs/curated_models.json "$ART_DIR/curated_models.json" || true
cp -rf docs/preset_samples "$ART_DIR/" 2>/dev/null || true
log "[Deploy] Artifacts stored in $ART_DIR"

# 7. Launch application
log "[Deploy] Starting PixStu Studio on port $PORT..."
export PCS_PORT="$PORT"
exec python -m chargen.studio

