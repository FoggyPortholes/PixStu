#!/usr/bin/env bash
set -euo pipefail

# Deployment script for PixStu
# Runs verification, installs requirements, downloads missing assets, and launches studio.

# 1. Environment setup
echo "[Deploy] Installing requirements..."
pip install --upgrade pip
pip install -r requirements-linux.txt

# 2. Verify repo structure
echo "[Deploy] Running repo verification..."
python tools/verify_repo.py

# 3. Preset conformance check
echo "[Deploy] Checking preset conformance..."
python tools/preset_conformance.py

# 4. Smoke test presets
echo "[Deploy] Running preset smoke tests..."
python tools/test_presets.py || echo "[WARN] Smoke test images may fail if GPU/VRAM is limited"

# 5. Run single-character guard (if sample outputs exist)
if [ -d docs/preset_samples ]; then
  echo "[Deploy] Running single-character guard on samples..."
  python tools/single_character_guard.py docs/preset_samples/*/*.png || true
fi

# 6. Launch application
echo "[Deploy] Starting PixStu Studio..."
exec python -m chargen.studio
