#!/usr/bin/env bash
set -euo pipefail

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m chargen.setup_all --install-torch --device mps
export PYTORCH_ENABLE_MPS_FALLBACK=1
echo "Environment ready. Run 'source .venv/bin/activate' and 'python run_pcs.py' to launch CharGen Studio."
