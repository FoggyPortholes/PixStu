#!/usr/bin/env bash
set -euo pipefail

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m chargen.setup_all --install-torch --device cuda
echo "Environment ready. Activate with 'source .venv/bin/activate'. Launch using 'python run_pcs.py'."
