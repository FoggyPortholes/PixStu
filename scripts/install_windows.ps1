$ErrorActionPreference = "Stop"

python -m venv .venv
. .\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m chargen.setup_all --install-torch
Write-Host "Environment ready. Activate with .\\.venv\\Scripts\\Activate.ps1 and launch using python run_pcs.py." -ForegroundColor Green
