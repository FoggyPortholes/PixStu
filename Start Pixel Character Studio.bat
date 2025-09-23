@echo off
setlocal
set HF_HOME=%~dp0hf_cache
set HF_HUB_DISABLE_SYMLINKS_WARNING=1
set PCS_MODELS_ROOT=%~dp0models
set PCS_PORT=7860
python -m venv .venv
call .\.venv\Scripts\activate
pip install -r requirements.txt
pip install torch torchvision
python -m chargen.studio
endlocal
