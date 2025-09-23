# PixStu

PixStu focuses on generating consistent, reusable character art using Stable Diffusion XL pipelines with curated presets and optional reference conditioning.

## Applications

- `app/pixel_char_studio.py`: Character Studio for prompt-driven character generation, LoRA presets, reference guidance, and metadata logging.
- `app/sprite_sheet_studio.py`: Sprite Sheet Studio for packaging an existing sprite into mapped, multi-frame sheets.

## Getting Started

```bash
pip install -r requirements.txt
python -m chargen.studio
```

To install the optimal PyTorch build for your accelerator (CUDA, ROCm, Apple M‑series MPS), run:

```bash
python -m chargen.setup_all --install-torch
```

By default the helper detects your hardware. Override with `--device cuda|rocm|mps|cpu` if needed. Apple Silicon users should export `PYTORCH_ENABLE_MPS_FALLBACK=1` to gracefully fall back on CPU for missing kernels.

Each generation writes a metadata JSON beside the output. Use the in-app rating control to assign a 1–5 score; ratings are saved into the metadata. Aggregate quality across presets with:

```bash
python tools/aggregate_ratings.py
```

Need ControlNet guidance or IP-Adapter styling? Open the accordions in the Character Studio tab, enable the module, select a model, and upload the required conditioning/reference image before hitting **Generate**. CharGen will fetch the weights automatically (cached under `models/controlnet/` and `models/ip_adapter/`).

For touch-ups after generation, expand **AI Edit (Experimental)**, drop in the output image (and optional mask), provide an edit prompt, and let the inpainting pipeline refine just the region you select.

Need to troubleshoot a render? Toggle **Show Diagnostics** in the Character Studio tab to reveal the live log tail and the absolute path to `logs/chargen.log`.

Prefer a guided install? Use the platform scripts:

```bash
./scripts/install_macos.sh    # macOS
./scripts/install_linux.sh    # Linux
```

```powershell
scripts\install_windows.ps1   # Windows PowerShell
```

Each script provisions `.venv`, installs dependencies, and selects the recommended PyTorch build for the platform.

The Character Studio exposes prompt input, preset selection, seed controls, reference uploads, and a built-in gallery sourced from `reference_gallery/`. Outputs (PNG, sprite sheets, metadata) are saved under `outputs/` at runtime.

Sprite sheet packaging can be launched separately:

```bash
python run_sprite_sheet_studio.py
```

## Configuration

Curated presets live in `configs/curated_models.json`. Each entry can define a base SDXL model, optional LoRAs, and recommended inference parameters. Drop additional LoRAs or presets into `models/` and `configs/` respectively to extend the library.

Reference assets can be added to `reference_gallery/` so they appear in the Character Studio gallery for quick conditioning.

## Testing

```bash
pytest
python tools/check_migration.py
python tools/aggregate_ratings.py
```
