# PixStu

PixStu focuses on generating consistent, reusable character art using Stable Diffusion XL pipelines with curated presets and optional reference conditioning.

## Applications

- `app/pixel_char_studio.py`: Character Studio for prompt-driven character generation, LoRA presets, reference guidance, and metadata logging. The built-in LoRA library lets you preview adapters and pull new ones from Hugging Face with a click.
- `app/sprite_sheet_studio.py`: Sprite Sheet Studio for packaging an existing sprite into mapped, multi-frame sheets.

## Getting Started

```bash
pip install -r requirements.txt
python -m chargen.studio
```

<<<<<<< Updated upstream
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
=======
The Character Studio exposes prompt input, preset selection, seed controls, reference uploads, a LoRA preview/download panel, and a gallery sourced from `reference_gallery/`. Outputs (PNG, sprite sheets, metadata) are saved under `outputs/` at runtime.

### Starter Presets

PixStu ships with ready-to-use style presets so you can explore quickly:

- `SDXL Character Base` – neutral baseline for prompt-driven experimentation.
- `SDXL Detail Offset` – adds Stability AI's official sharpening LoRA.
- `SDXL Pixel Character` – crisp sprite aesthetics.
- `SDXL Anime Companion` – stylised manga/anime characters.
- `Heroic Comics (Marvel)` – bold Marvel-inspired comic style.
- `Legendary Comics (DC)` – high-contrast heroic comic look.
- `JRPG Hero (Final Fantasy)` – cinematic JRPG fantasy styling.
- `Animated Feature Style` – modern animated-movie charm.
>>>>>>> Stashed changes

Sprite sheet packaging can be launched separately:

```bash
python run_sprite_sheet_studio.py
```

## Configuration

Curated presets live in `configs/curated_models.json`. Each entry can define a base SDXL model, optional LoRAs, and recommended inference parameters. Drop additional LoRAs or presets into `models/` and `configs/` respectively to extend the library.

LoRA metadata and download links are stored in `configs/lora_catalog.json`. Entries reference Hugging Face repositories, preview images, and local storage paths under `models/lora/`.

Reference assets can be added to `reference_gallery/` so they appear in the Character Studio gallery for quick conditioning.

## Testing

```bash
<<<<<<< Updated upstream
pytest
python tools/check_migration.py
python tools/aggregate_ratings.py
=======
python -m pytest
>>>>>>> Stashed changes
```
