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
```
