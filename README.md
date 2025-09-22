# PixStu

## Applications

- `app/pixel_char_studio.py`: Main Pixel Character Studio interface for generating pixel-art renders from text prompts.
- `app/preset_tuner.py`: Lightweight utility for adjusting curated preset defaults and previewing the effect of new inference settings.

## Curated Presets

Curated presets live in `configs/curated_models.json`. The preset tuner lets you tweak the recommended inference steps, guidance scale, and primary LoRA weight before saving the changes back to disk.

To launch the tuner use:

```bash
python -m app.preset_tuner
```

The tool provides:

- A dropdown to select the preset to edit.
- Sliders for denoising steps, guidance scale, and LoRA weight.
- Buttons to update the preset on disk or preview the difference between the existing and proposed values using a fixed seed.

Preview renders use small 64×64 generations to keep iteration fast. When satisfied with the adjustments, press **Update Preset** to persist them.
