# PixStu Quickstart & Setup (Immutable Reference)

## Intent
- Produce high-quality, single-character renders with consistent styling.
- Enforce clean backgrounds and strip text/speech bubbles by default.
- Maintain the retro-modern 16-bit UI aesthetic – visible tooltips, consistent grouping, no tab drift.
- Support CUDA, MPS, CPU, and experimental ZLUDA/zkluda paths.

## Repository Layout (post-migration)
```
chargen/
  generator.py         # BulletProofGenerator facade
  presets.py           # Curated preset loader + asset checks
  studio.py            # Gradio UI (Character, Substitution, Pin Editor, Downloads, Gallery)
  substitution.py      # identity?pose scaffold (ControlNet/OpenPose optional)
  pin_editor.py        # pin-based masking utilities
  metadata.py          # metadata helpers
configs/curated_models.json  # Curated presets
loras/                # LoRA checkpoints (gitignored)
outputs/              # Generated assets + metadata (gitignored)
tools/
  download_manager.py
  verify_repo.py
  test_presets.py
  check_migration.py
  sanitize_reports.py
  aggregate_ratings.py
```

## Environment
```
pip install -r requirements-linux.txt  # or requirements.txt on Windows/macOS
python -m chargen.studio
```
Environment variables:
- `PCS_PORT`, `GRADIO_SERVER_PORT` – override port (default 7860)
- `PCS_SERVER_NAME` – bind host (default 127.0.0.1)
- `PCS_OPEN_BROWSER` – set to 1/true to auto-launch browser

## Immutable UI Rules
- Tabs must include Character Studio, Substitution, Pin Editor, Reference Gallery, Downloads.
- Controls surfaced with tooltips; no hidden/conditional controls for core workflows.
- Retro CSS theme remains consistent (accent color, font, button radius).

## Preset Lifecycle
1. Define presets in `configs/curated_models.json` (model, positives, negatives, LoRAs).
2. Missing LoRA assets surface in the UI; queue downloads via Downloads tab.
3. Run smoke test after edits:
   ```bash
   python tools/test_presets.py
   ```
4. Use `tools/aggregate_ratings.py` to summarise ratings from metadata JSON.

## Verification
Before pushing:
```
python tools/verify_repo.py
python tools/check_migration.py
```
GitHub Actions (`preset_qa.yml`) repeats these checks and stores preset samples.

## Extensibility Notes
- Substitute engine expects optional ControlNet/OpenPose; fall back gracefully when unavailable.
- Pin editor currently calls a placeholder inpaint – slot in SDXL inpaint pipeline when ready.
- Extend `BulletProofGenerator` for upscale/two-pass workflows without breaking the simple API (`generate(prompt, seed) -> PIL.Image`).

Keep this guide immutable unless project intent changes.
