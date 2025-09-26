# PixStu

PixStu focuses on generating consistent, reusable character art across multiple styles (comic, anime, pixel, realism). The studio ships with curated presets, optional substitution/pin editing tools, and a retro-modern 16-bit UI.

## Quick Start

```bash
pip install -r requirements-linux.txt  # or requirements.txt on Windows/macOS
python -m chargen.studio
```

Environment variables:
- `PCS_PORT` / `GRADIO_SERVER_PORT` – choose listening port (default 7860)
- `PCS_SERVER_NAME` – bind host (default `127.0.0.1`)
- `PCS_OPEN_BROWSER` – set to `1` to auto-open the UI in a browser

## Presets

Curated preset definitions live in `configs/curated_models.json`. Each entry specifies the base model, LoRAs, positive/negative terms, step count, CFG scale, and default resolution. Missing LoRA assets surface in the UI and can be queued via the Downloads tab.

Run the preset smoke test (generates sample outputs under `docs/preset_samples/`):

```bash
python tools/test_presets.py
```

## Tabs Overview

- **Character Studio** – core text-to-image generation wrapped in `BulletProofGenerator`
- **Substitution** – identity & pose blending scaffold (OpenPose optional)
- **Pin Editor** – targeted masks per “pin” for future inpainting integrations
- **Reference Gallery** – placeholder for curated reference thumbnails
- **Downloads** – single-queue downloader for LoRAs/models

## Tooling

- `tools/download_manager.py` – serial downloader
- `tools/verify_repo.py` – structure guard (run before commits)
- `tools/sanitize_reports.py` – strips PII from QA outputs
- `tools/check_migration.py` – ensures legacy `app/` layout is retired
- `tools/aggregate_ratings.py` – aggregates ratings stored in metadata JSON files

A GitHub workflow (`.github/workflows/preset_qa.yml`) executes preset smoke tests plus sanitation on pull requests touching presets or tooling.

## Repository Verification

Before opening a PR:

```bash
python tools/verify_repo.py
python tools/check_migration.py
```

Optional QA:

```bash
python tools/aggregate_ratings.py  # summarise ratings from outputs/
```

## Directories


## 🧩 Presets & Generator (WAN2.2)
- Choose a preset, tweak **traits** (role, palette, pose), and see a **live synthesized prompt**.
- Use **⚡ Quick Preview** for a fast, low-step render.
- Save your own presets to `.pixstu/presets.json`; defaults live in `assets/presets/wan22.json`.
- `loras/` – place downloaded LoRA checkpoints here (ignored by git)
- `outputs/` – generated images + metadata (ignored by git)
- `docs/preset_samples/` – preset QA output (captured in CI artifacts)

## Contributing

- Respect the immutable design intent: single-character composition, uncluttered background, retro-modern UI styling.
- Keep presets bullet-proof: enforce negative prompts, surface missing assets, prefer reproducible seeds.
- Run the verification scripts and CI workflows locally when possible.

Enjoy building with PixStu!
