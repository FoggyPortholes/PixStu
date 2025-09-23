# CharGen Studio — Immutable Documentation (Codex-Ready)

## App Intent (Immutable)

CharGen Studio’s sole purpose is to use AI to **generate high-quality, consistent, reusable character images of any kind**.

- Lightweight by design.
- Runs on CPU, NVIDIA CUDA, AMD (ZLUDA), Intel (zkluda), Apple MPS.
- Presets, seeds, and metadata ensure consistency.
- Optional advanced tools: ControlNet, IP-Adapter, AI Edit (with automatic masking).
- UI must remain **retro-modern 16‑bit aesthetic**, clean, flashy, and uniform.
- Only **two tabs**: *Character Studio* and *Reference Gallery*.

## Canonical Repo (Immutable)

The official codebase lives at: 👉 [FoggyPortholes/PixStu](https://github.com/FoggyPortholes/PixStu)

This repo is the source of truth. All changes must align with the above **App Intent**. Legacy files currently present include `app/`, `run_pcs.py`, and legacy launch scripts. Migration to `chargen/` structure is required.

## Basic Structure (Post-Migration)

```
chargen/
  ├── studio.py          # Entry point
  ├── character_studio.py # UI (2 tabs)
  ├── generator.py        # Txt2Img, Img2Img, ControlNet, IP-Adapter
  ├── editor.py           # AI Edit (inpainting)
  ├── auto_mask.py        # Automatic masking helper
  ├── presets.py          # Curated presets loader
  ├── reference_gallery.py# Gallery integration
  ├── logging_config.py   # Logging setup
  ├── hw_detect.py        # Device detection (CUDA/ROCm/MPS/CPU)
  ├── ui_theme.py         # Retro-modern 16-bit theme
  ├── ui_guard.py         # Drift detection (UI rules)
  ├── setup_all.py        # Auto-install deps + download ControlNets
  └── model_setup.py      # Model management
configs/
  └── curated_models.json # Preset definitions
models/
  ├── controlnet/         # Auto-downloaded (canny, openpose, depth)
  └── ip_adapter/         # Drop-in weights
```

## UI Guidelines (Immutable)

- **Uniform grouping** of controls (Prompt, Preset, Seed/Jitter, Size, Reference).
- **Only two tabs**: Character Studio and Reference Gallery.
- **Retro-modern 16-bit design**: pixel grid background, neon accents, `Press Start 2P` font.
- **Tooltips everywhere**.
- **No drift**: Any extra tabs or styling changes must be flagged.

## UI Functional Coverage

All app functionality must be surfaced clearly in the UI:

- **Character Studio tab**:
  - Prompt input (auto-injected style descriptors from preset).
  - Preset selector (bullet‑proof presets like DC Comic Book).
  - Seed + Seed Jitter controls.
  - Output size selector.
  - Reference image upload and Ref Strength slider.
  - Generate button (calls BulletProofGenerator).
  - Diagnostics toggle (shows logs/metadata path).
  - **AI Edit (Experimental)** accordion:
    - Image to Edit upload.
    - Mask upload (optional).
    - Edit Prompt input.
    - Edit Strength slider.
    - Auto‑Mask toggle.
    - Target Region selector.
    - Apply Edit button.
    - Edited output preview + status box.
- **Reference Gallery tab**:
  - Pixel grid of thumbnails.
  - Clicking loads into Character Studio as reference image.
- **Status & Metadata**:
  - Always display generation status.
  - Path to saved metadata JSON shown in UI.

## UI Component Mapping

| Feature                | UI Element                  | Backend Function/Module           |
| ---------------------- | --------------------------- | --------------------------------- |
| Character Prompt       | Textarea (Character Studio) | `BulletProofGenerator.generate`   |
| Preset Selection       | Dropdown (Character Studio) | `presets.py` loader + generator   |
| Seed / Jitter          | Number + Range inputs       | `BulletProofGenerator.generate`   |
| Output Size            | Dropdown                    | `BulletProofGenerator.generate`   |
| Reference Image        | File upload                 | `generator.py` (img2img pipeline) |
| Ref Strength           | Slider                      | `generator.py` (img2img strength) |
| Generate Button        | Button                      | `BulletProofGenerator.generate`   |
| Diagnostics            | Toggle/Panel                | `logging_config.py` + metadata    |
| AI Edit: Image Upload  | File upload                 | `editor.py` inpainting            |
| AI Edit: Mask Upload   | File upload                 | `auto_mask.py` / `editor.py`      |
| AI Edit: Edit Prompt   | Text input                  | `editor.py`                       |
| AI Edit: Edit Strength | Slider                      | `editor.py`                       |
| AI Edit: Auto-Mask     | Dropdown (On/Off)           | `auto_mask.py`                    |
| AI Edit: Target Region | Dropdown                    | `auto_mask.py`                    |
| AI Edit: Apply Edit    | Button                      | `editor.py.edit_image`            |
| Reference Gallery      | Grid thumbnails             | `reference_gallery.py`            |
| Metadata Path Display  | Text box                    | `metadata.py`                     |
| Status Display         | Status box                  | `logging_config.py`               |

## User Rating Storage

Each generated image has a metadata JSON. Extend metadata with a `rating` field:

```python
# chargen/metadata.py (excerpt)
import json, os

def save_metadata(meta_path, metadata, rating=None):
    if rating is not None:
        metadata["rating"] = rating
    os.makedirs(os.path.dirname(meta_path), exist_ok=True)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
```

In the UI, after generation, display 1–5 star rating control. On selection, call `save_metadata()` with rating.

This allows users to curate and QA generations for quality tracking.

## Rating Aggregation Tool

```python
# tools/aggregate_ratings.py
"""Aggregate user ratings from metadata JSONs."""
import os, json
from collections import defaultdict

OUTPUTS = "outputs"

def collect_ratings():
    ratings = defaultdict(list)
    for root, _, files in os.walk(OUTPUTS):
        for f in files:
            if f.endswith(".json"):
                path = os.path.join(root, f)
                try:
                    data = json.load(open(path, encoding="utf-8"))
                    preset = data.get("preset", "unknown")
                    if "rating" in data:
                        ratings[preset].append(data["rating"])
                except Exception:
                    continue
    return ratings


def main():
    ratings = collect_ratings()
    for preset, vals in ratings.items():
        if not vals:
            continue
        avg = sum(vals)/len(vals)
        print(f"{preset}: avg {avg:.2f} from {len(vals)} ratings")

if __name__ == "__main__":
    main()
```

Run `python tools/aggregate_ratings.py` to print average ratings per preset. This enables QA teams to measure quality trends across styles.

## QA Checklist for UI Components

Each UI component must be verified manually and automatically.

## Output Management

- `outputs/` folder is created at first run.
- `.gitignore` must ignore all files under `outputs/` except the folder itself.

## Migration Steps (Immutable)

The repository still contains legacy `app/` and `run_pcs.py`. Migration to the new `chargen/` structure must be completed systematically. Steps:

1. Copy `app/` → `chargen/` (preserve folder structure).
2. Update imports: replace `from app` → `from chargen` across all files.
3. Replace `import app` → `import chargen` where used.
4. Move entrypoint: deprecate `run_pcs.py`; replace with `chargen/studio.py`.
5. Relocate preset config to `configs/curated_models.json`.
6. Validate by running `python chargen/studio.py` and confirming UI integrity.
7. Run QA checklist to verify all controls and backend mapping.

### Migration Verification Script

```python
# tools/check_migration.py
"""Scan repo for lingering legacy imports or files from app/."""
import os, re

ROOT = os.path.dirname(os.path.dirname(__file__))
legacy_hits = []

for root, _, files in os.walk(ROOT):
    for f in files:
        if f.endswith(".py"):
            path = os.path.join(root, f)
            text = open(path, encoding="utf-8").read()
            if re.search(r"\bimport app\b", text) or re.search(r"from app", text):
                legacy_hits.append(path)

# Check for lingering app/ dir and run_pcs.py
if os.path.exists(os.path.join(ROOT, "app")):
    legacy_hits.append("app/ directory still present")
if os.path.exists(os.path.join(ROOT, "run_pcs.py")):
    legacy_hits.append("run_pcs.py still present")

if legacy_hits:
    print("[FAIL] Migration incomplete. Found:")
    for hit in legacy_hits:
        print(" -", hit)
    raise SystemExit(1)
else:
    print("[OK] Migration complete. No legacy imports or files found.")
```

Run with `python tools/check_migration.py` after migration. CI can include this script to block merges until legacy traces are removed.

---

## Recommended Features & Updates (Prioritized)

### Security Patches (apply first)

1. **Sanitize QA Reports**
2. **Harden Metadata Handling**
3. **Safe Artifact Uploads**

### Core Stability

4. **Migration Script**
5. **UI Drift Guard**
6. **Logging/Debugging Enhancements**

### Feature Enhancements

7. **Bullet-Proof Presets**
8. **ControlNet & IP-Adapter Hooks**
9. **AI-Assisted Editing**
10. **Cross-Platform Setup Scripts**

### QA & User Experience

11. **User Ratings**
12. **Rating Aggregation**
13. **GitHub Actions QA Workflow**
14. **QA Checklist**
15. **Immutable Retro UI Theme**

## Patch Notes Template (Immutable)

```
# Patch Notes — CharGen Studio

## Version: <x.y.z>
## Date: <YYYY-MM-DD>

### Security Patches
- [PATCH-ID] <short description>

### Core Stability
- [CHANGE-ID] <short description>

### Feature Enhancements
- [FEAT-ID] <short description>

### QA & User Experience
- [QA-ID] <short description>

---
```

## Version Tracking Notes (Immutable)

A separate version tracking document must be maintained in `docs/version_history.md`.

### Initial Version File Scaffold

```
# CharGen Studio — Version History

## v0.1.0 — 2025-09-23
Commit: <initial-commit-hash>
- Initial migration to chargen/ structure and immutable documentation.
```

---

## CI Hook for Version Tracking (Immutable)

Add a GitHub Actions step to auto-append version history entries on tagged releases.

```yaml
# .github/workflows/version_tracking.yml
name: Version Tracking
on:
  push:
    tags:
      - 'v*.*.*'

jobs:
  update-history:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Append version entry
        run: |
          echo "## ${GITHUB_REF#refs/tags/} — $(date +'%Y-%m-%d')" >> docs/version_history.md
          echo "Commit: $(git rev-parse --short HEAD)" >> docs/version_history.md
          echo "- Automated entry for release" >> docs/version_history.md
      - name: Commit changes
        run: |
          git config --global user.name 'github-actions'
          git config --global user.email 'github-actions@users.noreply.github.com'
          git add docs/version_history.md
          git commit -m "chore: update version history"
          git push
```

This ensures immutable, append-only version tracking.

