# CharGen Studio — QA Checklist

## Character Studio
- [ ] Prompt field accepts text and appends preset style descriptors.
- [ ] Preset dropdown lists every entry from `configs/curated_models.json`.
- [ ] Seed and jitter controls update generation determinism.
- [ ] Size slider produces correctly sized outputs (256–1024).
- [ ] ControlNet plugin enables/disables, accepts conditioning map, and reflects status updates.
- [ ] IP-Adapter plugin loads reference image and adjusts influence scale.
- [ ] Live Preview updates while rendering and clears on new runs.
- [ ] Final image, metadata path, and status populate after completion.
- [ ] Rating controls unlock post-generation and persist to metadata JSON.
- [ ] Diagnostics toggle reveals log path and live log tail.

## AI Edit (Experimental)
- [ ] Edit prompt + reference image produce an edited output.
- [ ] Auto-mask generates a reasonable mask; manual mask overrides when provided.
- [ ] Target region options (upper/lower/left/right) limit the edited area.
- [ ] Edit metadata and status display correctly.

## Reference Gallery
- [ ] Gallery thumbnails render and can be selected as reference images.

## Output & Logging
- [ ] `outputs/` contains generated PNGs and associated metadata JSONs.
- [ ] `logs/chargen.log` rotates and includes generation events.
- [ ] `python tools/aggregate_ratings.py` reports averages when ratings exist.

## Automation
- [ ] `python -m compileall chargen` completes without errors.
- [ ] `python tools/check_migration.py` returns `[OK]`.
- [ ] GitHub Actions QA workflow passes on main branch and pull requests.
