import json
import os
from copy import deepcopy
from pathlib import Path
from typing import Dict

PRESET_FILE = os.path.join("configs", "curated_models.json")

_CACHE: Dict[str, dict] | None = None
_CACHE_MTIME: float | None = None


def _read_presets_from_disk(path: Path) -> Dict[str, dict]:
    if not path.exists():
        raise FileNotFoundError(f"Missing preset file: {path}")
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if isinstance(data, dict):
        presets = data.get("presets", [])
    elif isinstance(data, list):
        presets = data
    else:
        raise ValueError("Preset file must be a list or contain a 'presets' array")
    return {preset["name"]: preset for preset in presets}


def load_presets(*, force_reload: bool = False) -> Dict[str, dict]:
    """Return cached preset mapping, reloading when the source file changes."""

    global _CACHE, _CACHE_MTIME

    path = Path(PRESET_FILE)
    if force_reload:
        _CACHE = None
        _CACHE_MTIME = None

    current_mtime = path.stat().st_mtime if path.exists() else None

    if _CACHE is None or current_mtime != _CACHE_MTIME:
        _CACHE = _read_presets_from_disk(path)
        _CACHE_MTIME = current_mtime

    # Return a defensive copy so callers can safely mutate the result.
    return {name: deepcopy(payload) for name, payload in _CACHE.items()}


def get_preset_names() -> list[str]:
    return list(load_presets().keys())


def get_preset(name: str) -> dict | None:
    preset = load_presets().get(name)
    return deepcopy(preset) if preset is not None else None


def missing_assets(preset: dict):
    """Return list of missing files defined in preset['loras'] with download info."""
    missing = []
    for l in preset.get("loras", []):
        path = l.get("path", "") or ""
        if path and not os.path.exists(path):
            missing.append(
                {
                    "path": path,
                    "download": l.get("download"),
                    "size_gb": l.get("size_gb", 0.0),
                }
            )
    return missing
