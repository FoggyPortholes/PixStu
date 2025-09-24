import json
import os
from copy import deepcopy
from pathlib import Path
from typing import Dict


_REPO_ROOT = Path(__file__).resolve().parent.parent

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


def _resolve_asset_path(path: str) -> str:
    """Return an absolute path for preset assets, respecting PCS_MODELS_ROOT.

    Presets in the repository reference assets relative to the project root
    (for example ``loras/example.safetensors``).  When the portable launcher is
    used we instead place assets underneath ``models/`` and expose that
    directory via the ``PCS_MODELS_ROOT`` environment variable.  To keep the
    presets portable we normalise any configured path against a small set of
    candidate roots and return the first existing absolute path.
    """

    if not path:
        return ""

    original = Path(path)
    if original.is_absolute():
        return str(original)

    candidates = []

    # 1) Relative to the current working directory.
    candidates.append(Path.cwd() / original)

    # 2) Relative to PCS_MODELS_ROOT if provided (portable launcher).
    models_root = os.getenv("PCS_MODELS_ROOT")
    if models_root:
        candidates.append(Path(models_root) / original)

    # 3) Relative to the repository root (development checkout).
    candidates.append(_REPO_ROOT / original)

    for candidate in candidates:
        if candidate.exists():
            return str(candidate.resolve())

    # Fall back to an absolute path under the first candidate so downstream
    # consumers receive a consistent absolute location even when missing.
    return str(candidates[0].resolve()) if candidates else str(original.resolve())


def _normalise_paths(preset: dict | None) -> dict | None:
    if preset is None:
        return None

    for entry in preset.get("loras", []):
        original = entry.get("path", "")
        entry["display_path"] = original
        entry["resolved_path"] = _resolve_asset_path(original)
    for entry in preset.get("controlnets", []):
        local_dir = entry.get("local_dir") or entry.get("path")
        if local_dir:
            entry["display_local_dir"] = local_dir
            resolved_local = _resolve_asset_path(local_dir)
            entry["local_dir_resolved"] = resolved_local
            entry["local_dir"] = resolved_local
    return preset


def get_preset(name: str) -> dict | None:
    preset = deepcopy(load_presets().get(name))
    return _normalise_paths(preset)


def missing_assets(preset: dict):
    """Return list of missing files defined in preset['loras'] with download info."""
    missing = []
    for l in preset.get("loras", []):
        resolved = l.get("resolved_path") or _resolve_asset_path(l.get("path", "") or "")
        display = l.get("display_path") or l.get("path") or resolved
        if resolved and not os.path.exists(resolved):
            missing.append(
                {
                    "display_path": display,
                    "resolved_path": resolved,
                    "download": l.get("download"),
                    "size_gb": l.get("size_gb", 0.0),
                }
            )
    return missing
