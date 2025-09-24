"""Centralised loader for PixStu runtime configuration.

This module owns the ``.pixstu/config.json`` file which stores optional
runtime policies for features such as gallery retention and cache sizing.
When the configuration file is missing a sensible default is written to
disk so downstream modules can rely on the structure always being present.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

_REPO_ROOT = Path(__file__).resolve().parent
PIXSTU_DIR = _REPO_ROOT / ".pixstu"
CONFIG_PATH = PIXSTU_DIR / "config.json"

_DEFAULT_CONFIG: Dict[str, Any] = {
    "gallery": {
        "max_items": 24,
        "ttl_days": 30,
    },
    "cache": {
        "ttl_seconds": 7 * 24 * 3600,
        "max_entries": 128,
        "max_bytes": 512 * 1024 * 1024,
    },
}


def _ensure_dirs() -> None:
    PIXSTU_DIR.mkdir(parents=True, exist_ok=True)


def _write_default() -> None:
    _ensure_dirs()
    with CONFIG_PATH.open("w", encoding="utf-8") as handle:
        json.dump(_DEFAULT_CONFIG, handle, indent=2, sort_keys=True)
        handle.write("\n")


def load_config() -> Dict[str, Any]:
    """Return the runtime configuration, creating defaults if necessary."""

    if not CONFIG_PATH.exists():
        _write_default()
        return json.loads(json.dumps(_DEFAULT_CONFIG))

    with CONFIG_PATH.open("r", encoding="utf-8") as handle:
        try:
            data = json.load(handle)
        except json.JSONDecodeError:
            # If the file becomes corrupted fall back to defaults to avoid
            # breaking the UI.  The default file is rewritten so the user can
            # recover easily.
            _write_default()
            return json.loads(json.dumps(_DEFAULT_CONFIG))

    if not isinstance(data, dict):
        _write_default()
        return json.loads(json.dumps(_DEFAULT_CONFIG))

    # Merge missing keys from the defaults without overwriting user values.
    merged: Dict[str, Any] = json.loads(json.dumps(_DEFAULT_CONFIG))
    for key, value in data.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key].update(value)
        else:
            merged[key] = value
    return merged


def save_config(config: Dict[str, Any]) -> None:
    """Persist ``config`` back to ``.pixstu/config.json``."""

    _ensure_dirs()
    with CONFIG_PATH.open("w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2, sort_keys=True)
        handle.write("\n")


__all__ = ["CONFIG_PATH", "PIXSTU_DIR", "load_config", "save_config"]
