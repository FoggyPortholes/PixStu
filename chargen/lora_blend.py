"""Utilities for managing reusable LoRA weight blends."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Iterable, List, Sequence

from pixstu_config import PIXSTU_DIR


MAX_LORAS = 3
BLEND_FILE = PIXSTU_DIR / "lora_sets.json"


@dataclass(slots=True)
class BlendEntry:
    path: str
    weight: float


def _ensure_file() -> None:
    PIXSTU_DIR.mkdir(parents=True, exist_ok=True)
    if not BLEND_FILE.exists():
        with BLEND_FILE.open("w", encoding="utf-8") as handle:
            json.dump({}, handle)


def _load_raw() -> dict:
    _ensure_file()
    with BLEND_FILE.open("r", encoding="utf-8") as handle:
        try:
            payload = json.load(handle)
        except json.JSONDecodeError:
            payload = {}
    if not isinstance(payload, dict):
        payload = {}
    return payload


def _save_raw(payload: dict) -> None:
    _ensure_file()
    with BLEND_FILE.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def list_sets() -> List[str]:
    """Return the names of all stored blends."""

    payload = _load_raw()
    return sorted(str(name) for name in payload.keys())


def get_set(name: str) -> List[BlendEntry]:
    payload = _load_raw()
    entries = payload.get(name, [])
    result: List[BlendEntry] = []
    for item in entries:
        if not isinstance(item, dict):
            continue
        path = str(item.get("path", ""))
        if not path:
            continue
        try:
            weight = float(item.get("weight", 0.0))
        except (TypeError, ValueError):
            weight = 0.0
        result.append(BlendEntry(path=path, weight=weight))
    return result


def delete_set(name: str) -> None:
    payload = _load_raw()
    if name in payload:
        del payload[name]
        _save_raw(payload)


def _normalise_rows(rows: Iterable[Iterable[object]] | object) -> List[BlendEntry]:
    normalised: List[BlendEntry] = []
    if rows is None:
        return normalised

    if hasattr(rows, "to_numpy"):
        rows = rows.to_numpy().tolist()  # type: ignore[assignment]
    elif hasattr(rows, "tolist"):
        rows = rows.tolist()  # type: ignore[assignment]

    try:
        iterator = list(rows)  # type: ignore[arg-type]
    except TypeError:
        return normalised

    for row in iterator:
        if row is None:
            continue
        if isinstance(row, dict):
            candidate = [row.get("path"), row.get("weight")]
        else:
            candidate = list(row) if isinstance(row, (list, tuple)) else [row]
        if not candidate:
            continue
        path = str(candidate[0]).strip()
        if not path:
            continue
        try:
            weight = float(candidate[1]) if len(candidate) > 1 else 1.0
        except (TypeError, ValueError):
            weight = 1.0
        normalised.append(BlendEntry(path=path, weight=weight))
        if len(normalised) > MAX_LORAS:
            raise ValueError(f"A blend may include at most {MAX_LORAS} LoRAs")
    return normalised


def save_set(name: str, rows: Iterable[Iterable[object]] | object) -> None:
    if not name or not str(name).strip():
        raise ValueError("Blend name must not be empty")
    entries = _normalise_rows(rows)
    payload = _load_raw()
    payload[str(name)] = [{"path": entry.path, "weight": entry.weight} for entry in entries]
    _save_raw(payload)


def blend_to_rows(entries: Sequence[BlendEntry]) -> List[List[object]]:
    return [[entry.path, entry.weight] for entry in entries]


def apply_blend(preset: dict, rows: Iterable[Iterable[object]] | object) -> None:
    """Merge ``rows`` into ``preset['loras']`` respecting weight overrides."""

    if preset is None:
        return
    entries = _normalise_rows(rows)
    if not entries:
        return

    loras = preset.setdefault("loras", [])
    lookup = {}
    for idx, entry in enumerate(loras):
        key = entry.get("display_path") or entry.get("path") or entry.get("resolved_path")
        if key:
            lookup[str(key)] = idx

    for blend in entries:
        resolved = os.path.abspath(blend.path)
        target = None
        for candidate in (blend.path, resolved):
            if candidate in lookup:
                target = loras[lookup[candidate]]
                break
        if target is None:
            target = {
                "path": blend.path,
                "display_path": blend.path,
                "resolved_path": resolved,
            }
            loras.append(target)
        target["weight"] = blend.weight


__all__ = [
    "BLEND_FILE",
    "BlendEntry",
    "MAX_LORAS",
    "apply_blend",
    "blend_to_rows",
    "delete_set",
    "get_set",
    "list_sets",
    "save_set",
]
