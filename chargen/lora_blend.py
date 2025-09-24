"""Utilities for storing and applying LoRA blend configurations."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Mapping, MutableMapping, Sequence

BLEND_FILE = Path(".pixstu/lora_sets.json")
MAX_LORAS = 8


def _coerce_entry(entry: object) -> dict[str, float | str]:
    path: str
    weight: float

    if isinstance(entry, Mapping):
        raw_path = entry.get("path") or entry.get("lora") or entry.get("name")
        raw_weight = entry.get("weight", entry.get("alpha", 1.0))
        if raw_path is None:
            raise ValueError("LoRA entry is missing a path")
        path = str(raw_path)
        weight = float(raw_weight)
    elif isinstance(entry, (list, tuple)) and entry:
        if len(entry) == 1:
            path = str(entry[0])
            weight = 1.0
        else:
            path = str(entry[0])
            weight = float(entry[1])
    else:
        raise ValueError(f"Unsupported LoRA row format: {entry!r}")

    return {"path": path, "weight": weight}


def _normalise_rows(rows: Iterable[object]) -> list[dict[str, float | str]]:
    normalised: list[dict[str, float | str]] = []
    for row in rows:
        entry = _coerce_entry(row)
        normalised.append(entry)
    if len(normalised) > MAX_LORAS:
        raise ValueError(f"LoRA blends are limited to {MAX_LORAS} entries")
    return normalised


def _load_all() -> dict[str, list[dict[str, float | str]]]:
    if not BLEND_FILE.exists():
        return {}
    try:
        data = json.loads(BLEND_FILE.read_text())
    except Exception:
        return {}
    if not isinstance(data, dict):
        return {}

    cleaned: dict[str, list[dict[str, float | str]]] = {}
    for key, value in data.items():
        if not isinstance(value, Sequence):
            continue
        entries: list[dict[str, float | str]] = []
        for row in value:
            try:
                entries.append(_coerce_entry(row))
            except Exception:
                continue
        cleaned[str(key)] = entries
    return cleaned


def _save_all(data: Mapping[str, Sequence[Mapping[str, float | str]]]) -> None:
    BLEND_FILE.parent.mkdir(parents=True, exist_ok=True)
    serialisable = {
        name: [dict(entry) for entry in entries]
        for name, entries in data.items()
    }
    BLEND_FILE.write_text(json.dumps(serialisable, indent=2))


def list_sets() -> list[str]:
    return sorted(_load_all().keys())


def get_set(name: str) -> list[dict[str, float | str]]:
    return [dict(entry) for entry in _load_all().get(name, [])]


def save_set(name: str, rows: Iterable[object]) -> None:
    normalised = _normalise_rows(rows)
    data = _load_all()
    data[str(name)] = normalised
    _save_all(data)


def delete_set(name: str) -> None:
    data = _load_all()
    if name in data:
        del data[name]
        _save_all(data)


def blend_to_rows(blend: Iterable[object]) -> list[list[object]]:
    rows: list[list[object]] = []
    for entry in blend:
        coerced = _coerce_entry(entry)
        rows.append([coerced["path"], float(coerced["weight"])])
    return rows


def apply_blend(preset: MutableMapping[str, object], rows: Iterable[object]) -> None:
    normalised = _normalise_rows(rows)

    loras_obj = preset.setdefault("loras", [])
    if not isinstance(loras_obj, list):
        raise TypeError("Preset 'loras' must be a list")

    loras: list[Mapping[str, object]] = loras_obj

    by_path: dict[str, dict[str, object]] = {}
    for entry in loras:
        if isinstance(entry, Mapping) and "path" in entry:
            by_path[str(entry["path"])] = entry  # type: ignore[assignment]

    for entry in normalised:
        path = str(entry["path"])
        weight = float(entry["weight"])
        existing = by_path.get(path)
        if existing is not None:
            existing["weight"] = weight
        else:
            new_entry = {"path": path, "weight": weight}
            loras.append(new_entry)  # type: ignore[arg-type]
            by_path[path] = new_entry


__all__ = [
    "BLEND_FILE",
    "MAX_LORAS",
    "apply_blend",
    "blend_to_rows",
    "delete_set",
    "get_set",
    "list_sets",
    "save_set",
]

