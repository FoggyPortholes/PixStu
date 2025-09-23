import json
import os
from typing import Dict, List, Optional

from . import model_setup

PRESETS_FILE = os.path.join(model_setup.CONFIGS, "curated_models.json")


class Presets:
    def __init__(self, path: str = PRESETS_FILE):
        self.path = path
        self.presets: List[Dict] = []
        self._load()

    def _normalize_lora(self, entry: Dict) -> Dict:
        entry = dict(entry or {})
        path_hint = entry.get("path") or entry.get("local_path") or ""
        record = model_setup.resolve_path(path_hint) if path_hint else None
        if record is None and entry.get("name"):
            record = model_setup.find_record(entry["name"])
        if record is not None:
            entry["path"] = record.path
            entry["local_path"] = record.local_path
            entry["exists"] = record.exists
            entry.setdefault("name", record.name)
            if record.repo_id:
                entry.setdefault("repo_id", record.repo_id)
            if record.preview_url:
                entry.setdefault("preview_url", record.preview_url)
        else:
            if path_hint:
                abs_path = path_hint if os.path.isabs(path_hint) else os.path.join(model_setup.MODELS, path_hint)
                entry["local_path"] = abs_path
                entry["exists"] = os.path.exists(abs_path)
        entry.setdefault("weight", 1.0)
        return entry

    def _load(self):
        try:
            with open(self.path, "r", encoding="utf-8-sig") as handle:
                data = json.load(handle)
            raw_presets = data.get("presets", [])
        except Exception as exc:
            print(f"[WARNING] Unable to load presets: {exc}")
            raw_presets = []
        self.presets = []
        for preset in raw_presets:
            preset = dict(preset)
            loras = [self._normalize_lora(entry) for entry in preset.get("loras", [])]
            preset["loras"] = loras
            self.presets.append(preset)

    def names(self) -> List[str]:
        return [p.get("name", f"Preset {i}") for i, p in enumerate(self.presets)]

    def get(self, name: str) -> Optional[Dict]:
        return next((p for p in self.presets if p.get("name") == name), None)

    def describe(self, name: str) -> str:
        preset = self.get(name)
        if not preset:
            return "Select a preset to view details."
        lines = []
        base = preset.get("base_model", "Unknown base model")
        lines.append(f"**Base model:** `{base}`")
        suggested = preset.get("suggested", {})
        if suggested:
            lines.append(
                "**Suggested:** steps={steps}, guidance={guidance}".format(
                    steps=suggested.get("steps", "?"), guidance=suggested.get("guidance", "?")
                )
            )
        desc = preset.get("description")
        if desc:
            lines.append(desc)
        loras = preset.get("loras", [])
        if loras:
            lines.append("**LoRAs:**")
            for entry in loras:
                status = "available" if entry.get("exists") else "downloadable" if entry.get("repo_id") else "missing"
                weight = entry.get("weight", 1.0)
                name = entry.get("name") or os.path.basename(entry.get("path", ""))
                lines.append(f"- `{name}` (weight {weight}) ? {status}")
        else:
            lines.append("**LoRAs:** none")
        return "\n".join(lines)


_DEFAULT_PRESETS: Presets | None = None


def get_preset_names() -> List[str]:
    global _DEFAULT_PRESETS
    if _DEFAULT_PRESETS is None:
        _DEFAULT_PRESETS = Presets()
    return _DEFAULT_PRESETS.names()
