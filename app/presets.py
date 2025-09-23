import json
import os
from .paths import CONFIGS

PRESETS_FILE = os.path.join(CONFIGS, "curated_models.json")


class Presets:
    def __init__(self, path: str = PRESETS_FILE):
        self.path = path
        self.presets = []
        self._load()

    def _load(self):
        try:
            with open(self.path, "r", encoding="utf-8-sig") as f:
                data = json.load(f)
            self.presets = data.get("presets", [])
        except Exception as e:
            print(f"[WARNING] Unable to load presets: {e}")
            self.presets = []

    def names(self):
        return [p.get("name", f"Preset {i}") for i, p in enumerate(self.presets)]

    def get(self, name: str) -> dict | None:
        return next((p for p in self.presets if p.get("name") == name), None)
