import json, os

PRESET_FILE = os.path.join("configs", "curated_models.json")

def load_presets():
    if not os.path.exists(PRESET_FILE):
        raise FileNotFoundError(f"Missing preset file: {PRESET_FILE}")
    with open(PRESET_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {p["name"]: p for p in data}

def get_preset_names():
    return list(load_presets().keys())

def get_preset(name: str):
    return load_presets().get(name)


def missing_assets(preset: dict):
    """Return list of missing files defined in preset['loras'] with download info."""
    missing = []
    for l in preset.get("loras", []):
        if not os.path.exists(l.get("path", "")):
            missing.append({
                "path": l.get("path"),
                "download": l.get("download"),
                "size_gb": l.get("size_gb", 0.0)
            })
    return missing
