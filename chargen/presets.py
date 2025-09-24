import json, os

PRESET_FILE = os.path.join("configs", "curated_models.json")


def load_presets():
    if not os.path.exists(PRESET_FILE):
        raise FileNotFoundError(f"Missing preset file: {PRESET_FILE}")
    with open(PRESET_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        presets = data.get("presets", [])
    elif isinstance(data, list):
        presets = data
    else:
        raise ValueError("Preset file must be a list or contain a 'presets' array")
    return {p["name"]: p for p in presets}


def get_preset_names():
    return list(load_presets().keys())


def get_preset(name: str):
    return load_presets().get(name)


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
