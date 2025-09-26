"""
WAN2.2-style presets + character generator helpers.
- Loads default presets from assets/presets/wan22.json and user presets from .pixstu/presets.json
- Provides merge/apply utilities and prompt synthesis from selected traits
"""
from __future__ import annotations
from pathlib import Path
from typing import Dict, Any, List
import json

DEFAULT_PATH = Path("assets/presets/wan22.json")
USER_PATH = Path(".pixstu/presets.json")

Preset = Dict[str, Any]

def _load(path: Path) -> List[Preset]:
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return [p for p in data if isinstance(p, dict) and p.get("name")] if isinstance(data, list) else []
    except Exception:
        return []

def load_presets() -> List[Preset]:
    base = _load(DEFAULT_PATH)
    usr = _load(USER_PATH)
    # Dedup by name, user overrides default
    d: Dict[str, Preset] = {p["name"]: p for p in base}
    d.update({p["name"]: p for p in usr})
    return list(d.values())

def save_user_presets(presets: List[Preset]) -> None:
    USER_PATH.parent.mkdir(parents=True, exist_ok=True)
    USER_PATH.write_text(json.dumps(presets, indent=2), encoding="utf-8")

# --- Generator helpers ---

def synthesize_prompt(base_prompt: str, style: str | None, traits: Dict[str, str]) -> str:
    parts = [base_prompt.strip()]
    # Attach style as tag if present
    if style:
        parts.append(f"style:{style}")
    # Append chosen traits deterministically in a neat order
    for key in sorted(traits.keys()):
        val = traits[key]
        if key and val:
            parts.append(f"{key}:{val}")
    return ", ".join([p for p in parts if p])

def apply_preset_to_params(preset: Preset) -> Dict[str, Any]:
    return {
        "prompt": preset.get("prompt", ""),
        "negative": preset.get("negative", ""),
        "steps": int(preset.get("steps", 28)),
        "cfg_scale": float(preset.get("cfg_scale", 7.0)),
        "width": int(preset.get("width", 640)),
        "height": int(preset.get("height", 640)),
        "loras": [(l.get("path"), float(l.get("weight", 1.0))) for l in preset.get("loras", []) if l.get("path")],
    }

def preset_trait_options(preset: Preset) -> Dict[str, List[str]]:
    return {k: list(v) for k, v in (preset.get("traits") or {}).items() if isinstance(v, list)}
