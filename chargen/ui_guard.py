from __future__ import annotations

IMMUTABLE = {
    "tabs": {"Character Studio", "Reference Gallery", "Substitution", "Pin Editor"},
    "required_controls": {
        "Prompt",
        "Preset",
        "Seed",
        "Seed Jitter",
        "Output Size",
        "Reference Image (optional)",
        "Ref Strength (img2img)",
    },
}


def check_ui(demo):
    """Return a list of drift messages when immutable UI elements are missing."""

    def _flatten(node):
        items = []
        children = getattr(node, "children", []) or []
        label = getattr(node, "label", None)
        if label:
            items.append(label)
        for child in children:
            items.extend(_flatten(child))
        return items

    labels = set(_flatten(demo))
    drift = []
    missing_tabs = {tab for tab in IMMUTABLE["tabs"] if tab not in labels}
    for tab in missing_tabs:
        drift.append(f"[UI] Missing tab: {tab}")
    for control in IMMUTABLE["required_controls"]:
        if control not in labels:
            drift.append(f"[UI] Missing control: {control}")
    return drift
