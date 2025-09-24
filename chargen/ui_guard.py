IMMUTABLE = {
    "tabs": {"Character Studio", "Reference Gallery", "Substitution", "Pin Editor", "Downloads"},
    "required_controls": {"Prompt", "Preset", "Seed", "Seed Jitter", "Output Size", "Reference Image (optional)", "Ref Strength (img2img)"}
}

def check_ui(demo):
    def flatten(node):
        out = []
        children = getattr(node, "children", []) or []
        label = getattr(node, "label", None)
        if label:
            out.append(label)
        for ch in children:
            out.extend(flatten(ch))
        return out
    labels = set(flatten(demo))
    drift = []
    for t in IMMUTABLE["tabs"]:
        if t not in labels:
            drift.append(f"[UI] Missing tab: {t}")
    for rc in IMMUTABLE["required_controls"]:
        if rc not in labels:
            drift.append(f"[UI] Missing control: {rc}")
    return drift
