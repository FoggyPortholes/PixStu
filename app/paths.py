import os

ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
CONFIGS = os.path.join(ROOT, "configs")
OUTPUTS = os.path.join(ROOT, "outputs")
GALLERY = os.path.join(ROOT, "reference_gallery")
MODELS = os.path.join(ROOT, "models")
LORAS = os.path.join(MODELS, "lora")

for d in [OUTPUTS, GALLERY, LORAS]:
    os.makedirs(d, exist_ok=True)
