"""
Hugging Face asset helper for LoRA weights.
"""
from pathlib import Path
import shutil

try:
    from huggingface_hub import hf_hub_download
except Exception:  # pragma: no cover - optional dependency
    hf_hub_download = None

LORAS_DIR = Path("loras")
LORAS_DIR.mkdir(exist_ok=True)

KNOWN_LORAS = {
    "western_cartoon.safetensors": (
        "FoggyPortholes/western-cartoon-lora",
        "western_cartoon.safetensors",
    )
}


def have_lora(name: str) -> bool:
    return (LORAS_DIR / name).exists()


def download_lora(fname: str, repo_id: str | None = None, repo_fname: str | None = None) -> str:
    if hf_hub_download is None:
        raise RuntimeError("pip install huggingface_hub")

    if have_lora(fname):
        return str(LORAS_DIR / fname)

    if not (repo_id and repo_fname):
        if fname not in KNOWN_LORAS:
            raise FileNotFoundError(fname)
        repo_id, repo_fname = KNOWN_LORAS[fname]

    tmp = hf_hub_download(repo_id=repo_id, filename=repo_fname)
    dest = LORAS_DIR / fname
    shutil.copy2(tmp, dest)
    return str(dest)
