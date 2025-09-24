#!/usr/bin/env python3
from __future__ import annotations
from pathlib import Path
from typing import Optional
import shutil

try:
    from huggingface_hub import hf_hub_download
except Exception:
    hf_hub_download = None  # type: ignore

LORAS_DIR = Path("loras"); LORAS_DIR.mkdir(parents=True, exist_ok=True)

KNOWN_LORAS = {
    "western_cartoon.safetensors": (
        "FoggyPortholes/western-cartoon-lora", "western_cartoon.safetensors",
    ),
}

def ensure_hf():
    if hf_hub_download is None:
        raise RuntimeError("huggingface_hub not installed. pip install huggingface_hub")

def have_lora(filename: str) -> bool:
    return (LORAS_DIR / filename).exists()

def download_lora(filename: str, repo_id: Optional[str] = None, repo_filename: Optional[str] = None) -> str:
    ensure_hf()
    if have_lora(filename):
        return str(LORAS_DIR / filename)
    if repo_id is None or repo_filename is None:
        if filename not in KNOWN_LORAS:
            raise FileNotFoundError(f"No registry entry for {filename}.")
        repo_id, repo_filename = KNOWN_LORAS[filename]
    tmp_path = hf_hub_download(repo_id=repo_id, filename=repo_filename)
    dest = LORAS_DIR / filename
    shutil.copy2(tmp_path, dest)
    return str(dest)

def resolve_missing_loras(filenames: list[str]) -> list[str]:
    return [f for f in filenames if not have_lora(f)]
