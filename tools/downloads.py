#!/usr/bin/env python3
from __future__ import annotations
from pathlib import Path
from typing import Optional, Union
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

PathLike = Union[str, Path]


def _normalise_path(filename: PathLike) -> Path:
    return LORAS_DIR / Path(filename).name


def have_lora(filename: PathLike) -> bool:
    return _normalise_path(filename).exists()

def download_lora(
    filename: PathLike,
    repo_id: Optional[str] = None,
    repo_filename: Optional[str] = None,
) -> str:
    filename = Path(filename)
    ensure_hf()
    if have_lora(filename):
        return str(_normalise_path(filename))
    if repo_id is None or repo_filename is None:
        key = filename.name
        if repo_id is None and repo_filename is None:
            if key not in KNOWN_LORAS:
                raise FileNotFoundError(f"No registry entry for {key}.")
            repo_id, repo_filename = KNOWN_LORAS[key]
        else:
            repo_filename = repo_filename or key
    tmp_path = hf_hub_download(repo_id=repo_id, filename=repo_filename)
    dest = _normalise_path(filename)
    shutil.copy2(tmp_path, dest)
    return str(dest)

def resolve_missing_loras(filenames: list[PathLike]) -> list[str]:
    return [str(Path(f)) for f in filenames if not have_lora(f)]
