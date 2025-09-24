#!/usr/bin/env python3
"""
Asset downloader for PixStu (LoRA weights, etc.)
- Uses huggingface_hub to fetch files by repo_id + filename.
- Places results under the repo's local asset directories (e.g., loras/).
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional
import shutil

try:
    from huggingface_hub import hf_hub_download
except Exception:  # keep import optional
    hf_hub_download = None  # type: ignore[assignment]

LORAS_DIR = Path("loras")
LORAS_DIR.mkdir(parents=True, exist_ok=True)

# Registry of known LoRA assets used by presets (extend as needed)
KNOWN_LORAS = {
    "western_cartoon.safetensors": (
        "FoggyPortholes/western-cartoon-lora",
        "western_cartoon.safetensors",
    ),
    # Add more known items here if presets reference them
}


def ensure_hf() -> None:
    if hf_hub_download is None:
        raise RuntimeError(
            "huggingface_hub not installed. Install with: pip install huggingface_hub"
        )


def have_lora(filename: str) -> bool:
    return (LORAS_DIR / filename).exists()


def download_lora(
    filename: str, repo_id: Optional[str] = None, repo_filename: Optional[str] = None
) -> str:
    """Download a LoRA file into loras/ if missing."""

    ensure_hf()
    if have_lora(filename):
        return str(LORAS_DIR / filename)

    if repo_id is None or repo_filename is None:
        if filename not in KNOWN_LORAS:
            raise FileNotFoundError(
                f"No registry entry for {filename}. Provide repo_id/repo_filename."
            )
        repo_id, repo_filename = KNOWN_LORAS[filename]

    tmp_path = hf_hub_download(repo_id=repo_id, filename=repo_filename)  # type: ignore[misc]
    dest = LORAS_DIR / filename
    shutil.copy2(tmp_path, dest)
    return str(dest)


def resolve_missing_loras(filenames: list[str]) -> list[str]:
    return [name for name in filenames if not have_lora(name)]

