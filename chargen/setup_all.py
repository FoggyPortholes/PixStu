"""Cross-platform setup helper for CharGen Studio.

The module can print recommended dependency commands or attempt to install the
correct PyTorch build for the detected accelerator (CUDA/ROCm/MPS/CPU).
"""

from __future__ import annotations

import argparse
import os
import platform
import subprocess
import sys
from typing import Iterable, List

from .hw_detect import detect_device

REQUIREMENT_FILES = {
    "default": "requirements.txt",
    "cuda": "requirements-cuda.txt",
    "mps": "requirements-mps.txt",
}

TORCH_CHANNELS = {
    "cuda": "https://download.pytorch.org/whl/cu124",
    "rocm": "https://download.pytorch.org/whl/rocm6.1",
    "cpu": "https://download.pytorch.org/whl/cpu",
    "mps": None,  # Apple Silicon uses the default index (universal wheel)
}


def list_available_configs() -> Iterable[str]:
    return REQUIREMENT_FILES.keys()


def install_instructions(target: str = "default") -> str:
    req = REQUIREMENT_FILES.get(target, REQUIREMENT_FILES["default"])
    return (
        "python3 -m pip install --upgrade pip\n"
        f"python3 -m pip install -r {req}\n"
    )


def _is_apple_silicon() -> bool:
    return platform.system() == "Darwin" and platform.machine().startswith("arm")


def build_torch_command(device: str | None = None) -> List[str]:
    device = device or detect_device()
    base = [sys.executable, "-m", "pip", "install", "torch", "torchvision", "torchaudio"]
    index = TORCH_CHANNELS.get(device)
    if index:
        base.extend(["--index-url", index])
    return base


def run_install(cmd: List[str]) -> None:
    subprocess.check_call(cmd)


def summarize(device: str) -> None:
    print(f"Detected accelerator: {device}")
    if device == "mps" and _is_apple_silicon():
        print("Apple Silicon detected. Ensure the following environment variable is set for best results:")
        print("  export PYTORCH_ENABLE_MPS_FALLBACK=1")
    elif device == "cuda":
        print("CUDA detected. The installer uses PyTorch wheels built for CUDA 12.4. Adjust --device if needed.")
    elif device == "rocm":
        print("ROCm detected. Wheels target ROCm 6.1 by default.")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="CharGen Studio setup helper")
    parser.add_argument(
        "--requirements", choices=list(REQUIREMENT_FILES.keys()), default="default",
        help="Print pip commands for the specified requirements profile.",
    )
    parser.add_argument(
        "--install-torch", action="store_true",
        help="Attempt to install the correct torch build for the detected accelerator.",
    )
    parser.add_argument(
        "--device",
        choices=["cuda", "rocm", "mps", "cpu"],
        help="Override hardware detection when installing torch.",
    )
    args = parser.parse_args(argv)

    print("Suggested dependency install commands:\n")
    print(install_instructions(args.requirements))

    if not args.install_torch:
        return

    device = args.device or detect_device()
    summarize(device)
    cmd = build_torch_command(device)
    print("\nRunning:", " ".join(cmd))
    run_install(cmd)


if __name__ == "__main__":
    main()
