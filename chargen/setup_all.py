"""Cross-platform environment setup helpers."""

import os
import sys
from typing import Iterable

REQUIREMENT_FILES = {
    "default": "requirements.txt",
    "cuda": "requirements-cuda.txt",
    "mps": "requirements-mps.txt",
}


def list_available_configs() -> Iterable[str]:
    return REQUIREMENT_FILES.keys()


def install_instructions(target: str = "default") -> str:
    req = REQUIREMENT_FILES.get(target, REQUIREMENT_FILES["default"])
    return (
        "python3 -m pip install --upgrade pip\n"
        f"python3 -m pip install -r {req}\n"
    )


def main(args: list[str]) -> None:
    target = args[1] if len(args) > 1 else "default"
    print("Suggested install commands:\n")
    print(install_instructions(target))


if __name__ == "__main__":
    main(sys.argv)
