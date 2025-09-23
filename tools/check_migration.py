"""Sanity check to ensure legacy PixStu artifacts are removed."""

from __future__ import annotations

import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(__file__))
LEGACY_DIR = os.path.join(ROOT, "app")
LEGACY_ENTRYPOINT = os.path.join(ROOT, "run_pcs.py")
IGNORED_FILES = {
    os.path.join(ROOT, "run_sprite_sheet_studio.py"),
    os.path.join(ROOT, "tests", "test_sprite_sheet_builder.py"),
    os.path.join(ROOT, "app", "sprite_sheet_studio.py"),
    os.path.join(ROOT, "app", "sprite_sheet_builder.py"),
}

LEGACY_PATTERNS = (
    re.compile(r"\bimport app\b"),
    re.compile(r"from\s+app\b"),
)


def main() -> int:
    issues: list[str] = []

    for root, _dirs, files in os.walk(ROOT):
        if ".venv" in root or "__pycache__" in root or ".git" in root:
            continue
        for filename in files:
            if not filename.endswith(".py"):
                continue
            path = os.path.join(root, filename)
            if path in IGNORED_FILES:
                continue
            try:
                with open(path, "r", encoding="utf-8") as handle:
                    text = handle.read()
            except Exception:
                continue
            if any(pattern.search(text) for pattern in LEGACY_PATTERNS):
                issues.append(path)

    if os.path.isdir(LEGACY_DIR):
        unsupported = []
        for entry in os.listdir(LEGACY_DIR):
            if not entry.endswith(".py"):
                continue
            path = os.path.join(LEGACY_DIR, entry)
            if path in IGNORED_FILES:
                continue
            try:
                with open(path, "r", encoding="utf-8") as handle:
                    text = handle.read()
                if "chargen" not in text:
                    unsupported.append(f"app/{entry}")
            except Exception:
                continue
        if unsupported:
            issues.append("Legacy directory contains non-shim files:")
            issues.extend(f"  - {item}" for item in unsupported)

    if os.path.isfile(LEGACY_ENTRYPOINT):
        try:
            with open(LEGACY_ENTRYPOINT, "r", encoding="utf-8") as handle:
                entry_text = handle.read()
            if "chargen" not in entry_text:
                issues.append("Legacy entrypoint detected: run_pcs.py")
        except Exception:
            issues.append("Legacy entrypoint detected: run_pcs.py")

    if issues:
        print("[FAIL] Migration incomplete. Resolve the following:")
        for item in issues:
            print(f" - {item}")
        return 1

    print("[OK] Migration complete. No legacy imports or files found.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
