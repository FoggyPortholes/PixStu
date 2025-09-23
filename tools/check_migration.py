import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(__file__))
LEGACY_IMPORT = re.compile(r"\bimport app\b")
LEGACY_FROM = re.compile(r"from\s+app\b")
SKIP_DIRS = {".git", "__pycache__", ".venv"}


def _should_skip(path: str) -> bool:
    parts = set(part for part in path.split(os.sep) if part)
    return bool(parts & SKIP_DIRS)


def _scan_python_sources() -> list[str]:
    issues: list[str] = []
    for root, _dirs, files in os.walk(ROOT):
        if _should_skip(root):
            continue
        for filename in files:
            if not filename.endswith(".py"):
                continue
            path = os.path.join(root, filename)
            try:
                with open(path, "r", encoding="utf-8") as handle:
                    text = handle.read()
            except Exception:
                continue
            if LEGACY_IMPORT.search(text) or LEGACY_FROM.search(text):
                issues.append(path)
    return issues


def _check_legacy_files(issues: list[str]) -> None:
    legacy_dir = os.path.join(ROOT, "app")
    if os.path.isdir(legacy_dir):
        issues.append("app/ directory present")
    legacy_entry = os.path.join(ROOT, "run_pcs.py")
    if os.path.isfile(legacy_entry):
        try:
            with open(legacy_entry, "r", encoding="utf-8") as handle:
                text = handle.read()
            if "chargen" not in text:
                issues.append("run_pcs.py present")
        except Exception:
            issues.append("run_pcs.py present")


def main() -> int:
    issues = _scan_python_sources()
    _check_legacy_files(issues)
    if issues:
        print("[FAIL] Migration incomplete:")
        for item in issues:
            print(" -", item)
        return 1
    print("[OK] Migration complete. No legacy imports or files found.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
