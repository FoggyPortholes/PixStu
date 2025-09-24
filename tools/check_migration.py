import os

ROOT = os.path.dirname(os.path.dirname(__file__))
SKIP_DIRS = {".venv", "__pycache__"}
SELF_PATH = os.path.abspath(__file__)

issues: list[str] = []

for root, dirs, files in os.walk(ROOT):
    if any(skip in os.path.relpath(root, ROOT).split(os.sep) for skip in SKIP_DIRS):
        continue
    for file in files:
        if not file.endswith(".py"):
            continue
        path = os.path.join(root, file)
        if os.path.abspath(path) == SELF_PATH:
            continue
        try:
            text = open(path, encoding="utf-8").read()
        except Exception:
            continue
        if "import app" in text or "from app" in text:
            issues.append(path)

if os.path.isdir(os.path.join(ROOT, "app")):
    issues.append("app/ directory present")
if os.path.isfile(os.path.join(ROOT, "run_pcs.py")):
    issues.append("run_pcs.py present")

if issues:
    print("[FAIL] Migration incomplete:")
    for item in issues:
        print(" -", item)
    raise SystemExit(1)

print("[OK] Migration clean")
