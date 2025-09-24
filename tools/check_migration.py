import os

ROOT = os.path.dirname(os.path.dirname(__file__))
issues: list[str] = []

for root, _, files in os.walk(ROOT):
    for file in files:
        if not file.endswith(".py"):
            continue
        path = os.path.join(root, file)
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
