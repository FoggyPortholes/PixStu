import os
import re

ROOT = os.path.dirname(os.path.dirname(__file__))
issues = []
for r, _, fs in os.walk(ROOT):
    for f in fs:
        if f.endswith(".py"):
            p = os.path.join(r, f)
            with open(p, encoding="utf-8") as handle:
                s = handle.read()
            if re.search(r"\bimport app\b", s) or re.search(r"from app", s):
                issues.append(p)
if os.path.exists(os.path.join(ROOT, "app")):
    issues.append("app/ directory present")
if os.path.exists(os.path.join(ROOT, "run_pcs.py")):
    issues.append("run_pcs.py present")
if issues:
    print("[FAIL] Migration incomplete:")
    for item in issues:
        print(" -", item)
    raise SystemExit(1)
print("[OK] Migration clean")
