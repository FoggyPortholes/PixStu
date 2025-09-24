import os
import sys

ROOT = os.path.dirname(os.path.dirname(__file__))

checks: list[bool] = []


def expect_file(path: str, desc: str) -> bool:
    full = os.path.join(ROOT, path)
    if os.path.exists(full):
        print(f"[OK] {desc} ({path})")
        return True
    print(f"[FAIL] {desc} missing: {path}")
    return False


checks.append(expect_file("chargen/ui_guard.py", "UI Guard"))
checks.append(expect_file("chargen/generator.py", "BulletProofGenerator"))
checks.append(expect_file("chargen/presets.py", "Preset loader"))
checks.append(expect_file("chargen/studio.py", "Studio UI"))
checks.append(expect_file("chargen/metadata.py", "Metadata module"))
checks.append(expect_file("chargen/substitution.py", "Substitution module"))
checks.append(expect_file("chargen/pin_editor.py", "Pin Editor module"))
for tool in ["tools/test_presets.py", "tools/check_migration.py", "tools/sanitize_reports.py"]:
    checks.append(expect_file(tool, tool))
checks.append(expect_file("configs/curated_models.json", "Curated presets"))

for directory in ["loras", "outputs"]:
    full = os.path.join(ROOT, directory)
    if os.path.isdir(full):
        print(f"[OK] Directory {directory} exists")
    else:
        print(f"[FAIL] Directory {directory} missing")
        checks.append(False)

if not all(checks):
    sys.exit(1)

print("[VERIFY] Repo structure appears valid.")
