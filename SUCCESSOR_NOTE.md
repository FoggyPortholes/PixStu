# PixStu v2.2.4 — Successor Note

## Cleanup Changes
- Relaxed guardrails: now warnings unless PIXSTU_STRICT=1
- Centralized seeding logic in tools/util.py
- Requirements pinned to tested versions
- Added guardrail tests
- UI now surfaces errors instead of crashing the app

## Quickstart
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python tools/test_ui_smoke.py
pytest tools/test_guardrails.py
python -m pixstu.app.studio
