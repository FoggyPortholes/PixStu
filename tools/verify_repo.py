#!/usr/bin/env python3
from pathlib import Path
import sys

EXPECTED = [
    'chargen/inpaint.py',
    'chargen/studio.py',
    'chargen/lora_blend.py',
    'tools/cache.py',
]

missing = [p for p in EXPECTED if not Path(p).exists()]
if missing:
    print('[PixStu] Missing critical files:', ', '.join(missing))
    sys.exit(1)
print('[PixStu] Repo structure OK')
