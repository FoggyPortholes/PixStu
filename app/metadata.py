import json
import os
import time
from typing import Any, Dict
from .paths import OUTPUTS


def save_metadata(output_dir: str, meta: Dict[str, Any]) -> str:
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"meta_{int(time.time())}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    return path
