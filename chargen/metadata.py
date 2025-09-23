import json
import os
import time
from typing import Any, Dict

from .model_setup import OUTPUTS


def save_metadata(output_dir: str, meta: Dict[str, Any]) -> str:
    os.makedirs(output_dir or OUTPUTS, exist_ok=True)
    path = os.path.join(output_dir or OUTPUTS, f"meta_{int(time.time())}.json")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(meta, handle, indent=2, ensure_ascii=False)
    return path
