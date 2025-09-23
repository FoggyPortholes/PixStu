import json
import os
import time
from typing import Any, Dict, Optional

from .model_setup import OUTPUTS


def _target_dir(output_dir: Optional[str]) -> str:
    return output_dir or OUTPUTS


def save_metadata(output_dir: str, meta: Dict[str, Any], rating: Optional[int] = None) -> str:
    payload = dict(meta)
    if rating is not None:
        payload["rating"] = rating
    directory = _target_dir(output_dir)
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, f"meta_{int(time.time())}.json")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
    return path


def update_rating(meta_path: str, rating: int) -> None:
    if not meta_path or not os.path.exists(meta_path):
        raise FileNotFoundError(f"Metadata file not found: {meta_path}")
    with open(meta_path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    data["rating"] = int(rating)
    with open(meta_path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False)
