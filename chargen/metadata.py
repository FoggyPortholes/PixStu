import json
import os


def save_metadata(meta_path: str, metadata: dict, rating=None) -> None:
    if rating is not None:
        metadata["rating"] = rating
    os.makedirs(os.path.dirname(meta_path), exist_ok=True)
    with open(meta_path, "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
