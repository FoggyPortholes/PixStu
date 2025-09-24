import os, json

def save_metadata(meta_path, metadata, rating=None):
    if rating is not None:
        metadata["rating"] = rating
    os.makedirs(os.path.dirname(meta_path), exist_ok=True)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
