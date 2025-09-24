import json
import os
from collections import defaultdict

OUTPUTS_DIR = "outputs"


def collect_ratings():
    ratings = defaultdict(list)
    for root, _, files in os.walk(OUTPUTS_DIR):
        for file in files:
            if not file.endswith(".json"):
                continue
            path = os.path.join(root, file)
            try:
                data = json.load(open(path, encoding="utf-8"))
            except Exception:
                continue
            preset = data.get("preset", "unknown")
            if "rating" in data:
                ratings[preset].append(data["rating"])
    return ratings


def main():
    ratings = collect_ratings()
    for preset, values in ratings.items():
        if not values:
            continue
        average = sum(values) / len(values)
        print(f"{preset}: avg {average:.2f} from {len(values)} ratings")


if __name__ == "__main__":
    main()
