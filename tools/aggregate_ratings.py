"""Aggregate user ratings from metadata JSON files."""

from __future__ import annotations

import json
import os
from collections import defaultdict
from typing import DefaultDict, List

try:
    from chargen.model_setup import OUTPUTS as OUTPUTS_DIR
except Exception:  # pragma: no cover - fallback when run outside package context
    OUTPUTS_DIR = os.path.join(os.getcwd(), "outputs")


def collect_ratings() -> DefaultDict[str, List[int]]:
    ratings: DefaultDict[str, List[int]] = defaultdict(list)
    for root, _dirs, files in os.walk(OUTPUTS_DIR):
        for filename in files:
            if not filename.endswith(".json"):
                continue
            path = os.path.join(root, filename)
            try:
                with open(path, "r", encoding="utf-8") as handle:
                    payload = json.load(handle)
            except Exception:
                continue
            preset = payload.get("preset", "unknown")
            rating = payload.get("rating")
            if isinstance(rating, (int, float)):
                ratings[preset].append(int(rating))
    return ratings


def main() -> None:
    ratings = collect_ratings()
    if not ratings:
        print("No ratings found.")
        return
    for preset, values in ratings.items():
        if not values:
            continue
        average = sum(values) / len(values)
        print(f"{preset}: avg {average:.2f} from {len(values)} ratings")


if __name__ == "__main__":
    main()
