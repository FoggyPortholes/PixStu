import os, json
from collections import defaultdict

OUTPUTS = "outputs"

def collect_ratings():
    ratings = defaultdict(list)
    for root, _, files in os.walk(OUTPUTS):
        for f in files:
            if f.endswith(".json"):
                try:
                    data = json.load(open(os.path.join(root, f), encoding="utf-8"))
                    preset = data.get("preset", "unknown")
                    if "rating" in data:
                        ratings[preset].append(data["rating"])
                except Exception:
                    pass
    return ratings

if __name__ == "__main__":
    r = collect_ratings()
    for k, v in r.items():
        if v:
            print(f"{k}: avg {sum(v)/len(v):.2f} ({len(v)} ratings)")
