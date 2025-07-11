# feedback/collect.py
import json
from pathlib import Path

LOG_PATH = Path("data/feedback.json")

def log_result(entry: dict):
    if LOG_PATH.exists():
        with open(LOG_PATH, "r") as f:
            data = json.load(f)
    else:
        data = []

    data.append(entry)

    with open(LOG_PATH, "w") as f:
        json.dump(data, f, indent=2)
