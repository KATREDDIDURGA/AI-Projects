import os
import json

runs_file = "data/runs.json"

# Initialize file if not exists
if not os.path.exists(runs_file):
    with open(runs_file, "w") as f:
        json.dump([], f)

def save_run(run_id, query, steps, final, confidence):
    with open(runs_file, "r") as f:
        data = json.load(f)
    data.append({
        "run_id": run_id,
        "query": query,
        "steps": steps,
        "final": final,
        "confidence": confidence
    })
    with open(runs_file, "w") as f:
        json.dump(data, f, indent=2)

def load_runs():
    with open(runs_file, "r") as f:
        return json.load(f)
