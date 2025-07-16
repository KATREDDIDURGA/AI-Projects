import json, os

runs_file = "data/runs.json"
current_steps = []

def init_run(run_id, query):
    global current_steps
    current_steps = []
    
    save_run(run_id, query, [])

def log_step(step, desc, obs):
    global current_steps
    current_steps.append({"step": step, "desc": desc, "obs": obs})

def finalize_run(run_id, query):
    with open(runs_file, "r") as f:
        runs = json.load(f)
    for run in runs:
        if run["run_id"] == run_id:
            run["steps"] = current_steps
    with open(runs_file, "w") as f:
        json.dump(runs, f, indent=2)

def save_run(run_id, query, steps):
    if not os.path.exists(runs_file):
        with open(runs_file, "w") as f: json.dump([], f)
    with open(runs_file, "r") as f: runs = json.load(f)
    runs.append({"run_id": run_id, "query": query, "steps": steps})
    with open(runs_file, "w") as f: json.dump(runs, f, indent=2)