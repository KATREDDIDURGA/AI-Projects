import threading
import pandas as pd
from difflib import get_close_matches

agent_runs = {}
transactions = pd.read_csv("data/transactions.csv")
policies = pd.read_csv("data/policy.csv")

def classify_intent(query):
    query = query.lower()
    if any(word in query for word in ["refund", "return", "wrong", "damaged", "broken"]):
        return "refund"
    return "unknown"

def find_transaction(query):
    matches = get_close_matches(query.lower(), transactions["user_query"].str.lower(), n=1, cutoff=0.6)
    if matches:
        txn = transactions[transactions["user_query"].str.lower() == matches[0]]
        if not txn.empty:
            return txn.iloc[0].to_dict()
    return None

def lookup_policy(item):
    p = policies[policies["item"].str.lower() == item.lower()]
    if not p.empty:
        return p.iloc[0].to_dict()
    return {"policy_text": "No policy found", "refund_allowed": "no", "refund_window_days": 0}

def run_refund_fraud_agent_polling(run_id, query):
    def agent_logic():
        run = agent_runs[run_id]
        steps = run["steps"]

        def log(step_type, desc, obs):
            steps.append({"step": step_type, "desc": desc, "obs": obs})

        log("Thought", "Classifying user intent", query)
        intent = classify_intent(query)
        log("Observation", "Intent classified", {"intent": intent})

        txn = find_transaction(query)
        if txn:
            item = txn["item"]
            log("Observation", "Transaction found", txn)
        else:
            item = "unknown"
            log("Observation", "No transaction found", {})

        policy = lookup_policy(item)
        log("Thought", "Policy lookup", policy["policy_text"])
        refund_allowed = policy.get("refund_allowed") == "yes"
        confidence = 0.9 if refund_allowed and txn else 0.3
        log("Observation", "Confidence calculated", {"confidence": confidence})

        # ✅ Fallback based on policy
        if txn is None:
            log("Fallback", "Fallback: no matching transaction", None)
            run["final"] = "Fallback: no transaction match"
        elif refund_allowed == False:
            log("Fallback", "Fallback: policy disallows refund", None)
            run["final"] = "Fallback: policy restriction"
        else:
            log("Decision", "Refund approved", {"status": "approved"})
            run["final"] = "Refund Approved"

        run["confidence"] = confidence
        run["done"] = True

    threading.Thread(target=agent_logic).start()
