import threading
import pandas as pd
from together import Together
import os
from dotenv import load_dotenv

# ✅ Load API Key
load_dotenv()
from together import Together
import os

client = Together()
client.api_key = os.getenv("TOGETHER_API_KEY")
agent_runs = {}

# ✅ Load Data
transactions = pd.read_csv("data/transactions.csv")
policies = pd.read_csv("data/policy.csv")

def classify_intent(query):
    prompt = f"Classify the user intent (refund, fraud, unknown) for: '{query}'. Return only the intent."
    response = client.chat.completions.create(
        model="mistralai/Mixtral-8x7B-Instruct-v0.1",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content.strip().lower()

def lookup_transaction(query):
    # Simple fuzzy match
    matches = transactions[transactions["user_query"].str.contains(query.split()[0], case=False, na=False)]
    return matches.iloc[0].to_dict() if not matches.empty else None

def lookup_policy(item):
    match = policies[policies["item"].str.lower() == item.lower()]
    return match.iloc[0].to_dict() if not match.empty else None

def calculate_confidence(policy):
    if not policy or policy.get("refund_allowed") != "yes":
        return 0.3
    return 0.9

def run_refund_fraud_agent_polling(run_id, query):
    def agent_logic():
        run = agent_runs[run_id]
        steps = run["steps"]

        def log(step_type, desc, obs):
            steps.append({"step": step_type, "desc": desc, "obs": obs})

        # Step 1: Intent Classification
        log("Thought", "Classifying user intent", query)
        intent = classify_intent(query)
        log("Observation", "Intent classified", {"intent": intent})

        # Step 2: Transaction Lookup
        txn = lookup_transaction(query)
        if txn:
            item = txn["item"]
            log("Observation", "Transaction lookup result", txn)
        else:
            log("Observation", "No matching transaction found", {})
            item = "unknown"

        # Step 3: Policy Lookup
        policy = lookup_policy(item)
        log("Thought", "Fetched refund policy", policy.get("policy_text", "No policy found") if policy else "No policy")

        # Step 4: Confidence
        confidence = calculate_confidence(policy)
        log("Observation", "Policy-based confidence scoring", {"confidence": confidence})

        # Step 5: Decision + Fallback
        if txn is None or not policy:
            log("Fallback", "Unknown user/query fallback triggered — no transaction found", None)
            final = "Fallback due to no transaction match"
        elif policy["refund_allowed"] != "yes":
            log("Fallback", "Fallback due to policy restriction", None)
            final = "Fallback due to policy restriction"
        else:
            log("Decision", "Refund Approved", {"status": "approved"})
            final = "Refund Approved"

        run["final"] = final
        run["confidence"] = confidence
        run["done"] = True

    threading.Thread(target=agent_logic).start()
