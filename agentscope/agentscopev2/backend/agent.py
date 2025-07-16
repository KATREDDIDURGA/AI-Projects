import threading
import pandas as pd
from together import Together
import os
from dotenv import load_dotenv
import re
from datetime import datetime, timedelta

# ✅ Load API Key
load_dotenv()

# ✅ Initialize Together client with API key
client = Together(api_key=os.getenv("TOGETHER_API_KEY"))

# Global storage for agent runs
agent_runs = {}

# ✅ Load Data
transactions = pd.read_csv("data/transactions.csv")
policies = pd.read_csv("data/policy.csv")

def extract_item_from_query(query):
    """Extract the actual item mentioned in the user query using AI"""
    prompt = f"""Extract the specific product/item mentioned in this query. Return only the item name (like 'mouse', 'webcam', 'laptop', etc.):
    
    Query: "{query}"
    
    Item:"""
    
    try:
        response = client.chat.completions.create(
            model="mistralai/Mixtral-8x7B-Instruct-v0.1",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=10,
            temperature=0.1
        )
        return response.choices[0].message.content.strip().lower()
    except Exception as e:
        print(f"Error extracting item: {e}")
        # Fallback: simple keyword extraction
        items = ['mouse', 'webcam', 'laptop', 'keyboard', 'monitor', 'headset']
        query_lower = query.lower()
        for item in items:
            if item in query_lower:
                return item
        return "unknown"

def extract_timeframe_from_query(query):
    """Extract timeframe information from query (e.g., '40 days', 'last week')"""
    # Look for patterns like "40 days", "2 weeks", "last month"
    day_patterns = re.findall(r'(\d+)\s*days?', query.lower())
    week_patterns = re.findall(r'(\d+)\s*weeks?', query.lower())
    
    if day_patterns:
        return int(day_patterns[0])
    elif week_patterns:
        return int(week_patterns[0]) * 7
    elif 'last week' in query.lower():
        return 7
    elif 'last month' in query.lower():
        return 30
    
    return None

def classify_intent(query):
    """Classify user intent using Together API"""
    prompt = f"Classify the user intent (refund, fraud, unknown) for: '{query}'. Return only the intent."
    
    try:
        response = client.chat.completions.create(
            model="mistralai/Mixtral-8x7B-Instruct-v0.1",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=10,
            temperature=0.1
        )
        return response.choices[0].message.content.strip().lower()
    except Exception as e:
        print(f"Error in classify_intent: {e}")
        return "unknown"

def lookup_transaction_by_item(item):
    """Look up transaction by the actual item mentioned"""
    try:
        # First try exact match
        matches = transactions[transactions["item"].str.lower() == item.lower()]
        if not matches.empty:
            return matches.iloc[0].to_dict()
        
        # Then try partial match
        matches = transactions[transactions["item"].str.contains(item, case=False, na=False)]
        if not matches.empty:
            return matches.iloc[0].to_dict()
        
        return None
    except Exception as e:
        print(f"Error in lookup_transaction_by_item: {e}")
        return None

def lookup_policy(item):
    """Look up policy for a specific item"""
    try:
        # First try exact match
        match = policies[policies["item"].str.lower() == item.lower()]
        if not match.empty:
            return match.iloc[0].to_dict()
        
        # Then try partial match
        match = policies[policies["item"].str.contains(item, case=False, na=False)]
        if not match.empty:
            return match.iloc[0].to_dict()
        
        return None
    except Exception as e:
        print(f"Error in lookup_policy: {e}")
        return None

def extract_policy_days(policy_text):
    """Extract number of days from policy text"""
    if not policy_text:
        return 0
    
    # Look for patterns like "30 days", "within 30 days"
    day_patterns = re.findall(r'(\d+)\s*days?', policy_text.lower())
    if day_patterns:
        return int(day_patterns[0])
    
    return 0

def calculate_confidence(policy, user_timeframe, policy_days):
    """Calculate confidence score based on policy and timeframe"""
    if not policy:
        return 0.1
    
    if policy.get("refund_allowed") != "yes":
        return 0.2
    
    # Check timeframe compliance
    if user_timeframe and policy_days:
        if user_timeframe <= policy_days:
            return 0.9
        else:
            return 0.3  # Outside policy timeframe
    
    return 0.7  # Policy allows but no timeframe info

def run_refund_fraud_agent_polling(run_id, query):
    """Main agent logic with proper error handling"""
    def agent_logic():
        try:
            run = agent_runs[run_id]
            steps = run["steps"]

            def log(step_type, desc, obs):
                steps.append({"step": step_type, "desc": desc, "obs": obs})

            # Step 1: Intent Classification
            log("Thought", "Classifying user intent", query)
            intent = classify_intent(query)
            log("Observation", "Intent classified", {"intent": intent})

            # Step 2: Extract Item from Query
            log("Thought", "Extracting item mentioned in query", query)
            mentioned_item = extract_item_from_query(query)
            log("Observation", "Item extracted from query", {"mentioned_item": mentioned_item})

            # Step 3: Extract Timeframe
            user_timeframe = extract_timeframe_from_query(query)
            if user_timeframe:
                log("Observation", "Timeframe extracted from query", {"timeframe_days": user_timeframe})

            # Step 4: Transaction Lookup (by mentioned item)
            txn = lookup_transaction_by_item(mentioned_item)
            if txn:
                log("Observation", "Transaction found for mentioned item", txn)
            else:
                log("Observation", "No transaction found for mentioned item", {"searched_item": mentioned_item})

            # Step 5: Policy Lookup (for mentioned item)
            policy = lookup_policy(mentioned_item)
            if policy:
                policy_days = extract_policy_days(policy.get("policy_text", ""))
                log("Observation", "Policy found for mentioned item", {
                    "policy": policy.get("policy_text", ""),
                    "refund_allowed": policy.get("refund_allowed", ""),
                    "policy_days": policy_days
                })
            else:
                policy_days = 0
                log("Observation", "No policy found for mentioned item", {"searched_item": mentioned_item})

            # Step 6: Timeframe Validation
            if user_timeframe and policy_days:
                if user_timeframe > policy_days:
                    log("Thought", "Timeframe validation failed", {
                        "user_timeframe": user_timeframe,
                        "policy_limit": policy_days,
                        "status": "EXCEEDS_POLICY_LIMIT"
                    })
                else:
                    log("Observation", "Timeframe validation passed", {
                        "user_timeframe": user_timeframe,
                        "policy_limit": policy_days,
                        "status": "WITHIN_POLICY_LIMIT"
                    })

            # Step 7: Confidence Calculation
            confidence = calculate_confidence(policy, user_timeframe, policy_days)
            log("Observation", "Confidence calculated", {"confidence": confidence})

            # Step 8: Final Decision
            if not txn:
                log("Fallback", "No transaction found for the mentioned item", {"item": mentioned_item})
                final = "Fallback: No transaction found for the mentioned item"
            elif not policy:
                log("Fallback", "No policy found for the mentioned item", {"item": mentioned_item})
                final = "Fallback: No policy found for the mentioned item"
            elif policy.get("refund_allowed") != "yes":
                log("Fallback", "Policy does not allow refunds", {"policy": policy.get("policy_text", "")})
                final = "Refund Denied: Policy does not allow refunds"
            elif user_timeframe and policy_days and user_timeframe > policy_days:
                log("Fallback", "Request exceeds policy timeframe", {
                    "user_timeframe": user_timeframe,
                    "policy_limit": policy_days
                })
                final = f"Refund Denied: Request exceeds policy timeframe ({user_timeframe} days > {policy_days} days)"
            else:
                log("Decision", "Refund Approved", {"status": "approved"})
                final = "Refund Approved"

            run["final"] = final
            run["confidence"] = confidence
            run["done"] = True

        except Exception as e:
            print(f"Error in agent_logic: {e}")
            run["final"] = "Error occurred during processing"
            run["confidence"] = 0.0
            run["done"] = True

    threading.Thread(target=agent_logic).start()

# ✅ Helper function to initialize and start agent
def start_agent(query):
    """Initialize and start the refund agent"""
    import uuid
    run_id = str(uuid.uuid4())
    
    agent_runs[run_id] = {
        "steps": [],
        "final": None,
        "confidence": 0.0,
        "done": False
    }
    
    run_refund_fraud_agent_polling(run_id, query)
    return run_id

# ✅ Helper function to get agent status
def get_agent_status(run_id):
    """Get the current status of an agent run"""
    return agent_runs.get(run_id, {"error": "Run ID not found"})

# ✅ Example usage
if __name__ == "__main__":
    # Test the agent
    test_query = "I used my mouse for just 40 days and it is not working now. Want to replace or refund"
    
    print("Starting refund agent...")
    run_id = start_agent(test_query)
    
    import time
    # Poll for results
    while not agent_runs[run_id]["done"]:
        time.sleep(1)
    
    # Print results
    result = get_agent_status(run_id)
    print(f"\nFinal Decision: {result['final']}")
    print(f"Confidence: {result['confidence']}")
    print("\nSteps taken:")
    for step in result["steps"]:
        print(f"  {step['step']}: {step['desc']} -> {step['obs']}")