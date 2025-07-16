import requests
import pandas as pd
import time
import os

os.makedirs("evaluation", exist_ok=True)

test_cases = [
    {"query": "Wrong product received Gaming Mouse", "expected_decision": "Refund Approved", "expected_confidence": 0.9},
    {"query": "Broken Smartwatch Strap", "expected_decision": "Fallback due to policy restriction", "expected_confidence": 0.3},
    {"query": "Refund for damaged Battery", "expected_decision": "Fallback due to policy restriction", "expected_confidence": 0.3},
    {"query": "Refund for Digital Goods purchased last week", "expected_decision": "Fallback due to policy restriction", "expected_confidence": 0.3},
]

results = []

for case in test_cases:
    query = case["query"]
    expected_decision = case["expected_decision"]
    expected_confidence = case["expected_confidence"]

    print(f"\n🔵 Testing: {query}")
    try:
        r = requests.post("http://127.0.0.1:8000/init-agent-run/", json={"query": query})
        r.raise_for_status()
        run_id = r.json()["run_id"]
        print(f"▶️ Run ID: {run_id}")

        print("🔄 Polling for completion...")
        for attempt in range(60):
            status = requests.get(f"http://127.0.0.1:8000/get-next-step/{run_id}")
            status.raise_for_status()
            status_data = status.json()
            
            if status_data.get("done"):
                print(f"✅ Completed after {attempt + 1} attempts")
                break
            time.sleep(1)
        else:
            print("⏰ Timeout after 60 seconds")

        final = requests.get(f"http://127.0.0.1:8000/get-full-run/{run_id}").json()
        actual_decision = final.get("final_decision")
        actual_confidence = final.get("final_confidence")

        pass_decision = actual_decision == expected_decision
        pass_confidence = actual_confidence is not None and abs(actual_confidence - expected_confidence) <= 0.1

        print(f"✅ Expected: {expected_decision} | Got: {actual_decision}")
        print(f"✅ Expected Confidence: {expected_confidence} | Got: {actual_confidence}")
        print(f"✅ Pass: {pass_decision and pass_confidence}")

        results.append({
            "query": query,
            "expected_decision": expected_decision,
            "actual_decision": actual_decision,
            "expected_confidence": expected_confidence,
            "actual_confidence": actual_confidence,
            "decision_match": pass_decision,
            "confidence_match": pass_confidence,
            "overall_pass": pass_decision and pass_confidence
        })

    except Exception as e:
        print(f"❌ Error: {str(e)}")
        results.append({
            "query": query,
            "expected_decision": expected_decision,
            "actual_decision": "error",
            "expected_confidence": expected_confidence,
            "actual_confidence": None,
            "decision_match": False,
            "confidence_match": False,
            "overall_pass": False
        })

df = pd.DataFrame(results)
print("\n📊 EVALUATION RESULTS SUMMARY")
print(df.to_string(index=False))

df.to_csv("evaluation/evaluation_results.csv", index=False)

pass_rate = (df["overall_pass"].sum() / len(df)) * 100

with open("evaluation/evaluation_summary.txt", "w") as f:
    f.write(f"Pass Rate: {pass_rate:.1f}%\n")
    for idx, row in df.iterrows():
        f.write(f"{idx + 1}. {row['query']} => {'PASS' if row['overall_pass'] else 'FAIL'}\n")

print("\n✅ Results saved in evaluation/ folder ✅")
