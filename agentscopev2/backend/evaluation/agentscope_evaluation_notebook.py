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
        # Start agent run
        r = requests.post("http://127.0.0.1:8000/init-agent-run/", json={"query": query})
        r.raise_for_status()
        run_id = r.json()["run_id"]
        print(f"▶️ Run ID: {run_id}")

        # Poll every 1s until done (removed SSE entirely)
        print("🔄 Polling for completion...")
        for attempt in range(60):
            try:
                status = requests.get(f"http://127.0.0.1:8000/get-next-step/{run_id}")
                status.raise_for_status()
                status_data = status.json()
                
                if status_data.get("done"):
                    print(f"✅ Completed after {attempt + 1} attempts")
                    break
                    
                # Show progress
                if attempt % 10 == 0:  # Every 10 seconds
                    print(f"  Still running... ({attempt + 1}/60)")
                    
                time.sleep(1)
            except requests.exceptions.RequestException as e:
                print(f"  Polling error on attempt {attempt + 1}: {e}")
                time.sleep(1)
        else:
            print("⏰ Timeout after 60 seconds")

        # Get full reasoning trace
        final_response = requests.get(f"http://127.0.0.1:8000/get-full-run/{run_id}")
        final_response.raise_for_status()
        final = final_response.json()
        
        actual_decision = final.get("final_decision")
        actual_confidence = final.get("final_confidence")

        pass_decision = actual_decision == expected_decision
        pass_conf = actual_confidence is not None and abs(actual_confidence - expected_confidence) <= 0.1

        print(f"✅ Expected: {expected_decision} | Got: {actual_decision}")
        print(f"✅ Expected Confidence: {expected_confidence} | Got: {actual_confidence}")
        print(f"✅ Pass: {pass_decision and pass_conf}")

        results.append({
            "query": query,
            "expected_decision": expected_decision,
            "actual_decision": actual_decision,
            "expected_confidence": expected_confidence,
            "actual_confidence": actual_confidence,
            "decision_match": pass_decision,
            "confidence_match": pass_conf,
            "overall_pass": pass_decision and pass_conf
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

# Create results summary
df = pd.DataFrame(results)
print("\n" + "="*60)
print("📊 EVALUATION RESULTS SUMMARY")
print("="*60)
print(df.to_string(index=False))

# Calculate pass rate
total_tests = len(results)
passed_tests = sum(1 for r in results if r["overall_pass"])
pass_rate = (passed_tests / total_tests) * 100

print(f"\n✅ Pass Rate: {passed_tests}/{total_tests} ({pass_rate:.1f}%)")

# Save to CSV
df.to_csv("evaluation/evaluation_results.csv", index=False)
print(f"\n💾 Saved detailed results to evaluation/evaluation_results.csv")

# Create a summary file
summary = {
    "total_tests": total_tests,
    "passed_tests": passed_tests,
    "pass_rate": pass_rate,
    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
}

with open("evaluation/evaluation_summary.txt", "w") as f:
    f.write(f"Evaluation Summary - {summary['timestamp']}\n")
    f.write("="*50 + "\n")
    f.write(f"Total Tests: {total_tests}\n")
    f.write(f"Passed Tests: {passed_tests}\n")
    f.write(f"Pass Rate: {pass_rate:.1f}%\n\n")
    f.write("Individual Results:\n")
    for i, result in enumerate(results, 1):
        f.write(f"{i}. {result['query']}: {'PASS' if result['overall_pass'] else 'FAIL'}\n")

print("📄 Saved summary to evaluation/evaluation_summary.txt")
print("\n✅ Evaluation completed successfully! ✅")