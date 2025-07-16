import requests, time, pandas as pd

queries = [
    {"query": "Wrong product received Gaming Mouse", "expected": "Refund Approved", "confidence": 0.9},
    {"query": "Broken Smartwatch Strap", "expected": "Fallback: policy restriction", "confidence": 0.3},
    {"query": "Refund for damaged Battery", "expected": "Fallback: no transaction match", "confidence": 0.3},
    {"query": "Refund for Digital Goods", "expected": "Fallback: no transaction match", "confidence": 0.3},
]

results = []

for q in queries:
    print(f"🔵 Testing query: {q['query']}")
    r = requests.post("http://127.0.0.1:8000/init-agent-run/", json={"query": q["query"]})
    run_id = r.json()["run_id"]
    for _ in range(20):
        time.sleep(1)
        run = requests.get(f"http://127.0.0.1:8000/get-run/{run_id}").json()
        if run.get("done"):
            break

    final = run.get("final")
    conf = run.get("confidence")
    print(f"✅ Expected: {q['expected']} | Got: {final}")
    print(f"✅ Expected Confidence: {q['confidence']} | Got: {conf}\n")

    pass_logic = final == q["expected"]
    pass_conf = abs(conf - q["confidence"]) < 0.1 if conf is not None else False
    results.append({"Query": q["query"], "Expected": q["expected"], "Got": final, "Expected_Conf": q["confidence"], "Got_Conf": conf, "Pass": pass_logic and pass_conf})

df = pd.DataFrame(results)
print("\n✅ Evaluation Summary:")
print(df)

df.to_csv("data/evaluation_results.csv", index=False)
print("✅ Saved evaluation_results.csv ✅")
