# scripts/run_pipeline.py
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
from llm.generate import generate_response
from eval.scorer import evaluate_response
from feedback.collect import log_result

def main():
    df = pd.read_csv("data/prompts.csv")
    for _, row in df.iterrows():
        print(f"⏳ Prompt: {row['prompt']}")
        response = generate_response(row['prompt'])
        eval_result = evaluate_response(response)
        log_entry = {
            "id": int(row["id"]),
            "prompt": row["prompt"],
            "response": response,
            "evaluation": eval_result
        }
        log_result(log_entry)
        print(f"✅ Scored: {eval_result}\n")

if __name__ == "__main__":
    main()
