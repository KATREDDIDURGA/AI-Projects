# ui/app.py

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
import pandas as pd
import json
from pathlib import Path
from llm.generate import generate_response
from eval.scorer import evaluate_response
from feedback.collect import log_result

st.set_page_config(page_title="LLM EvalFlow", layout="centered")

st.title("🧠 LLM EvalFlow Dashboard")
st.write("View prompt results, LLM outputs, and evaluation scores.")

# Load feedback
feedback_file = Path("../data/feedback.json")
if feedback_file.exists():
    feedback = json.loads(feedback_file.read_text())
    df = pd.DataFrame(feedback)
else:
    st.warning("No feedback.json found.")
    df = pd.DataFrame()

if not df.empty:
    st.subheader("📊 Evaluation Results")
    st.dataframe(df[["id", "prompt", "response", "evaluation"]], use_container_width=True)
else:
    st.info("Run the pipeline first to generate results.")

st.markdown("---")

st.subheader("➕ Test a New Prompt")

user_prompt = st.text_area("Enter a prompt:", height=100)
if st.button("Run LLM + Evaluate"):
    if user_prompt.strip() != "":
        with st.spinner("Getting LLM response..."):
            response = generate_response(user_prompt)
            result = evaluate_response(response)

            # Display
            st.markdown("#### 📝 Response")
            st.write(response)

            st.markdown("#### 🧪 Evaluation")
            st.json(result)

            log_result({
                "id": len(df) + 1,
                "prompt": user_prompt,
                "response": response,
                "evaluation": result
            })
            st.success("Logged to feedback.json!")
    else:
        st.error("Prompt cannot be empty.")
