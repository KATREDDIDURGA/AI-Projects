import streamlit as st
import requests
import time

BACKEND = "http://localhost:8000"

st.set_page_config(page_title="AgentScope Polling", layout="centered")
st.title("🧠 AgentScope — Live Reasoning with Polling")

query = st.text_input("Enter your query:", "I want a refund for last week’s order")

if st.button("Run Agent"):
    with st.spinner("Running agent..."):
        run_resp = requests.post(f"{BACKEND}/init-agent-run/", json={"query": query})
        run_id = run_resp.json()["run_id"]

        timeline_placeholder = st.empty()
        final_placeholder = st.empty()
        seen_steps = 0

        while True:
            time.sleep(1)
            poll_resp = requests.get(f"{BACKEND}/get-next-step/{run_id}")
            data = poll_resp.json()
            steps = data.get("steps", [])
            done = data.get("done", False)
            final = data.get("final", None)

            if steps:
                with timeline_placeholder.container():
                    for idx, step in enumerate(steps):
                        st.markdown(f"**Step {idx+1}: {step['step']} — {step['desc']}**")
                        obs = step.get("obs")
                        st.write(obs)

            if done:
                final_placeholder.success(f"✅ Final Decision: {final}")
                break

st.info("Polling-based live debugging — **zero buffering**, **real-time timeline**, **early fallback visibility**.")
