import streamlit as st
import requests
import time

st.set_page_config(page_title="🧠 AgentScope — Live Reasoning with Polling")
st.title("🧠 AgentScope: Multi-Step AI Agent Debugger")

st.write("### Enter your query:")
user_query = st.text_input("User Query", "Wrong product received Gaming Mouse")

if st.button("Run Agent"):
    with st.spinner("Initializing agent..."):
        r = requests.post("http://127.0.0.1:8000/init-agent-run/", json={"query": user_query})
        run_id = r.json().get("run_id")
        st.session_state.run_id = run_id
        st.success(f"✅ Agent Run Started with ID: {run_id}")

if "run_id" in st.session_state:
    st.subheader(f"Replay of Run ID: {st.session_state.run_id}")

    steps_placeholder = st.empty()
    final_placeholder = st.empty()
    confidence_placeholder = st.empty()

    fetched_steps = []

    while True:
        res = requests.get(f"http://127.0.0.1:8000/get-next-step/{st.session_state.run_id}").json()
        steps = res.get("steps", [])
        done = res.get("done", False)
        final = res.get("final")
        confidence = res.get("confidence")

        new_steps = steps[len(fetched_steps):]
        if new_steps:
            with steps_placeholder.container():
                for idx, step in enumerate(steps):
                    st.markdown(f"**Step {idx+1}: {step['step']} — {step['desc']}**")
                    st.code(step['obs'])
            fetched_steps = steps

        if done:
            final_placeholder.success(f"✅ Final Decision: {final}")
            confidence_placeholder.info(f"Confidence Score: {confidence}")
            break

        time.sleep(1)

st.write("---")
st.caption("Polling-based live debugging — zero buffering, real-time timeline, early fallback visibility.")


