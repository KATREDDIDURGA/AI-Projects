# app/frontend/streamlit_app.py
import streamlit as st
import requests
import json

st.set_page_config(page_title="AgentScope — Viewer", layout="centered")

st.markdown("## 🔍 AgentScope — Decision Trace Viewer")

run_id = st.text_input("Search Agent Execution", placeholder="Enter Agent Run ID")

if st.button("View Trace") and run_id:
    with st.spinner("Fetching trace..."):
        try:
            response = requests.get(f"http://localhost:8000/api/trace/{run_id}")
            if response.status_code == 200:
                data = response.json()
                
                # Display run information
                run_info = data.get('run', {})
                st.success(f"✅ Found Agent Run")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"**Agent Type**: {run_info.get('agent_type', 'N/A')}")
                    st.write(f"**Status**: {run_info.get('status', 'N/A')}")
                    st.write(f"**Started**: {run_info.get('started_at', 'N/A')}")
                
                with col2:
                    st.write(f"**Run ID**: {run_info.get('run_id', 'N/A')}")
                    st.write(f"**Completed**: {run_info.get('completed_at', 'N/A')}")
                
                st.write(f"**Query**: {run_info.get('query', 'N/A')}")
                st.write(f"**Final Decision**: {run_info.get('final_decision', 'N/A')}")
                
                # Display steps
                steps = data.get('steps', [])
                if steps:
                    st.write("### 🧩 Reasoning Steps")
                    
                    for step in steps:
                        step_type = step.get('step_type', 'unknown')
                        description = step.get('description', 'No description')
                        observation = step.get('observation', '')
                        confidence = step.get('confidence', None)
                        timestamp = step.get('timestamp', '')
                        
                        # Create a nice card for each step
                        with st.expander(f"Step {step.get('step_number', '?')}: {step_type.title()}"):
                            st.write(f"**Description**: {description}")
                            if observation:
                                st.write(f"**Observation**: {observation}")
                            if confidence is not None:
                                st.write(f"**Confidence**: {confidence:.2f}")
                            if timestamp:
                                st.write(f"**Timestamp**: {timestamp}")
                else:
                    st.warning("No steps found for this run.")
                
                # Display metrics
                metrics = data.get('metrics', [])
                if metrics:
                    st.write("### 📈 Metrics")
                    for metric in metrics:
                        st.write(f"**{metric.get('metric_name', 'Unknown')}**: {metric.get('value', 'N/A')}")
                        
            elif response.status_code == 404:
                st.error(f"❌ Run ID '{run_id}' not found")
            else:
                st.error(f"❌ Backend error (Status: {response.status_code})")
                st.write(f"Response: {response.text}")
                
        except requests.ConnectionError:
            st.error("❌ Cannot connect to backend. Make sure the FastAPI server is running on http://localhost:8000")
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")

# Add some helpful info
st.markdown("---")
st.markdown("**Usage:**")
st.markdown("1. Start the FastAPI server: `uvicorn app.main:app --reload`")
st.markdown("2. Insert test data: `python scripts/insert_test_data.py`")
st.markdown("3. Use this test run ID: `13025c79-065a-42d5-b9d4-915b30442027`")