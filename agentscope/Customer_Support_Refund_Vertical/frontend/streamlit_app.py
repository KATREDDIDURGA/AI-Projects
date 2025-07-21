"""
AgentScope - Working Frontend
"""

import streamlit as st
import requests
import json
import time
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="🧠 AgentScope - AI Agent Debugger",
    page_icon="🧠",
    layout="wide"
)

# Constants
API_BASE_URL = "http://127.0.0.1:8000"

def test_backend():
    """Test if backend is running"""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=2)
        return response.status_code == 200
    except:
        return False

def start_agent(query):
    """Start agent execution"""
    try:
        response = requests.post(
            f"{API_BASE_URL}/init-agent-run/",
            json={"query": query},
            timeout=10
        )
        if response.status_code == 200:
            return response.json().get("run_id")
        else:
            st.error(f"Failed to start agent: {response.text}")
            return None
    except Exception as e:
        st.error(f"Error starting agent: {e}")
        return None

def get_agent_results(run_id):
    """Get agent execution results"""
    try:
        response = requests.get(f"{API_BASE_URL}/get-next-step/{run_id}", timeout=5)
        if response.status_code == 200:
            return response.json()
        else:
            return None
    except Exception as e:
        st.error(f"Error getting results: {e}")
        return None

def log_human_decision(run_id, query, agent_decision, human_decision, reasoning):
    """Log human override decision"""
    try:
        log_data = {
            "timestamp": datetime.now().isoformat(),
            "run_id": run_id,
            "query": query,
            "agent_decision": agent_decision,
            "human_decision": human_decision,
            "reasoning": reasoning
        }
        
        if 'human_decisions' not in st.session_state:
            st.session_state.human_decisions = []
        
        st.session_state.human_decisions.append(log_data)
        return True
    except Exception as e:
        st.error(f"Failed to log decision: {e}")
        return False

def extract_evidence(steps):
    """Extract evidence from agent steps"""
    evidence = {}
    
    for step in steps:
        desc = step.get("desc", "").lower()
        obs = step.get("obs")
        
        if "intent" in desc and obs:
            if isinstance(obs, str):
                try:
                    obs = json.loads(obs)
                except:
                    pass
            if isinstance(obs, dict) and "intent" in obs:
                evidence["intent"] = obs["intent"]
        
        elif "product" in desc and obs:
            if isinstance(obs, str):
                try:
                    obs = json.loads(obs)
                except:
                    pass
            if isinstance(obs, dict) and "item" in obs:
                evidence["product"] = obs["item"]
        
        elif "policy" in desc and obs:
            if isinstance(obs, dict):
                evidence["policy"] = obs
        
        elif "timeframe" in desc and obs:
            if isinstance(obs, dict):
                evidence["timeframe"] = obs
    
    return evidence

def render_human_review(run_id, query, decision, steps):
    """Render human review interface"""
    
    st.markdown("---")
    st.markdown("### 🚨 HUMAN REVIEW REQUIRED")
    st.error(decision)
    
    # Extract evidence
    evidence = extract_evidence(steps)
    
    # Show evidence
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Customer Request:**")
        st.write(f"• Query: {query}")
        st.write(f"• Intent: {evidence.get('intent', 'Unknown')}")
        st.write(f"• Product: {evidence.get('product', 'Unknown')}")
    
    with col2:
        st.markdown("**Policy Analysis:**")
        if evidence.get('policy'):
            policy = evidence['policy']
            st.write(f"• Policy: {policy.get('policy_text', 'N/A')}")
            st.write(f"• Policy Days: {policy.get('policy_days', 'N/A')}")
        
        if evidence.get('timeframe'):
            timeframe = evidence['timeframe']
            st.write(f"• Timeframe Status: {timeframe.get('status', 'N/A')}")
    
    # Human decision buttons
    st.markdown("**👤 Your Decision:**")
    
    col_a, col_b, col_c = st.columns(3)
    
    with col_a:
        if st.button("✅ APPROVE", key=f"approve_{run_id}"):
            reasoning = st.text_input("Approval reason:", key=f"approve_reason_{run_id}")
            if reasoning:
                if log_human_decision(run_id, query, decision, "APPROVED", reasoning):
                    st.success("✅ **APPROVED BY HUMAN**")
                    st.info(f"Reason: {reasoning}")
                    st.balloons()
    
    with col_b:
        if st.button("❌ DENY", key=f"deny_{run_id}"):
            reasoning = st.text_input("Denial reason:", key=f"deny_reason_{run_id}")
            if reasoning:
                if log_human_decision(run_id, query, decision, "DENIED", reasoning):
                    st.error("❌ **DENIED BY HUMAN**")
                    st.info(f"Reason: {reasoning}")
    
    with col_c:
        if st.button("📋 MORE INFO", key=f"info_{run_id}"):
            info_needed = st.text_input("Information needed:", key=f"info_needed_{run_id}")
            if info_needed:
                if log_human_decision(run_id, query, decision, "INFO_REQUESTED", info_needed):
                    st.warning("📋 **MORE INFO REQUESTED**")
                    st.info(f"Info needed: {info_needed}")
    
    # Explanation
    st.markdown("**🔍 Why Human Review:**")
    st.markdown("""
    - Customer said "after 30 days" but policy limit is 30 days
    - This creates a contradiction that requires human judgment
    - Agent blocked automatic processing to prevent policy violations
    """)

def main():
    """Main application"""
    
    # Header
    st.title("🧠 AgentScope: AI Agent Debugger")
    st.markdown("### Real-time AI Agent Debugging with Human Review")
    
    # Backend check
    if not test_backend():
        st.error("❌ Backend not running! Start with: `uvicorn app.main:app --host 0.0.0.0 --port 8000`")
        return
    else:
        st.success("✅ Backend connected")
    
    # Sidebar
    with st.sidebar:
        st.header("🎛️ Control Panel")
        
        try:
            health = requests.get(f"{API_BASE_URL}/health").json()
            st.metric("System Status", "Healthy")
        except:
            st.error("Backend Error")
        
        # Human decisions stats
        if 'human_decisions' in st.session_state:
            decisions = st.session_state.human_decisions
            st.metric("Human Overrides", len(decisions))
            
            if decisions:
                st.subheader("Recent Decisions")
                for decision in decisions[-3:]:
                    st.write(f"**{decision['human_decision']}**")
                    st.caption(decision['timestamp'][:16])
    
    # Main interface
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("🚀 Start Agent")
        
        query = st.text_area(
            "Enter your query:",
            value="I want a refund for my gaming mouse that broke after 30 days",
            height=100
        )
        
        if st.button("🚀 Start Agent", type="primary"):
            if query.strip():
                with st.spinner("Running agent..."):
                    run_id = start_agent(query.strip())
                    
                if run_id:
                    st.success(f"✅ Agent started: {run_id}")
                    
                    with st.spinner("Getting results..."):
                        time.sleep(1)
                        results = get_agent_results(run_id)
                    
                    if results:
                        st.session_state.last_results = results
                        st.session_state.last_run_id = run_id
                        st.session_state.last_query = query.strip()
                    else:
                        st.error("❌ Could not get results")
            else:
                st.error("Please enter a query")
    
    with col2:
        st.subheader("📊 Quick Actions")
        
        if st.button("🔄 Refresh"):
            if hasattr(st.session_state, 'last_run_id'):
                results = get_agent_results(st.session_state.last_run_id)
                if results:
                    st.session_state.last_results = results
        
        st.subheader("💡 Test Cases")
        examples = [
            "I want a refund for my gaming mouse that broke after 30 days",
            "I want a refund for my gaming mouse that broke after 20 days",
            "My laptop stopped working after 2 weeks"
        ]
        
        for i, example in enumerate(examples):
            if st.button(f"Test {i+1}", key=f"test_{i}"):
                st.session_state.example_query = example
    
    # Display results
    if hasattr(st.session_state, 'last_results'):
        results = st.session_state.last_results
        run_id = st.session_state.last_run_id
        query = st.session_state.last_query
        
        st.markdown("---")
        st.subheader("🔍 Results")
        
        # Metrics
        col_m1, col_m2, col_m3 = st.columns(3)
        
        with col_m1:
            st.metric("Steps", len(results.get("steps", [])))
        
        with col_m2:
            confidence = results.get("confidence")
            if confidence:
                st.metric("Confidence", f"{confidence*100:.1f}%")
            else:
                st.metric("Confidence", "N/A")
        
        with col_m3:
            status = "Done" if results.get("done") else "Running"
            st.metric("Status", status)
        
        # Check for human review needed - FIX THE NONE ERROR HERE
        decision = results.get("final") or ""  # Handle None case
        needs_review = any(word in decision.lower() for word in ["blocked", "human review", "violation"]) if decision else False
        
        if needs_review:
            render_human_review(run_id, query, decision, results.get("steps", []))
        
        # Show timeline
        st.subheader("🔄 Timeline")
        
        steps = results.get("steps", [])
        for i, step in enumerate(steps, 1):
            step_type = step.get("step", "Unknown")
            description = step.get("desc", "")
            observation = step.get("obs")
            
            emoji = {"Thought": "🤔", "Observation": "👁️", "Decision": "⚖️", "Fallback": "🚨"}.get(step_type, "📋")
            
            st.markdown(f"**Step {i}: {emoji} {step_type}**")
            st.write(description)
            
            if observation:
                with st.expander("View Details"):
                    if isinstance(observation, str):
                        try:
                            obs_data = json.loads(observation)
                            st.json(obs_data)
                        except:
                            st.code(observation)
                    else:
                        st.json(observation)
            
            st.markdown("---")
        
        # Final decision (if no human review needed)
        if not needs_review and decision:
            st.subheader("🎯 Final Decision")
            
            if "approved" in decision.lower():
                st.success(f"✅ {decision}")
            elif "denied" in decision.lower():
                st.error(f"❌ {decision}")
            else:
                st.info(f"📋 {decision}")
    
    # Handle example selection
    if hasattr(st.session_state, 'example_query'):
        st.info(f"💡 Selected: {st.session_state.example_query}")
        del st.session_state.example_query

if __name__ == "__main__":
    main()