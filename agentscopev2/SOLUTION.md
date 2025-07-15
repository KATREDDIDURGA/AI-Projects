# AgentScope - AI Agent Reasoning Debugger

## 📌 Problem Statement
As AI agents become more complex, debugging their multi-step reasoning becomes difficult. Businesses need a transparent, reliable system to trace decisions, confidence levels, and fallback triggers in real-time.

## 🎯 Solution Overview
AgentScope is a production-ready multi-step AI agent debugger with:
- Live step-wise reasoning timeline (Thought → Action → Observation)
- Confidence score tracing
- Fallback monitoring for safety
- Real-time API with FastAPI + Streamlit dashboard

## 🛠️ Architecture
- **Agent Layer:** TogetherAI-powered ReAct agent
- **Backend:** FastAPI serving reasoning trace via API
- **Observability:** Timeline logger with fallback triggers
- **Frontend:** Streamlit timeline replay interface
- **Evaluation Notebook:** Automated pass/fail metrics and confidence scoring

## 🎁 Features
| Feature | Description |
|----------|-------------|
| Multi-step Reasoning | Traces every reasoning step in real-time |
| Confidence Monitoring | Shows confidence progression per query |
| Fallback Detection | Identifies low-confidence decisions & triggers fallback |
| Timeline Replay | Full reasoning trace replay in Streamlit UI |
| API-first Design | Modular FastAPI backend for future extensions |

## 📂 Project Structure
agentscopev2/
├── backend/ (FastAPI + Agent Logic)
├── frontend/ (Streamlit Dashboard)
├── evaluation/ (Automated evaluation notebook)
├── data/ (transactions.csv, policy.csv)
└── README.md, SOLUTION.md, DEMO.md



## 🚀 How to Run
1. `cd backend && source venv/bin/activate && uvicorn app:app --reload`
2. `streamlit run frontend/streamlit_app.py`
3. Open browser at `http://localhost:8501`

## 🎉 Outcome
AgentScope delivers a CEO-demo ready AI reasoning debugger with complete transparency and traceability, suitable for any LLM-based system.
