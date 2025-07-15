🧠 AgentScope: AI Agent Timeline Debugger & Reliability Monitor
AgentScope is a developer-focused observability tool designed to debug multi-step AI agents by providing live reasoning timelines, confidence monitoring, and early fallback detection.

🚀 Overview
Problem: AI agents often hallucinate, silently fail, or provide low-confidence decisions without exposing reasoning steps.

Solution: AgentScope visualizes step-wise reasoning chains, policy compliance checks, and confidence scores, enabling rapid debugging and reliability analysis—especially useful in customer support, fraud detection, and LLM workflow testing.

📌 Features
✅ Live Timeline Debugger — See thoughts, actions, observations, and decisions step by step.

✅ Confidence Monitor — Live tracking of confidence scores during reasoning.

✅ Fallback Detector — Triggers fallback pathways when policy restrictions or low confidence are detected.

✅ Real Transaction Lookup — Uses CSV data for mock transaction & policy lookup.

✅ Streaming Timeline UI — Built with Streamlit for real-time, smooth debugging experience.

⚙️ Tech Stack
Layer	Technology
Backend	Python (FastAPI) + TogetherAI API
Frontend	Streamlit Timeline Visualizer
Agent Logic	ReAct-style LLM + Policy CSV Lookup
Data	transactions.csv, policy.csv

🏁 Quickstart Guide
1. Clone Repository
bash
Copy
Edit
git clone https://github.com/your-username/agentscope.git
cd agentscope
2. Setup Backend
bash
Copy
Edit
cd backend
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
✅ Set your TogetherAI API Key in .env:

bash
Copy
Edit
echo "TOGETHER_API_KEY=your_key_here" > .env
3. Start Backend API
bash
Copy
Edit
uvicorn app:app --reload
4. Setup Frontend
bash
Copy
Edit
cd ../frontend
streamlit run streamlit_app.py
Access frontend at http://localhost:8501.

🧪 Evaluation
Evaluation notebook at /evaluation/agentscope_evaluation_notebook.py

Supports automatic tests for:

Policy compliance

Correct fallback triggering

Confidence range validation

📂 Project Structure
bash
Copy
Edit
agentscope/
│
├── backend/
│   ├── app.py              # FastAPI server
│   ├── agent.py            # Refund agent logic
│   ├── timeline_logger.py  # Trace logging
│   ├── data/transactions.csv
│   ├── data/policy.csv
│
├── frontend/
│   └── streamlit_app.py    # Live timeline UI
│
├── evaluation/
│   └── agentscope_evaluation_notebook.py
│
└── README.md
📈 Demo Video

📌 **Architecture Diagram:** See [`[docs/architecture_diagram.md](https://github.com/KATREDDIDURGA/AI-Projects/blob/main/agentscopev2/agentscope_architecture.png)`]([docs/architecture_diagram.md](https://github.com/KATREDDIDURGA/AI-Projects/blob/main/agentscopev2/agentscope_architecture.png)).

📌 Future Extensions
✅ Multi-agent support (LangGraph)

✅ Slack/Email alerting

✅ Docker containerization

✅ Token-level uncertainty monitoring

📜 License
MIT License.
