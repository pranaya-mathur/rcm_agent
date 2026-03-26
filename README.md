# AI-Powered Smart RCM Prototype

This repository contains a Streamlit-based Revenue Cycle Management (RCM) prototype with:

- Predictive denial modeling (XGBoost + SHAP)
- Smart claim scrubbing (CPT-ICD mismatch probability + recommendation)
- Appeals prioritization (success probability + expected recovery)
- Fraud enhancement (supervised fraud probability + Isolation Forest anomaly signal)
- Agentic workflow demo (coordinator + specialized agents)
- LangGraph chatbot for claim-level conversational insights
- Groq/Ollama-based 2-page summary generation for demo artifacts

---

## Prototype Architecture

```mermaid
flowchart LR
    A[CSV Data Sources<br/>claims, denials, appeals, payments,<br/>fraud, scrubbing, icd, cpt_lines, events] --> B[data_loader.py<br/>build_master + get_cpt_summary]
    B --> C[ml_engine.py<br/>Model Training + Scoring]
    C --> D[models/*.joblib<br/>Persisted Artifacts]
    B --> E[app.py<br/>Streamlit Dashboard]
    C --> E
    F[rcm_agent.py<br/>Coordinator + Agents] --> E
    G[langgraph_rcm_chatbot.py<br/>LangGraph Flow] --> E
    H[groq_agent_summary.py<br/>2-page summary via Groq] --> E
    I[ollama_agent_summary.py<br/>2-page summary via Ollama] --> E
```

---

## ML Pipeline (Training + Inference)

```mermaid
flowchart TD
    A[build_master + get_cpt_summary] --> B[Feature Engineering]
    B --> C1[Denial Model<br/>XGBoost Classifier]
    B --> C2[Mismatch Model<br/>XGBoost Classifier]
    B --> C3[Appeals Success<br/>XGBoost Classifier]
    B --> C4[Appeals Recovery<br/>XGBoost Regressor]
    B --> C5[Fraud Probability<br/>XGBoost Classifier]
    A --> C6[Fraud Anomaly<br/>Isolation Forest]

    C1 --> D1[denial_probability + SHAP]
    C2 --> D2[mismatch_probability]
    C3 --> D3[p_appeal_success]
    C4 --> D4[expected_recovery]
    C5 --> D5[fraud_probability]
    C6 --> D6[anomaly_probability]

    D5 --> E[fraud_probability_improved<br/>weighted blend]
    D6 --> E
```

---

## Agentic Workflow (Prototype)

```mermaid
flowchart LR
    A[Claim Selected] --> B[Coordinator Agent]
    B --> C[Denial Prediction Agent]
    B --> D[Scrubbing Agent]
    B --> E[Appeals Agent]
    B --> F[Fraud Agent]

    C --> G[Risk Assessment]
    D --> H[Recommendation<br/>Is claim mein Auth required add kar do]
    E --> I[Appealability + Draft Letter]
    F --> J[Fraud Review Flag]

    G --> K[Final Action Plan]
    H --> K
    I --> K
    J --> K
    K --> L[Human-in-the-loop Approval]
```

---

## LangGraph Chatbot Flow

```mermaid
flowchart TD
    A[User Message + Claim ID] --> B[Prepare Claim Insights<br/>denial, mismatch, appeals, fraud]
    B --> C[LangGraph Node: decide_topics]
    C --> D[LangGraph Node: generate_response]
    D --> E[Chat Response in Streamlit]

    F[Optional SHAP Request<br/>why/shap/driver] --> G[predict_single_claim + SHAP]
    G --> D
```

---

## Main Application Pages

- Executive Summary
- Denial Intelligence
- Appeals Analytics
- Fraud Detection
- Smart Scrubbing
- AR Aging & Lifecycle
- AI Denial Predictor
- Agentic RCM Agent (Demo)
- LangGraph Chatbot (Demo)

---

## Run Locally

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

### Optional: React UI Prototype (No Node Required)

A polished React-based no-build UI is available in `frontend/` for presentation/demo.

Run:

```bash
python3 backend_api.py
```

API endpoints:
- `GET /api/health`
- `GET /api/summary`
- `GET /api/agent/claim/<claim_id>`

In another terminal:

```bash
cd frontend
python3 -m http.server 8080
```

Then open:

- http://localhost:8080

---

## Notes

- This is a prototype/demo implementation.
- Models are trained from CSV datasets in `data/` and persisted in `models/`.
- Summary generation supports:
  - Groq (`GROQ_API_KEY` required)
  - Ollama (local server required)
