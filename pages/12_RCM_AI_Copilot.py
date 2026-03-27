"""
12_RCM_AI_Copilot.py — RCM AI Copilot (Groq + LangGraph) with Premium UI
"""

import streamlit as st
from utils_ui import shared_page_init, render_page_header, COLORS, metric_card
from src.agents.langgraph_rcm_chatbot import rcm_langgraph_app
from src.core import ml_engine
from src.core.data_loader import get_cpt_summary
from config import DEFAULT_LLM, GROQ_MODEL


def page_rcm_ai_copilot():
    master, _, _ = shared_page_init()

    render_page_header(
        "🧠 RCM AI Copilot (Groq + LangGraph)",
        "Claim-level copilot for denial, coding mismatch, and fraud reasoning with explainable outputs."
    )

    @st.cache_resource(show_spinner=False)
    def cached_models():
        cpt_summary = get_cpt_summary()
        denial_model, denial_meta = ml_engine.load_model()
        if denial_model is None:
            res = ml_engine.train_model(master, cpt_summary)
            denial_model, denial_meta = res["model"], res

        mismatch_model, mismatch_meta = ml_engine.load_mismatch_model()
        if mismatch_model is None:
            res = ml_engine.train_mismatch_model(master, cpt_summary)
            mismatch_model, mismatch_meta = res["model"], res["meta"]

        fraud_model, fraud_meta = ml_engine.load_fraud_probability_model()
        if fraud_model is None:
            res = ml_engine.train_fraud_probability_model(master, cpt_summary)
            fraud_model, fraud_meta = res["model"], res["meta"]

        return {
            "denial": (denial_model, denial_meta),
            "mismatch": (mismatch_model, mismatch_meta),
            "fraud": (fraud_model, fraud_meta),
            "cpt_summary": cpt_summary
        }

    models = cached_models()

    if "lg_messages" not in st.session_state:
        st.session_state["lg_messages"] = []

    st.sidebar.markdown("### 🔍 Copilot Context")
    st.sidebar.caption(
        f"LLM Provider: **{'Groq' if DEFAULT_LLM != 'ollama' else 'Ollama'}**  \n"
        f"Model: **{GROQ_MODEL if DEFAULT_LLM != 'ollama' else 'local-ollama'}**"
    )
    claim_id = st.sidebar.number_input("Focus Claim ID", min_value=1, step=1, value=int(master["claim_id"].iloc[0]))

    if st.session_state.get("last_chatbot_id") != claim_id:
        st.session_state["lg_messages"] = [{
            "role": "assistant",
            "content": (
                f"I have initialized Copilot analysis for **Claim #{claim_id}**. "
                "Ask about denial probability, top risk drivers, coding mismatch, or fraud indicators."
            )
        }]
        st.session_state["last_chatbot_id"] = claim_id

    claim_rows = master[master["claim_id"] == claim_id]
    if not claim_rows.empty:
        claim_row = claim_rows.iloc[0].to_dict()
        denial_model, denial_meta = models["denial"]
        d_res = ml_engine.predict_single_claim(claim_row, denial_model, denial_meta["feature_cols"])

        m_col1, m_col2, m_col3 = st.columns(3)
        with m_col1:
            metric_card(
                "Denial Probability",
                f"{d_res['denial_probability']*100:.1f}%",
                d_res["risk_level"],
                delta_up=False if d_res["denial_probability"] > 0.5 else True
            )
        with m_col2:
            status_text = "🔴 Denied" if claim_row["is_denied"] else ("🟡 Scrubbing" if not claim_row["is_clean_claim"] else "🟢 Clean")
            st.markdown(f"<div style='font-size:0.8rem; color:#94A3B8; margin-top:0.5rem;'><strong>Status:</strong> {status_text}</div>", unsafe_allow_html=True)
            st.markdown(f"<div style='font-size:0.8rem; color:#94A3B8;'><strong>Payer:</strong> {claim_row['insurance']}</div>", unsafe_allow_html=True)
        with m_col3:
            st.markdown(f"<div style='font-size:0.8rem; color:#94A3B8; margin-top:0.5rem;'><strong>Amount:</strong> ${claim_row['claim_amount']:,.2f}</div>", unsafe_allow_html=True)

    st.markdown("---")

    for msg in st.session_state["lg_messages"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if q := st.chat_input("Ask the RCM AI Copilot about this claim..."):
        st.session_state["lg_messages"].append({"role": "user", "content": q})
        with st.chat_message("user"):
            st.markdown(q)

        if claim_rows.empty:
            ans = "I could not find any data for that claim ID."
        else:
            with st.status("🧠 Running Copilot reasoning...", expanded=False) as status:
                st.write("Preparing model context...")
                denial_model, denial_meta = models["denial"]
                d_res = ml_engine.predict_single_claim(claim_row, denial_model, denial_meta["feature_cols"])

                st.write("Computing coding mismatch risk...")
                mismatch_model, mismatch_meta = models["mismatch"]
                cpt_summary = models["cpt_summary"]
                m_probs = ml_engine.score_all_mismatch(
                    master[master["claim_id"] == claim_id],
                    cpt_summary[cpt_summary["claim_id"] == claim_id],
                    mismatch_model,
                    mismatch_meta["feature_cols"],
                )

                st.write("Estimating fraud risk...")
                fraud_model, fraud_meta = models["fraud"]
                fraud_features = ml_engine._build_fraud_feature_df(
                    master[master["claim_id"] == claim_id],
                    cpt_summary[cpt_summary["claim_id"] == claim_id],
                    include_fraud_score=False,
                )[0].reindex(columns=fraud_meta["feature_cols"], fill_value=0)
                f_probs = fraud_model.predict_proba(fraud_features)[:, 1]

                insights = {
                    "denial_probability": d_res["denial_probability"],
                    "mismatch_probability": float(m_probs[0]) if len(m_probs) > 0 else 0.0,
                    "fraud_probability_improved": float(f_probs[0]) if len(f_probs) > 0 else 0.0,
                    "risk_level": d_res["risk_level"],
                    "top_drivers": d_res["shap_explanation"].to_dict(orient="records") if "shap_explanation" in d_res else [],
                }
                status.update(label="✅ Copilot context ready", state="complete")

                state = {
                    "claim_id": claim_id,
                    "user_message": q,
                    "claim": claim_row,
                    "insights": insights,
                }
                res = rcm_langgraph_app.invoke(state)
                ans = res.get("response", "Error processing request.")

        st.session_state["lg_messages"].append({"role": "assistant", "content": ans})
        with st.chat_message("assistant"):
            st.markdown(ans)


if __name__ == "__main__":
    page_rcm_ai_copilot()

