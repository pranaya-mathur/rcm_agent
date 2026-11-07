"""
12_RCM_AI_Copilot.py — RCM AI Copilot (LangGraph) with Premium UI
"""

import streamlit as st
from utils_ui import shared_page_init, render_page_header, COLORS, metric_card
from src.agents.langgraph_rcm_chatbot import rcm_langgraph_app
from src.core.data_loader import get_cpt_summary
from config import llm_provider_label, active_llm_model


def page_rcm_ai_copilot():
    master, _, _ = shared_page_init()

    render_page_header(
        "🧠 RCM AI Copilot (LangGraph)",
        "Claim-level copilot for denial, coding mismatch, and fraud reasoning with explainable outputs."
    )
    st.caption(
        "Explanations are grounded in model estimates + SHAP drivers for the selected claim. "
        "Synthetic demo data — recommendations require human review. Provider failures are shown as errors."
    )

    from src.core.demo_db import db_signature
    from src.core.claim_scoring import get_or_train_models, score_existing_claim

    @st.cache_resource(show_spinner="Loading models for Copilot...")
    def cached_models(db_sig: float):
        cpt_summary = get_cpt_summary(db_sig=db_sig)
        bundle = get_or_train_models(master, cpt_summary, train_if_missing=True)
        a = bundle.artifacts
        return {
            "denial": (a["denial_model"], a["denial_meta"]),
            "mismatch": (a["mismatch_model"], a["mismatch_meta"]),
            "fraud": (a["fraud_prob_model"], a["fraud_prob_meta"]),
            "cpt_summary": cpt_summary,
            "bundle": bundle,
        }

    models = cached_models(db_signature())

    if "lg_messages" not in st.session_state:
        st.session_state["lg_messages"] = []

    st.sidebar.markdown("### 🔍 Copilot Context")
    st.sidebar.caption(
        f"LLM Provider: **{llm_provider_label()}**  \n"
        f"Model: **{active_llm_model()}**"
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
    scored = None
    claim_row = None
    if not claim_rows.empty:
        try:
            scored = score_existing_claim(int(claim_id), train_if_missing=True)
        except Exception as e:
            st.error(f"Scoring failed for claim {claim_id}: {e}")
            scored = {"ok": False, "error": str(e)}
        if scored.get("ok"):
            claim_row = scored["claim"]
            preds = scored["predictions"]
            m_col1, m_col2, m_col3 = st.columns(3)
            with m_col1:
                metric_card(
                    "Denial Probability",
                    f"{float(preds.get('denial_probability', 0))*100:.1f}%",
                    preds.get("risk_level", ""),
                    delta_up=False if float(preds.get("denial_probability", 0)) > 0.5 else True,
                )
            with m_col2:
                status_text = (
                    "🔴 Denied"
                    if claim_row.get("is_denied")
                    else ("🟡 Scrubbing" if not claim_row.get("is_clean_claim") else "🟢 Clean")
                )
                st.markdown(
                    f"<div style='font-size:0.8rem; color:#94A3B8; margin-top:0.5rem;'>"
                    f"<strong>Status:</strong> {status_text}</div>",
                    unsafe_allow_html=True,
                )
                st.markdown(
                    f"<div style='font-size:0.8rem; color:#94A3B8;'>"
                    f"<strong>Payer:</strong> {claim_row.get('insurance')}</div>",
                    unsafe_allow_html=True,
                )
            with m_col3:
                st.markdown(
                    f"<div style='font-size:0.8rem; color:#94A3B8; margin-top:0.5rem;'>"
                    f"<strong>Amount:</strong> ${float(claim_row.get('claim_amount', 0)):,.2f}</div>",
                    unsafe_allow_html=True,
                )
                st.caption(
                    f"Mismatch {float(preds.get('mismatch_probability', 0))*100:.1f}% · "
                    f"Fraud {float(preds.get('fraud_probability_improved', 0))*100:.1f}%"
                )
        else:
            st.warning(scored.get("error", "Unable to score claim"))

    st.markdown("---")

    for msg in st.session_state["lg_messages"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if q := st.chat_input("Ask the RCM AI Copilot about this claim..."):
        st.session_state["lg_messages"].append({"role": "user", "content": q})
        with st.chat_message("user"):
            st.markdown(q)

        if claim_rows.empty or not scored or not scored.get("ok"):
            ans = "I could not find or score data for that claim ID."
        else:
            with st.status("🧠 Running Copilot reasoning...", expanded=False) as status:
                st.write("Using scored claim context (denial / mismatch / fraud / appeals estimates)...")
                preds = scored["predictions"]
                insights = {
                    "denial_probability": float(preds.get("denial_probability", 0.0)),
                    "mismatch_probability": float(preds.get("mismatch_probability", 0.0)),
                    "fraud_probability_improved": float(preds.get("fraud_probability_improved", 0.0)),
                    "p_appeal_success": float(preds.get("p_appeal_success", 0.0)),
                    "expected_recovery": float(preds.get("expected_recovery", 0.0)),
                    "risk_level": preds.get("risk_level", "LOW"),
                    "top_drivers": preds.get("top_drivers") or [],
                }
                state = {
                    "claim_id": int(claim_id),
                    "user_message": q,
                    "claim": claim_row,
                    "insights": insights,
                    "mode": "denial_copilot",
                }
                res = rcm_langgraph_app.invoke(state)
                ans = res.get("response", "Error processing request.")
                if res.get("error"):
                    status.update(label="⚠️ LLM provider error", state="error")
                else:
                    status.update(label="✅ Copilot response ready (human review required)", state="complete")

        st.session_state["lg_messages"].append({"role": "assistant", "content": ans})
        with st.chat_message("assistant"):
            st.markdown(ans)


if __name__ == "__main__":
    page_rcm_ai_copilot()

