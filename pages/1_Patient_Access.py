"""
1_Patient_Access.py — Patient Access & Eligibility Page with Real-Time Groq AI
"""

import streamlit as st
import plotly.graph_objects as go
from utils_ui import (
    shared_page_init, render_page_header, plotly_layout, 
    COLORS, fmt_number, fmt_dollar, metric_card
)
from src.core import ml_engine
from src.agents.langgraph_rcm_chatbot import rcm_langgraph_app

def page_patient_access_eligibility():
    master, _, _ = shared_page_init()
    
    render_page_header(
        "🧾 Patient Access & Eligibility",
        "Front-end automation for registration quality, eligibility checks, and AI-powered registration support."
    )

    df = master.copy()
    if "insurance_pat" not in df.columns:
        df["insurance_pat"] = df["insurance"]
    df["insurance_pat"] = df["insurance_pat"].fillna("Unknown")
    df["insurance_match"] = (df["insurance"].astype(str) == df["insurance_pat"].astype(str))

    @st.cache_resource(show_spinner=False)
    def cached_eligibility_model():
        m, meta = ml_engine.load_eligibility_risk_model()
        if m is None:
            with st.spinner("🚀 Training eligibility risk model..."):
                res = ml_engine.train_eligibility_risk_model(df)
                return res["model"], res["meta"]
        return m, meta

    elig_model, elig_meta = cached_eligibility_model()
    df["eligibility_risk"] = ml_engine.score_eligibility_risk(df, elig_model, elig_meta["feature_cols"])

    match_rate = df["insurance_match"].mean() * 100
    high_risk_count = int((df["eligibility_risk"] >= 0.65).sum())
    est_avoidable = float(df[df["eligibility_risk"] >= 0.65]["claim_amount"].sum() * 0.15)

    # Custom Metric Cards
    k1, k2, k3, k4 = st.columns(4)
    with k1:
         metric_card("Patients Covered", fmt_number(df["patient_id"].nunique()))
    with k2:
         metric_card("Insurance Match Rate", f"{match_rate:.1f}%", "Target: 98%+")
    with k3:
         metric_card("High Eligibility Risk", fmt_number(high_risk_count), "Requires Review", delta_up=False)
    with k4:
         metric_card("Avoidable Leakage (Est.)", fmt_dollar(est_avoidable), "Front-end Savings")

    st.markdown("<br>", unsafe_allow_html=True)
    
    c1, c2 = st.columns([1.2, 1])
    with c1:
        st.markdown("### 📊 Insurance Match Rate by Payer")
        payer_match = df.groupby("insurance").agg(
            total=("claim_id", "count"),
            match=("insurance_match", "mean"),
        ).reset_index()
        payer_match["match"] = payer_match["match"] * 100
        fig = go.Figure(go.Bar(
            x=payer_match["insurance"],
            y=payer_match["match"],
            marker_color=COLORS["secondary"],
            text=payer_match["match"].round(1).astype(str) + "%",
            textposition="outside"
        ))
        plotly_layout(fig, "", 350)
        fig.update_yaxes(title="Match %", range=[0, 110])
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.markdown("### ⚠️ High-Risk Eligibility Cases")
        high_elig = df[df["eligibility_risk"] >= 0.65][
            ["claim_id", "patient_id", "insurance", "claim_amount", "eligibility_risk"]
        ].sort_values("eligibility_risk", ascending=False).head(15)
        st.dataframe(high_elig, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown("### 💬 Patient Access AI (Groq-Powered)")
    
    chat_key = "pa_ai_messages"
    if chat_key not in st.session_state:
        st.session_state[chat_key] = [
            {
                "role": "assistant",
                "content": "I am your AI Patient Access specialist. I can help with insurance match verification, eligibility risk analysis, and patient lookups using Real-time Groq Reasoning."
            }
        ]

    # Use a specific patient context for the chat if selected
    selected_claim_id = st.selectbox("Select Claim ID to Analyze", options=df["claim_id"].head(50).tolist())
    
    for m in st.session_state[chat_key]:
        with st.chat_message(m["role"]): st.markdown(m["content"])

    if q := st.chat_input("Ask about eligibility risk or patient registration..."):
        st.session_state[chat_key].append({"role": "user", "content": q})
        with st.chat_message("user"): st.markdown(q)
        
        # Build context for the agent
        claim_rows = df[df["claim_id"] == selected_claim_id]
        if claim_rows.empty:
            ans = "I couldn't find data for that claim ID."
        else:
            claim_row = claim_rows.iloc[0].to_dict()
            with st.status("🧠 AI Patient Access Reasoning...", expanded=False) as status:
                # Eligibility score for this claim
                elig_risk = float(claim_row["eligibility_risk"])
                insights = {
                    "eligibility_risk": elig_risk,
                    "insurance_match": bool(claim_row["insurance_match"]),
                    "denial_probability": elig_risk, # repurposed for context
                    "mismatch_probability": 0.0,
                    "fraud_probability_improved": 0.0,
                    "topics": ["patient access", "eligibility"]
                }
                state = {
                    "claim_id": selected_claim_id,
                    "user_message": q,
                    "claim": claim_row,
                    "insights": insights,
                }
                res = rcm_langgraph_app.invoke(state)
                ans = res.get("response", "Error processing request.")
                status.update(label="✅ Analysis Complete", state="complete")

        st.session_state[chat_key].append({"role": "assistant", "content": ans})
        with st.chat_message("assistant"): st.markdown(ans)

if __name__ == "__main__":
    page_patient_access_eligibility()
