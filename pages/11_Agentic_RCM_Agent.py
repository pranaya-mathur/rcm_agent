"""
11_Agentic_RCM_Agent.py — Agentic RCM Agent Page with Premium UI
"""

import streamlit as st
import os
from utils_ui import shared_page_init, render_page_header, metric_card, fmt_dollar
from src.core import ml_engine
from src.agents.rcm_agent import CoordinatorAgent
from src.core.data_loader import load_all, get_cpt_summary

def page_agentic_rcm_agent():
    master, _, _ = shared_page_init()
    
    render_page_header(
        "🤖 Agentic RCM Orchestrator",
        "Multi-step AI orchestration for coding validation, payer compliance, and automated clinical reasoning."
    )

    @st.cache_resource(show_spinner=False)
    def cached_predictions():
        cpt_s = get_cpt_summary()
        raw_l = load_all()
        # simplified for demo
        by_id = {}
        for _, row in master.head(100).iterrows(): # limited for performance
            by_id[int(row['claim_id'])] = {
                "denial_probability": 0.1,
                "mismatch_probability": 0.05,
                "fraud_probability_improved": 0.02,
                "reconciliation_risk_probability": 0.01,
            }
        return by_id

    preds_by_id = cached_predictions()
    
    st.sidebar.markdown("### 🔍 Execution Context")
    claim_id = st.sidebar.number_input("Target Claim ID", min_value=1, step=1, value=int(master["claim_id"].iloc[0]))
    
    if claim_id not in master["claim_id"].values:
        st.error("Claim ID not found.")
        return

    claim_row = master[master["claim_id"] == claim_id].iloc[0].to_dict()
    
    # Showcase metrics for the claim
    m1, m2, m3 = st.columns(3)
    with m1:
        metric_card("Claim Amount", fmt_dollar(claim_row['claim_amount']))
    with m2:
        status_text = "🔴 Denied" if claim_row['is_denied'] else ("🟡 Scrubbing" if not claim_row['is_clean_claim'] else "🟢 Clean")
        metric_card("Current Status", status_text, "RCM Workflow")
    with m3:
        metric_card("Payer", claim_row['insurance'], "Verification Active")

    st.markdown("---")
    
    if st.button("🚀 Execute Agentic Workflow", use_container_width=True):
        with st.status("🏗️ Orchestrating RCM Agents...", expanded=True) as status:
            st.write("Initializing Coordinator Agent...")
            agent = CoordinatorAgent()
            st.write("Fetching multi-modal predictions...")
            agent_out = agent.run(claim=claim_row, predictions=preds_by_id.get(claim_id, {}))
            status.update(label="✅ Workflow Execution Complete", state="complete")

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("### 📋 Final Agentic Recommendation")
        st.success(agent_out.recommendation)
        
        st.markdown("#### 🔄 Trace Log")
        for s in agent_out.steps:
            with st.expander(f"📌 {s.agent} | {s.step}"):
                st.write(s.summary)

if __name__ == "__main__":
    page_agentic_rcm_agent()
