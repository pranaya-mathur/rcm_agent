"""
2_Smart_Scrubbing.py — Smart Scrubbing & Clean Claim Rate Page
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from utils_ui import shared_page_init, plotly_layout, COLORS, fmt_number, fmt_dollar, PALETTE
from config import SCRUB_RECOMMENDATION_HIGH_RISK, SCRUB_RECOMMENDATION_STANDARD
from src.core import ml_engine
from src.core.data_loader import get_cpt_summary

def page_scrubbing():
    master, _, _ = shared_page_init()
    
    st.markdown("# 🧹 Smart Scrubbing & Clean Claim Rate")
    st.caption("Pre-submission validation flags — reduce denials before they happen")

    clean_rate = master["is_clean_claim"].mean() * 100
    mismatch_rate = master["cpt_icd_mismatch"].mean() * 100
    high_amt_rate = master["high_amount_flag"].mean() * 100
    strict_ins_rate = master["strict_insurance_flag"].mean() * 100

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Clean Claim Rate", f"{clean_rate:.1f}%", delta="Target: 95%+")
    k2.metric("CPT-ICD Mismatch", f"{mismatch_rate:.1f}%", delta="Should be <2%", delta_color="inverse")
    k3.metric("High Amount Flagged", f"{high_amt_rate:.1f}%")
    k4.metric("Strict Insurance", f"{strict_ins_rate:.1f}%")

    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=clean_rate,
            title={"text": "Clean Claim Rate", "font": {"size": 18, "color": COLORS["text"]}},
            number={"suffix": "%", "font": {"size": 48, "color": COLORS["success"]}},
            gauge=dict(
                axis=dict(range=[0, 100], tickcolor=COLORS["muted"]),
                bar=dict(color=COLORS["success"]),
                bgcolor=COLORS["card"],
                steps=[
                    dict(range=[0, 70], color="rgba(239,68,68,0.15)"),
                    dict(range=[70, 90], color="rgba(245,158,11,0.15)"),
                    dict(range=[90, 100], color="rgba(34,197,94,0.15)"),
                ],
                threshold=dict(line=dict(color=COLORS["accent"], width=3), value=95),
            )
        ))
        plotly_layout(fig, "", 380)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        flags = pd.DataFrame({
            "Flag": ["CPT-ICD Mismatch", "High Amount", "Strict Insurance"],
            "Count": [master["cpt_icd_mismatch"].sum(), master["high_amount_flag"].sum(),
                      master["strict_insurance_flag"].sum()],
            "Rate": [mismatch_rate, high_amt_rate, strict_ins_rate]
        })
        fig = go.Figure(go.Bar(
            x=flags["Flag"], y=flags["Count"],
            marker_color=[COLORS["danger"], COLORS["warning"], COLORS["secondary"]],
            text=flags["Rate"].round(1).astype(str) + "%", textposition="outside"
        ))
        plotly_layout(fig, "🚩 Scrubbing Flags Breakdown", 380)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.markdown("### 🤖 Phase 3 — ML CPT-ICD Mismatch Detection")

    @st.cache_resource(show_spinner=False)
    def cached_models():
        m_model, m_meta = ml_engine.load_mismatch_model()
        if m_model is None:
            with st.spinner("🚀 Training CPT-ICD mismatch detector..."):
                res = ml_engine.train_mismatch_model(master, get_cpt_summary())
                m_model, m_meta = res["model"], res["meta"]
        
        d_model, d_meta = ml_engine.load_model()
        if d_model is None:
            with st.spinner("🚀 Training denial predictor..."):
                res = ml_engine.train_model(master, get_cpt_summary())
                d_model = res["model"]
                d_meta = {"feature_cols": res["feature_cols"], "all_probabilities": res["all_probabilities"]}
        return m_model, m_meta, d_model, d_meta

    mismatch_model, mismatch_meta, denial_model, denial_meta = cached_models()
    cpt_s = get_cpt_summary()
    mismatch_probs = ml_engine.score_all_mismatch(master, cpt_s, mismatch_model, mismatch_meta["feature_cols"])
    denial_probs = ml_engine.score_denial_portfolio(master, cpt_s, denial_model, denial_meta["feature_cols"])

    phase3_df = master[["claim_id", "insurance", "claim_amount", "visit_type", "icd_code", "cpt_icd_mismatch"]].copy()
    phase3_df["mismatch_probability"] = mismatch_probs
    phase3_df["denial_probability"] = denial_probs
    phase3_df["recommendation"] = np.where(
        (phase3_df["denial_probability"] >= 0.7) | (phase3_df["mismatch_probability"] >= 0.7),
        SCRUB_RECOMMENDATION_HIGH_RISK,
        SCRUB_RECOMMENDATION_STANDARD,
    )
    st.dataframe(phase3_df.sort_values("mismatch_probability", ascending=False).head(20), use_container_width=True, hide_index=True)

if __name__ == "__main__":
    page_scrubbing()
