"""
8_Fraud_Detection.py — Fraud Detection Page
"""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from utils_ui import shared_page_init, plotly_layout, COLORS, PALETTE, fmt_number, fmt_dollar
from src.core import ml_engine
from src.core.data_loader import get_cpt_summary

def page_fraud_detection():
    master, _, _ = shared_page_init()
    
    st.markdown("# 🔍 Fraud Detection")
    st.caption("AI-powered fraud scoring, risk analysis, and anomaly detection")

    flagged = master[master["fraud_flag"] == True]

    @st.cache_resource(show_spinner=False)
    def cached_fraud_models():
        prob_model, prob_meta = ml_engine.load_fraud_probability_model()
        if prob_model is None:
            with st.spinner("🚀 Training fraud probability model..."):
                res = ml_engine.train_fraud_probability_model(master, get_cpt_summary())
                prob_model, prob_meta = res["model"], res["meta"]
        
        anom_model, anom_meta = ml_engine.load_fraud_anomaly_model()
        if anom_model is None:
            with st.spinner("🚀 Training fraud anomaly model..."):
                res = ml_engine.train_fraud_anomaly_model(master)
                anom_model, anom_meta = res["model"], res["meta"]
        return prob_model, prob_meta, anom_model, anom_meta

    prob_model, prob_meta, anom_model, anom_meta = cached_fraud_models()
    fraud_enhanced = ml_engine.score_fraud_enhanced(master, get_cpt_summary(), prob_model, prob_meta["feature_cols"], anom_model, anom_meta)

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total Claims Analyzed", fmt_number(len(master)))
    k2.metric("Fraud Flagged", fmt_number(len(flagged)), delta=f"{len(flagged)/len(master)*100:.1f}%", delta_color="inverse")
    k3.metric("Flagged Amount", fmt_dollar(flagged["claim_amount"].sum()))
    k4.metric("Avg Fraud Prob", f"{fraud_enhanced['fraud_probability_improved'].mean():.3f}")

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        fig = px.histogram(fraud_enhanced, x="fraud_probability_improved", nbins=50, color_discrete_sequence=[COLORS["primary"]])
        fig.add_vline(x=0.7, line_dash="dash", line_color=COLORS["danger"])
        plotly_layout(fig, "📊 Fraud Probability Distribution", 380)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        payer_fraud = master.groupby("insurance").agg(total=("claim_id", "count"), flagged=("fraud_flag", "sum")).reset_index()
        payer_fraud["rate"] = (payer_fraud["flagged"] / payer_fraud["total"] * 100).round(2)
        fig = go.Figure(go.Bar(y=payer_fraud["insurance"], x=payer_fraud["rate"], orientation="h", marker_color=PALETTE[:len(payer_fraud)]))
        plotly_layout(fig, "🏢 Fraud Rate by Payer", 380)
        st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    page_fraud_detection()
