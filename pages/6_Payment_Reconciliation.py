"""
6_Payment_Reconciliation.py — Payment Reconciliation Page
"""

import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from utils_ui import shared_page_init, plotly_layout, COLORS, fmt_number, fmt_dollar
from src.core import ml_engine

def page_payment_reconciliation():
    master, _, _ = shared_page_init()
    
    st.markdown("# 💳 Payment Reconciliation")
    st.caption("Payment posting quality and reconciliation opportunities (ML-powered).")

    df = master.copy()
    df["posting_gap"] = (df["claim_amount"] - df["paid_amount"]).clip(lower=0)

    @st.cache_resource(show_spinner=False)
    def cached_recon_model():
        model, meta = ml_engine.load_reconciliation_risk_model()
        if model is None:
            with st.spinner("🚀 Training reconciliation risk model..."):
                res = ml_engine.train_reconciliation_risk_model(df)
                return res["model"], res["meta"]
        return model, meta

    recon_model, recon_meta = cached_recon_model()
    df["recon_risk_probability"] = ml_engine.score_reconciliation_risk(df, recon_model, recon_meta["feature_cols"])
    df["recon_status"] = np.where(df["recon_risk_probability"] >= 0.65, "Needs Review", "Auto-Matched")

    auto_rate = (df["recon_status"] == "Auto-Matched").mean() * 100
    review_count = int((df["recon_status"] == "Needs Review").sum())
    review_amount = float(df[df["recon_status"] == "Needs Review"]["posting_gap"].sum())

    k1, k2, k3 = st.columns(3)
    k1.metric("Auto-Matched Rate", f"{auto_rate:.1f}%")
    k2.metric("Needs Review Claims", fmt_number(review_count))
    k3.metric("Unreconciled Amount", fmt_dollar(review_amount))

    st.markdown("---")
    payer_recon = df.groupby("insurance").agg(
        claims=("claim_id", "count"),
        auto_match=("recon_status", lambda x: (x == "Auto-Matched").mean() * 100),
        unreconciled=("posting_gap", "sum"),
    ).reset_index()
    c1, c2 = st.columns(2)
    with c1:
        fig = go.Figure(go.Bar(x=payer_recon["insurance"], y=payer_recon["auto_match"], marker_color=COLORS["success"]))
        plotly_layout(fig, "Auto-Match Rate by Payer", 330)
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        top_review = df[df["recon_status"] == "Needs Review"][["claim_id", "insurance", "claim_amount", "paid_amount", "posting_gap", "recon_risk_probability"]].sort_values("recon_risk_probability", ascending=False).head(15)
        st.dataframe(top_review, use_container_width=True, hide_index=True)

if __name__ == "__main__":
    page_payment_reconciliation()
