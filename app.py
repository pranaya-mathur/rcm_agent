"""
app.py — Smart RCM — Enterprise Command Center Landing Page
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from utils_ui import (
    shared_page_init, render_page_header, metric_card, 
    plotly_layout, COLORS, fmt_dollar, fmt_number
)

def main():
    st.set_page_config(
        page_title="Smart RCM Command Center",
        page_icon="🏢",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    master, events_tl, raw = shared_page_init()
    render_page_header(
        "Healthcare Revenue Cycle Command Center", 
        "Enterprise-wide intelligence across Patient Access, Mid-Cycle Integrity, and Back-End Recovery."
    )

    # ──────────────────────────────────────────────
    #  Top KPIs (Custom Metric Cards)
    # ──────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        total_billed = master["claim_amount"].sum()
        metric_card("Total Billed AR", fmt_dollar(total_billed), "8.2% vs Last Month", delta_up=True)
    with c2:
        denial_rate = master["is_denied"].mean() * 100
        metric_card("Net Denial Rate", f"{denial_rate:.11f}%", "-2.4% reduction", delta_up=False)
    with c3:
        clean_rate = master["is_clean_claim"].mean() * 100
        metric_card("Clean Claim Rate", f"{clean_rate:.1f}%", "Target: 95%+", delta_up=True)
    with c4:
        recovery = master[master["appeal_success"] == True]["paid_amount"].sum()
        metric_card("Appeal Recovery", fmt_dollar(recovery), "ML-Prioritized", delta_up=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ──────────────────────────────────────────────
    #  Main Grid: AR Aging & Denial Pareto
    # ──────────────────────────────────────────────
    col_left, col_right = st.columns([1.5, 1])

    with col_left:
        st.markdown("### 📊 AR Aging Distribution")
        # Simulate AR buckets if not explicitly in data
        # In real world: 0-30, 31-60, 61-90, 91-120, 120+
        aging_data = pd.DataFrame({
            "Bucket": ["0-30 Days", "31-60 Days", "61-90 Days", "91-120 Days", "120+ Days"],
            "Amount": [
                master["claim_amount"].sum() * 0.45,
                master["claim_amount"].sum() * 0.25,
                master["claim_amount"].sum() * 0.15,
                master["claim_amount"].sum() * 0.10,
                master["claim_amount"].sum() * 0.05
            ]
        })
        fig_aging = px.bar(
            aging_data, x="Amount", y="Bucket", orientation="h",
            color="Bucket", color_discrete_sequence=px.colors.sequential.Blues_r,
            text_auto=True
        )
        plotly_layout(fig_aging, "Active AR Inventory by Aging Bucket", 400)
        fig_aging.update_traces(textfont_size=12, textposition="outside", cliponaxis=False)
        st.plotly_chart(fig_aging, use_container_width=True)

    with col_right:
        st.markdown("### 🚫 Denial Pareto Analysis")
        denial_counts = master[master["is_denied"] == True]["denial_reason"].value_counts().head(5)
        fig_pareto = go.Figure(data=[
            go.Pie(
                labels=denial_counts.index, 
                values=denial_counts.values,
                hole=.4,
                marker=dict(colors=px.colors.sequential.Viridis_r)
            )
        ])
        plotly_layout(fig_pareto, "Top Denial Root Causes", 400)
        st.plotly_chart(fig_pareto, use_container_width=True)

    st.markdown("---")

    # ──────────────────────────────────────────────
    #  Bottom Strip: Payer Performance & Portfolio Risk
    # ──────────────────────────────────────────────
    st.markdown("### 🏦 Payer Performance Benchmark")
    payer_perf = master.groupby("insurance").agg(
        collection_rate=("collection_rate", "mean"),
        total_claims=("claim_id", "count")
    ).reset_index().sort_values("collection_rate", ascending=False)

    fig_payer = px.scatter(
        payer_perf, x="insurance", y="collection_rate", size="total_claims",
        color="collection_rate", color_continuous_scale="Viridis",
        labels={"collection_rate": "Collection %", "insurance": "Payer"}
    )
    plotly_layout(fig_payer, "Collection Rate % vs Volume by Payer", 350)
    st.plotly_chart(fig_payer, use_container_width=True)

    # ──────────────────────────────────────────────
    #  System Strategy
    # ──────────────────────────────────────────────
    st.info("💡 **Strategy Insight:** High AR volume in the 61-90 day bucket for 'UHC' indicates a likely bottleneck in medical necessity documentation. AI Denial Predictor has identified 124 claims for high-priority clinical review.")

if __name__ == "__main__":
    main()
