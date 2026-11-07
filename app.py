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
from src.core.demo_db import get_live_stats

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
    live = get_live_stats()
    st.caption("Synthetic demo dataset — figures are prototype analytics, not production KPIs.")
    st.markdown(
        (
            "🟦 **Live DB Intake (Demo Mode):** "
            f"Persisted claims: **{live.get('count', 0)}**"
            + (
                f" | Last Claim ID: **{live.get('last_claim_id')}** | Updated: **{live.get('last_updated_at')}**"
                if live.get("last_claim_id") is not None
                else ""
            )
            + " *(banner reads SQLite directly; KPI tables below use cached merged data)*"
        )
    )

    # ──────────────────────────────────────────────
    #  Top KPIs (Custom Metric Cards)
    # ──────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        total_billed = master["claim_amount"].sum()
        outstanding = float((master["claim_amount"] - master["paid_amount"]).clip(lower=0).sum())
        metric_card("Total Billed (claims)", fmt_dollar(total_billed), "sum of claim_amount", delta_up=True)
    with c2:
        denial_rate = master["is_denied"].mean() * 100
        metric_card("Denial Rate", f"{denial_rate:.1f}%", "dataset rate (synthetic)", delta_up=False)
    with c3:
        clean_rate = master["is_clean_claim"].mean() * 100
        metric_card("Clean Claim Rate", f"{clean_rate:.1f}%", f"Outstanding gap {fmt_dollar(outstanding)}", delta_up=True)
    with c4:
        recovery = master[master["appeal_success"] == True]["paid_amount"].sum()
        metric_card("Appeal Recovery (paid)", fmt_dollar(recovery), "from successful appeal rows", delta_up=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ──────────────────────────────────────────────
    #  Main Grid: AR Aging & Denial Pareto
    # ──────────────────────────────────────────────
    col_left, col_right = st.columns([1.5, 1])

    with col_left:
        st.markdown("### 📊 Claim Cycle Duration (from events)")
        st.caption(
            "This chart uses processing cycle days from the events timeline — "
            "it is **not** true outstanding AR aging inventory."
        )
        if events_tl is not None and len(events_tl) and "total_cycle_days" in events_tl.columns:
            cycle = events_tl[["claim_id", "total_cycle_days"]].dropna().merge(
                master[["claim_id", "claim_amount"]], on="claim_id", how="inner"
            )
            bins = [-0.1, 30, 60, 90, 120, 10_000]
            labels = ["0-30 Days", "31-60 Days", "61-90 Days", "91-120 Days", "120+ Days"]
            cycle["Bucket"] = pd.cut(cycle["total_cycle_days"], bins=bins, labels=labels)
            aging_data = cycle.groupby("Bucket", observed=False)["claim_amount"].sum().reset_index()
            aging_data.columns = ["Bucket", "Amount"]
        else:
            aging_data = pd.DataFrame({"Bucket": ["Unavailable"], "Amount": [0.0]})
        fig_aging = px.bar(
            aging_data, x="Amount", y="Bucket", orientation="h",
            color="Bucket", color_discrete_sequence=px.colors.sequential.Blues_r,
            text_auto=True
        )
        plotly_layout(fig_aging, "Billed amount by claim cycle-duration bucket", 400)
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
    st.info(
        "💡 **Demo note:** Use Denial Predictor, Smart Scrubbing, Agentic Orchestrator, and RCM AI Copilot "
        "for claim-level walkthroughs. Insights on this page are descriptive analytics over synthetic data only."
    )

if __name__ == "__main__":
    main()
