"""
5_Appeals_Analytics.py — Appeals Analytics Page
"""

import streamlit as st
import plotly.graph_objects as go
from utils_ui import shared_page_init, plotly_layout, COLORS, PALETTE, fmt_number, fmt_dollar
from src.core import ml_engine
from src.core.data_loader import get_cpt_summary

def page_appeals_analytics():
    master, _, _ = shared_page_init()
    
    st.markdown("# 📋 Appeals Analytics")
    st.caption("Track appeal filings, success rates, and revenue recovery")

    denied = master[master["is_denied"]].copy()
    appealed = master[master["is_appealed"]].copy()
    successful = appealed[appealed["appeal_success"]]

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total Denied", fmt_number(len(denied)))
    k2.metric("Appeals Filed", fmt_number(len(appealed)))
    k3.metric("Success Rate", f"{len(successful)/len(appealed)*100:.1f}%" if len(appealed) else "N/A")
    k4.metric("Recovered Revenue", fmt_dollar(successful["paid_amount"].sum()))

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        fig = go.Figure(go.Funnel(
            y=["Total Denied", "Appeals Filed", "Successful", "Revenue Recovered"],
            x=[len(denied), len(appealed), len(successful), int(successful["paid_amount"].sum())],
            textinfo="value+percent previous",
            marker=dict(color=[COLORS["danger"], COLORS["warning"], COLORS["success"], COLORS["accent"]]),
        ))
        plotly_layout(fig, "📊 Appeal Recovery Funnel", 400)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        appeal_by_reason = appealed.groupby("denial_reason").agg(total=("claim_id", "count"), success=("appeal_success", "sum")).reset_index()
        fig = go.Figure()
        fig.add_trace(go.Bar(x=appeal_by_reason["denial_reason"], y=appeal_by_reason["total"], name="Filed", marker_color=COLORS["warning"]))
        fig.add_trace(go.Bar(x=appeal_by_reason["denial_reason"], y=appeal_by_reason["success"], name="Successful", marker_color=COLORS["success"]))
        plotly_layout(fig, "📋 Appeals by Denial Reason", 400)
        st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    page_appeals_analytics()
