"""
4_Denial_Intelligence.py — Denial Intelligence Page
"""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from utils_ui import shared_page_init, plotly_layout, COLORS, PALETTE, fmt_number, fmt_dollar

def page_denial_intelligence():
    master, _, _ = shared_page_init()
    
    st.markdown("# 🚫 Denial Intelligence")
    st.caption("Deep-dive into denial patterns, root causes, and recovery opportunities")

    denied = master[master["is_denied"]].copy()

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total Denials", fmt_number(len(denied)))
    k2.metric("Denial Rate", f"{len(denied)/len(master)*100:.2f}%")
    k3.metric("Denied Amount", fmt_dollar(denied["claim_amount"].sum()))
    k4.metric("Avg Denied Claim", fmt_dollar(denied["claim_amount"].mean()))

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        reasons = denied["denial_reason"].value_counts().reset_index()
        reasons.columns = ["Reason", "Count"]
        fig = px.pie(reasons, values="Count", names="Reason", hole=0.55, color_discrete_sequence=PALETTE)
        plotly_layout(fig, "🔍 Denial Reasons Breakdown", 380)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        payer_denial = master.groupby("insurance").agg(total=("claim_id", "count"), denied=("is_denied", "sum")).reset_index()
        payer_denial["rate"] = (payer_denial["denied"] / payer_denial["total"] * 100).round(2)
        fig = go.Figure(go.Bar(y=payer_denial["insurance"], x=payer_denial["rate"], orientation="h", marker_color=PALETTE[:len(payer_denial)]))
        plotly_layout(fig, "📊 Denial Rate by Payer", 380)
        st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    page_denial_intelligence()
