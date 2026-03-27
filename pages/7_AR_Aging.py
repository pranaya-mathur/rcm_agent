"""
7_AR_Aging.py — AR Aging & Lifecycle Page
"""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from utils_ui import shared_page_init, plotly_layout, COLORS, fmt_number, fmt_dollar

def page_ar_aging():
    master, events_tl, raw = shared_page_init()
    
    st.markdown("# ⏱️ AR Aging & Claim Lifecycle")
    st.caption("Track claim processing times and identify bottlenecks")

    tl = events_tl.copy()

    k1, k2, k3 = st.columns(3)
    k1.metric("Avg Days to Submit", f"{tl['days_to_submit'].mean():.1f} days")
    k2.metric("Avg Days to Process", f"{tl['days_to_process'].mean():.1f} days")
    k3.metric("Avg Total Cycle", f"{tl['total_cycle_days'].mean():.1f} days")

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        bins = [0, 7, 14, 30, 60, 999]
        labels = ["0-7d", "7-14d", "14-30d", "30-60d", "60+d"]
        tl["aging_bucket"] = pd.cut(tl["total_cycle_days"], bins=bins, labels=labels, right=True)
        aging = tl["aging_bucket"].value_counts().reindex(labels).reset_index()
        aging.columns = ["Bucket", "Claims"]
        fig = go.Figure(go.Bar(x=aging["Bucket"], y=aging["Claims"], marker_color=COLORS["success"]))
        plotly_layout(fig, "📊 AR Aging Distribution", 400)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.histogram(tl, x="total_cycle_days", nbins=40, color_discrete_sequence=[COLORS["accent"]])
        plotly_layout(fig, "📈 Total Cycle Days Distribution", 400)
        st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    page_ar_aging()
