"""
9_Revenue_Forecasting.py — Revenue Forecasting Page
"""

import streamlit as st
import plotly.graph_objects as go
import pandas as pd
from utils_ui import shared_page_init, render_page_header, plotly_layout, COLORS, fmt_number, fmt_dollar
from src.core import ml_engine

def page_revenue_forecasting():
    master, events_tl, _ = shared_page_init()
    
    st.markdown("# 📈 Revenue Forecasting & What-If")
    st.caption("Trend projection and denial-reduction scenario planning.")

    merged = master.merge(events_tl[["claim_id", "CREATED"]], on="claim_id", how="left")
    merged = merged.dropna(subset=["CREATED"]).copy()
    merged["month"] = merged["CREATED"].dt.to_period("M").astype(str)
    monthly = merged.groupby("month").agg(billed=("claim_amount", "sum"), collected=("paid_amount", "sum"), denied=("is_denied", "mean")).reset_index()
    monthly["denied"] = monthly["denied"] * 100

    forecast_res = ml_engine.fit_revenue_forecast_model(monthly, value_col="collected")
    forecast_collected = float(forecast_res["next_forecast"])

    k1, k2 = st.columns(2)
    k1.metric("Next-Month Revenue Forecast", fmt_dollar(forecast_collected))
    k2.metric("Avg Monthly Denial Rate", f"{monthly['denied'].mean():.1f}%")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=monthly["month"], y=monthly["billed"], name="Billed", line=dict(color=COLORS["primary"])))
    fig.add_trace(go.Scatter(x=monthly["month"], y=monthly["collected"], name="Collected", line=dict(color=COLORS["accent"])))
    plotly_layout(fig, "Monthly Revenue Trend", 340)
    st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    page_revenue_forecasting()
