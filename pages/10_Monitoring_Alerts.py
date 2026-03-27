"""
10_Monitoring_Alerts.py — Monitoring & Alerts Page
"""

import streamlit as st
import pandas as pd
from utils_ui import shared_page_init, COLORS, fmt_number

def page_monitoring_alerts():
    master, _, _ = shared_page_init()
    
    st.markdown("# 🖥️ Monitoring & Alerts")
    st.caption("Continuous KPI watchlist for denial, fraud, and payment risk (near real-time simulation).")

    denial_rate = master["is_denied"].mean() * 100
    fraud_high = (master["fraud_score"] > 0.8).mean() * 100
    recon_backlog = ((master["claim_amount"] - master["paid_amount"]).clip(lower=0) > 50).sum()

    alerts = []
    if denial_rate > 8: alerts.append(("Denial rate above threshold", "danger"))
    if fraud_high > 3: alerts.append(("High fraud-risk pool increased", "warning"))
    if recon_backlog > 5000: alerts.append(("Payment reconciliation backlog high", "warning"))
    if not alerts: alerts.append(("All core KPIs within expected bands", "success"))

    for txt, lvl in alerts:
        if lvl == "danger": st.error(f"🚨 {txt}")
        elif lvl == "warning": st.warning(f"⚠️ {txt}")
        else: st.success(f"✅ {txt}")

    st.markdown("---")
    monitor_df = pd.DataFrame({
        "Metric": ["Denial Rate %", "Fraud High-Risk %", "Reconciliation Backlog"],
        "Current": [round(denial_rate, 2), round(fraud_high, 2), int(recon_backlog)],
        "Threshold": [8.0, 3.0, 5000],
        "Status": ["ALERT" if denial_rate > 8 else "OK", "ALERT" if fraud_high > 3 else "OK", "ALERT" if recon_backlog > 5000 else "OK"]
    })
    st.dataframe(monitor_df, use_container_width=True, hide_index=True)

if __name__ == "__main__":
    page_monitoring_alerts()
