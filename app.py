"""
app.py — AI-Powered Smart RCM Dashboard
========================================
A full-featured Streamlit dashboard for Revenue Cycle Management analytics.
Covers: Executive Summary, Denial Intelligence, Appeals Analytics,
        Fraud Detection, Smart Scrubbing, and AR Aging.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
from dotenv import load_dotenv

# Load environment variables from .env (if present).
load_dotenv()

from data_loader import load_all, build_master, build_events_timeline, get_cpt_summary
import ml_engine
from ml_engine import train_model, load_model, predict_single_claim
from rcm_agent import CoordinatorAgent
from groq_agent_summary import generate_two_page_agent_summary, markdown_to_html_page
from ollama_agent_summary import generate_two_page_agent_summary_ollama
from langgraph_rcm_chatbot import rcm_langgraph_app
from custom_coding_agent import build_cpt_icd_knowledge, build_coding_recommendation
from clinical_nlp_agent import (
    train_notes_to_icd_model,
    predict_icd_from_notes_batch,
    build_nlp_coding_recommendation,
)
from backend_api import build_realtime_agent_payload

# ──────────────────────────────────────────────
#  Page config & CSS
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="AI-Powered Smart RCM Dashboard",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

def load_css():
    css_path = os.path.join(os.path.dirname(__file__), "style.css")
    with open(css_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css()

# ──────────────────────────────────────────────
#  Plotly theme helper
# ──────────────────────────────────────────────
COLORS = {
    "primary": "#6C63FF",
    "secondary": "#A78BFA",
    "accent": "#22D3EE",
    "success": "#22C55E",
    "warning": "#F59E0B",
    "danger": "#EF4444",
    "bg": "#0B0F19",
    "card": "#1E293B",
    "text": "#E2E8F0",
    "muted": "#94A3B8",
}

PALETTE = ["#6C63FF", "#22D3EE", "#A78BFA", "#F59E0B", "#22C55E", "#EF4444", "#EC4899", "#3B82F6"]

def plotly_layout(fig, title="", height=400):
    fig.update_layout(
        title=dict(text=title, font=dict(size=16, color=COLORS["text"], family="Inter")),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter", color=COLORS["muted"], size=12),
        height=height,
        margin=dict(l=40, r=20, t=50, b=40),
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            font=dict(color=COLORS["muted"], size=11),
        ),
        xaxis=dict(gridcolor="rgba(108,99,255,0.08)", zerolinecolor="rgba(108,99,255,0.08)"),
        yaxis=dict(gridcolor="rgba(108,99,255,0.08)", zerolinecolor="rgba(108,99,255,0.08)"),
    )
    return fig


def fmt_dollar(val):
    if val >= 1_000_000:
        return f"${val / 1_000_000:.2f}M"
    elif val >= 1_000:
        return f"${val / 1_000:.1f}K"
    return f"${val:,.0f}"


def fmt_number(val):
    if val >= 1_000_000:
        return f"{val / 1_000_000:.2f}M"
    elif val >= 1_000:
        return f"{val / 1_000:.1f}K"
    return f"{val:,}"

# ──────────────────────────────────────────────
#  Load data
# ──────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def get_data():
    master = build_master()
    events = build_events_timeline()
    raw = load_all()
    return master, events, raw

with st.spinner("🔄 Loading RCM Data..."):
    master, events_tl, raw = get_data()

# ──────────────────────────────────────────────
#  Sidebar navigation
# ──────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding: 1rem 0 0.5rem 0;">
        <span style="font-size:2.2rem;">🏥</span>
        <h2 style="margin:0.2rem 0; font-size:1.2rem; border:none; padding:0;
            background: linear-gradient(135deg, #6C63FF, #A78BFA);
            -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
            Smart RCM
        </h2>
        <p style="color:#64748B; font-size:0.75rem; margin:0;">AI-Powered Revenue Cycle</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    page = st.radio(
        "📊 Navigation",
        [
            "🏠 Executive Summary",
            "🧾 Patient Access & Eligibility",
            "🚫 Denial Intelligence",
            "📋 Appeals Analytics",
            "🔍 Fraud Detection",
            "🧹 Smart Scrubbing",
            "💳 Payment Reconciliation",
            "📈 Revenue Forecasting",
            "🖥️ Monitoring & Alerts",
            "⏱️ AR Aging & Lifecycle",
            "🤖 Agentic RCM Agent",
            "💬 LangGraph Chatbot",
            "🧠 AI Denial Predictor",
        ],
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown(f"""
    <div style="padding:0.8rem; background:rgba(108,99,255,0.06); border-radius:12px;
                border:1px solid rgba(108,99,255,0.12); font-size:0.75rem; color:#94A3B8;">
        <div><span class="pulse-dot"></span> <strong style="color:#E2E8F0;">System Active</strong></div>
        <div style="margin-top:6px;">📄 {fmt_number(len(master))} claims loaded</div>
        <div>🏷️ {master['insurance'].nunique()} payers tracked</div>
        <div>👤 {fmt_number(master['patient_id'].nunique())} patients</div>
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════
#  PAGE 1 — EXECUTIVE SUMMARY
# ══════════════════════════════════════════════
def page_executive_summary():
    st.markdown("# 🏠 Executive Summary")
    st.caption("End-to-end Revenue Cycle performance at a glance")

    # ── KPI row ──
    total_billed = master["claim_amount"].sum()
    total_collected = master["paid_amount"].sum()
    revenue_leakage = total_billed - total_collected
    denial_rate = master["is_denied"].mean() * 100
    clean_claim_rate = master["is_clean_claim"].mean() * 100
    fraud_rate = master["fraud_flag"].mean() * 100
    avg_collection = (total_collected / total_billed) * 100

    k1, k2, k3, k4, k5, k6 = st.columns(6)
    k1.metric("Total Claims", fmt_number(len(master)))
    k2.metric("Total Billed", fmt_dollar(total_billed))
    k3.metric("Total Collected", fmt_dollar(total_collected))
    k4.metric("Revenue Leakage", fmt_dollar(revenue_leakage), delta=f"-{revenue_leakage/total_billed*100:.1f}%", delta_color="inverse")
    k5.metric("Denial Rate", f"{denial_rate:.1f}%", delta="Target < 5%", delta_color="inverse")
    k6.metric("Collection Rate", f"{avg_collection:.1f}%")

    st.markdown("")

    k7, k8, k9 = st.columns(3)
    k7.metric("Clean Claim Rate", f"{clean_claim_rate:.1f}%", delta="Target 95%+")
    k8.metric("Fraud Flagged", f"{fraud_rate:.1f}%", delta=f"{master['fraud_flag'].sum():,} claims", delta_color="inverse")
    k9.metric("Appeals Success", f"{master[master['is_appealed']]['appeal_success'].mean()*100:.0f}%")

    st.markdown("---")

    # ── Row 2: Charts ──
    col1, col2 = st.columns(2)

    with col1:
        payer = master.groupby("insurance").agg(
            billed=("claim_amount", "sum"),
            collected=("paid_amount", "sum"),
        ).reset_index().sort_values("billed", ascending=True)

        fig = go.Figure()
        fig.add_trace(go.Bar(y=payer["insurance"], x=payer["billed"], name="Billed",
                             orientation="h", marker_color=COLORS["primary"], opacity=0.85))
        fig.add_trace(go.Bar(y=payer["insurance"], x=payer["collected"], name="Collected",
                             orientation="h", marker_color=COLORS["accent"], opacity=0.85))
        plotly_layout(fig, "💰 Revenue by Payer", 350)
        fig.update_layout(barmode="group", yaxis_title="", xaxis_title="Amount ($)")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Denial funnel
        total = len(master)
        denied = master["is_denied"].sum()
        appealed = master["is_appealed"].sum()
        recovered = master[(master["is_appealed"]) & (master["appeal_success"])].shape[0]

        fig = go.Figure(go.Funnel(
            y=["Total Claims", "Denied", "Appealed", "Recovered"],
            x=[total, denied, appealed, recovered],
            textinfo="value+percent initial",
            marker=dict(color=[COLORS["primary"], COLORS["danger"], COLORS["warning"], COLORS["success"]]),
            connector=dict(line=dict(color=COLORS["muted"], width=1)),
        ))
        plotly_layout(fig, "📉 Denial → Appeals → Recovery Funnel", 350)
        st.plotly_chart(fig, use_container_width=True)

    # ── Row 3: Monthly trend + visit type ──
    col3, col4 = st.columns(2)

    with col3:
        merged = master.merge(events_tl[["claim_id", "CREATED"]], on="claim_id", how="left")
        merged["month"] = merged["CREATED"].dt.to_period("M").astype(str)
        monthly = merged.groupby("month").agg(
            claims=("claim_id", "count"),
            billed=("claim_amount", "sum"),
        ).reset_index()

        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Bar(x=monthly["month"], y=monthly["claims"], name="Claims",
                             marker_color=COLORS["primary"], opacity=0.7), secondary_y=False)
        fig.add_trace(go.Scatter(x=monthly["month"], y=monthly["billed"], name="Billed ($)",
                                 line=dict(color=COLORS["accent"], width=2.5), mode="lines+markers"),
                      secondary_y=True)
        plotly_layout(fig, "📅 Monthly Claim Volume & Billing", 350)
        fig.update_yaxes(title_text="Claims", secondary_y=False)
        fig.update_yaxes(title_text="Billed ($)", secondary_y=True)
        st.plotly_chart(fig, use_container_width=True)

    with col4:
        visit = master.groupby("visit_type").agg(
            count=("claim_id", "count"),
            avg_amount=("claim_amount", "mean"),
            denial_rate=("is_denied", "mean"),
        ).reset_index()
        visit["denial_rate"] = (visit["denial_rate"] * 100).round(1)

        fig = go.Figure()
        fig.add_trace(go.Bar(x=visit["visit_type"], y=visit["count"], name="Claims",
                             marker_color=COLORS["primary"], opacity=0.8, yaxis="y"))
        fig.add_trace(go.Scatter(x=visit["visit_type"], y=visit["denial_rate"], name="Denial %",
                                 mode="lines+markers+text", text=visit["denial_rate"].astype(str)+"%",
                                 textposition="top center",
                                 line=dict(color=COLORS["danger"], width=2.5), yaxis="y2"))
        plotly_layout(fig, "🏥 Claims & Denials by Visit Type", 350)
        fig.update_layout(
            yaxis=dict(title="Claims Count"),
            yaxis2=dict(title="Denial Rate %", overlaying="y", side="right",
                        range=[0, visit["denial_rate"].max() * 1.5]),
        )
        st.plotly_chart(fig, use_container_width=True)

    # ── AI Insights ──
    st.markdown("---")
    st.markdown("### 🤖 AI-Powered Insights")
    ic1, ic2, ic3 = st.columns(3)
    with ic1:
        top_denial = master[master["is_denied"]]["denial_reason"].value_counts().idxmax()
        top_denial_count = master[master["is_denied"]]["denial_reason"].value_counts().max()
        top_denial_amount = master[master["is_denied"] & (master["denial_reason"] == top_denial)]["claim_amount"].sum()
        st.info(f"🎯 **Top Denial Reason:** {top_denial} ({top_denial_count:,} claims)\n\n"
                f"Fixing this could recover up to **{fmt_dollar(top_denial_amount)}** in revenue.")
    with ic2:
        worst_payer = master.groupby("insurance")["is_denied"].mean().idxmax()
        worst_rate = master.groupby("insurance")["is_denied"].mean().max() * 100
        st.warning(f"⚠️ **Highest Denial Payer:** {worst_payer} ({worst_rate:.1f}% denial rate)\n\n"
                   f"Consider contract renegotiation or enhanced pre-auth workflows.")
    with ic3:
        projected_gain = revenue_leakage * 0.20
        st.success(f"💡 **AI Projection:** Implementing smart scrubbing + denial prediction could reduce "
                   f"revenue leakage by 20%, recovering **{fmt_dollar(projected_gain)}**.")


# ══════════════════════════════════════════════
#  PAGE 2 — DENIAL INTELLIGENCE
# ══════════════════════════════════════════════
def page_denial_intelligence():
    st.markdown("# 🚫 Denial Intelligence")
    st.caption("Deep-dive into denial patterns, root causes, and recovery opportunities")

    denied = master[master["is_denied"]].copy()

    # KPIs
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
        fig = px.pie(reasons, values="Count", names="Reason", hole=0.55,
                     color_discrete_sequence=PALETTE)
        plotly_layout(fig, "🔍 Denial Reasons Breakdown", 380)
        fig.update_traces(textinfo="percent+label", textfont_size=12)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        payer_denial = master.groupby("insurance").agg(
            total=("claim_id", "count"),
            denied=("is_denied", "sum")
        ).reset_index()
        payer_denial["rate"] = (payer_denial["denied"] / payer_denial["total"] * 100).round(2)
        payer_denial = payer_denial.sort_values("rate", ascending=True)

        fig = go.Figure(go.Bar(
            y=payer_denial["insurance"], x=payer_denial["rate"],
            orientation="h", marker_color=PALETTE[:len(payer_denial)],
            text=payer_denial["rate"].astype(str) + "%", textposition="outside"
        ))
        plotly_layout(fig, "📊 Denial Rate by Payer", 380)
        fig.update_xaxes(title="Denial Rate %")
        st.plotly_chart(fig, use_container_width=True)

    col3, col4 = st.columns(2)

    with col3:
        visit_denial = master.groupby("visit_type").agg(
            total=("claim_id", "count"),
            denied=("is_denied", "sum"),
            denied_amount=("claim_amount", lambda x: x[master.loc[x.index, "is_denied"]].sum())
        ).reset_index()
        visit_denial["rate"] = (visit_denial["denied"] / visit_denial["total"] * 100).round(2)

        fig = go.Figure()
        fig.add_trace(go.Bar(x=visit_denial["visit_type"], y=visit_denial["denied"],
                             name="Denied Claims", marker_color=COLORS["danger"], opacity=0.8))
        fig.add_trace(go.Bar(x=visit_denial["visit_type"], y=visit_denial["total"] - visit_denial["denied"],
                             name="Approved Claims", marker_color=COLORS["success"], opacity=0.6))
        plotly_layout(fig, "🏥 Denials by Visit Type", 380)
        fig.update_layout(barmode="stack")
        st.plotly_chart(fig, use_container_width=True)

    with col4:
        age_denial = master.groupby("age_group", observed=True).agg(
            total=("claim_id", "count"),
            denied=("is_denied", "sum")
        ).reset_index()
        age_denial["rate"] = (age_denial["denied"] / age_denial["total"] * 100).round(2)

        fig = go.Figure()
        fig.add_trace(go.Bar(x=age_denial["age_group"].astype(str), y=age_denial["rate"],
                             marker=dict(color=age_denial["rate"], colorscale="Reds"),
                             text=age_denial["rate"].astype(str) + "%", textposition="outside"))
        plotly_layout(fig, "👥 Denial Rate by Age Group", 380)
        fig.update_yaxes(title="Denial Rate %")
        st.plotly_chart(fig, use_container_width=True)

    # Top ICD codes causing denials
    st.markdown("---")
    st.markdown("### 🧬 Top ICD Codes Causing Denials")
    icd_denial = denied.groupby("icd_code").agg(
        count=("claim_id", "count"),
        total_amount=("claim_amount", "sum")
    ).reset_index().sort_values("count", ascending=False).head(10)

    fig = go.Figure(go.Bar(
        x=icd_denial["icd_code"], y=icd_denial["count"],
        marker=dict(color=icd_denial["total_amount"], colorscale="Viridis", showscale=True,
                    colorbar=dict(title="$ Amount")),
        text=icd_denial["count"], textposition="outside"
    ))
    plotly_layout(fig, "", 350)
    fig.update_xaxes(title="ICD Code")
    fig.update_yaxes(title="Denied Claims")
    st.plotly_chart(fig, use_container_width=True)

    # AI insight
    st.markdown("### 🤖 AI Recommendation")
    reason_amounts = denied.groupby("denial_reason")["claim_amount"].sum().sort_values(ascending=False)
    top_r = reason_amounts.index[0]
    top_a = reason_amounts.iloc[0]
    st.info(f"📌 **Focus Area:** \"{top_r}\" denials account for **{fmt_dollar(top_a)}** in lost revenue.\n\n"
            f"AI recommends implementing automated pre-authorization checks to reduce this category by up to 60%.")


# ══════════════════════════════════════════════
#  PAGE 3 — APPEALS ANALYTICS
# ══════════════════════════════════════════════
def page_appeals_analytics():
    st.markdown("# 📋 Appeals Analytics")
    st.caption("Track appeal filings, success rates, and revenue recovery")

    denied = master[master["is_denied"]].copy()
    appealed = master[master["is_appealed"]].copy()
    successful = appealed[appealed["appeal_success"]]

    # KPIs
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total Denied", fmt_number(len(denied)))
    k2.metric("Appeals Filed", fmt_number(len(appealed)), delta=f"{len(appealed)/len(denied)*100:.0f}% of denials")
    k3.metric("Success Rate", f"{len(successful)/len(appealed)*100:.1f}%" if len(appealed) else "N/A")
    k4.metric("Recovered Revenue", fmt_dollar(successful["paid_amount"].sum()))

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        # Appeal funnel
        fig = go.Figure(go.Funnel(
            y=["Total Denied", "Appeals Filed", "Successful", "Revenue Recovered"],
            x=[len(denied), len(appealed), len(successful),
               int(successful["paid_amount"].sum())],
            textinfo="value+percent previous",
            marker=dict(color=[COLORS["danger"], COLORS["warning"], COLORS["success"], COLORS["accent"]]),
        ))
        plotly_layout(fig, "📊 Appeal Recovery Funnel", 400)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Appeals by denial reason
        appeal_by_reason = appealed.groupby("denial_reason").agg(
            total=("claim_id", "count"),
            success=("appeal_success", "sum")
        ).reset_index()
        appeal_by_reason["success_rate"] = (appeal_by_reason["success"] / appeal_by_reason["total"] * 100).round(1)

        fig = go.Figure()
        fig.add_trace(go.Bar(x=appeal_by_reason["denial_reason"], y=appeal_by_reason["total"],
                             name="Filed", marker_color=COLORS["warning"], opacity=0.8))
        fig.add_trace(go.Bar(x=appeal_by_reason["denial_reason"], y=appeal_by_reason["success"],
                             name="Successful", marker_color=COLORS["success"], opacity=0.8))
        plotly_layout(fig, "📋 Appeals by Denial Reason", 400)
        fig.update_layout(barmode="group")
        st.plotly_chart(fig, use_container_width=True)

    # Recovery by payer
    st.markdown("---")
    col3, col4 = st.columns(2)

    with col3:
        payer_appeal = appealed.groupby("insurance").agg(
            filed=("claim_id", "count"),
            success=("appeal_success", "sum"),
            recovered=("paid_amount", "sum")
        ).reset_index()
        payer_appeal["rate"] = (payer_appeal["success"] / payer_appeal["filed"] * 100).round(1)

        fig = go.Figure(go.Bar(
            x=payer_appeal["insurance"], y=payer_appeal["recovered"],
            marker_color=PALETTE[:len(payer_appeal)],
            text=payer_appeal["rate"].astype(str) + "% success", textposition="outside"
        ))
        plotly_layout(fig, "💰 Recovery Amount by Payer", 380)
        fig.update_yaxes(title="Recovered ($)")
        st.plotly_chart(fig, use_container_width=True)

    with col4:
        # Not-appealed denials (opportunity)
        not_appealed = denied[~denied["is_appealed"]]
        opportunity = not_appealed["claim_amount"].sum()

        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=len(appealed) / len(denied) * 100 if len(denied) else 0,
            title={"text": "Appeal Filing Rate", "font": {"size": 16, "color": COLORS["text"]}},
            number={"suffix": "%", "font": {"size": 40, "color": COLORS["accent"]}},
            delta={"reference": 80, "suffix": "%"},
            gauge=dict(
                axis=dict(range=[0, 100], tickcolor=COLORS["muted"]),
                bar=dict(color=COLORS["accent"]),
                bgcolor=COLORS["card"],
                steps=[
                    dict(range=[0, 50], color="rgba(239,68,68,0.2)"),
                    dict(range=[50, 75], color="rgba(245,158,11,0.2)"),
                    dict(range=[75, 100], color="rgba(34,197,94,0.2)"),
                ],
            )
        ))
        plotly_layout(fig, "", 380)
        st.plotly_chart(fig, use_container_width=True)

        st.warning(f"⚠️ **{fmt_number(len(not_appealed))} denied claims** were never appealed, "
                   f"representing **{fmt_dollar(opportunity)}** in potential recovery.")

    # ──────────────────────────────────────────────
    # Phase 4 — Automated Appeals Prioritization (ML)
    # ──────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 🤖 Phase 4 — Automated Appeals Prioritization")

    @st.cache_resource(show_spinner=False)
    def cached_appeals_success_model():
        model, meta = ml_engine.load_appeals_success_model()
        if model is None:
            with st.spinner("🚀 Training appeals success model..."):
                cpt_summary = get_cpt_summary()
                res = ml_engine.train_appeals_success_model(master, cpt_summary)
                return res["model"], res["meta"]
        return model, meta

    @st.cache_resource(show_spinner=False)
    def cached_appeals_recovery_model():
        model, meta = ml_engine.load_appeals_recovery_model()
        if model is None:
            with st.spinner("🚀 Training appeals recovery model..."):
                cpt_summary = get_cpt_summary()
                res = ml_engine.train_appeals_recovery_model(master, cpt_summary)
                return res["model"], res["meta"]
        return model, meta

    appeals_success_model, appeals_success_meta = cached_appeals_success_model()
    appeals_recovery_model, appeals_recovery_meta = cached_appeals_recovery_model()

    cpt_summary = get_cpt_summary()
    candidates = not_appealed.copy()
    ranked = ml_engine.predict_appeals_ranked_candidates(
        candidates,
        cpt_summary,
        appeals_success_model,
        appeals_success_meta["feature_cols"],
        appeals_recovery_model,
        appeals_recovery_meta["feature_cols"],
    )

    ranked = ranked.sort_values("expected_recovery", ascending=False)
    top_appeals = ranked.head(15)[
        ["claim_id", "insurance", "claim_amount", "visit_type", "icd_code", "denial_reason", "p_appeal_success", "expected_recovery"]
    ]

    st.dataframe(top_appeals, use_container_width=True, hide_index=True)
    st.info(
        f"Estimated expected recovery from all unappealed denials: "
        f"**{fmt_dollar(ranked['expected_recovery'].sum())}**"
    )


# ══════════════════════════════════════════════
#  PAGE 4 — FRAUD DETECTION
# ══════════════════════════════════════════════
def page_fraud_detection():
    st.markdown("# 🔍 Fraud Detection")
    st.caption("AI-powered fraud scoring, risk analysis, and anomaly detection")

    flagged = master[master["fraud_flag"] == True]

    @st.cache_resource(show_spinner=False)
    def cached_fraud_probability_model():
        model, meta = ml_engine.load_fraud_probability_model()
        if model is None:
            with st.spinner("🚀 Training fraud probability (supervised) model..."):
                cpt_summary = get_cpt_summary()
                res = ml_engine.train_fraud_probability_model(master, cpt_summary)
                return res["model"], res["meta"]
        return model, meta

    @st.cache_resource(show_spinner=False)
    def cached_fraud_anomaly_model():
        model, meta = ml_engine.load_fraud_anomaly_model()
        if model is None:
            with st.spinner("🚀 Training fraud anomaly (IsolationForest) model..."):
                res = ml_engine.train_fraud_anomaly_model(master)
                return res["model"], res["meta"]
        return model, meta

    fraud_prob_model, fraud_prob_meta = cached_fraud_probability_model()
    fraud_anomaly_model, fraud_anomaly_meta = cached_fraud_anomaly_model()
    cpt_summary = get_cpt_summary()
    fraud_enhanced = ml_engine.score_fraud_enhanced(
        master,
        cpt_summary,
        fraud_prob_model,
        fraud_prob_meta["feature_cols"],
        fraud_anomaly_model,
        fraud_anomaly_meta,
        alpha=0.6,
    )

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total Claims Analyzed", fmt_number(len(master)))
    k2.metric("Fraud Flagged", fmt_number(len(flagged)), delta=f"{len(flagged)/len(master)*100:.1f}%", delta_color="inverse")
    k3.metric("Flagged Amount", fmt_dollar(flagged["claim_amount"].sum()))
    k4.metric("Avg Fraud Prob (Improved)", f"{fraud_enhanced['fraud_probability_improved'].mean():.3f}")

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        fig = px.histogram(fraud_enhanced, x="fraud_probability_improved", nbins=50,
                           color_discrete_sequence=[COLORS["primary"]])
        plotly_layout(fig, "📊 Fraud Probability (Improved) Distribution", 380)
        fig.update_xaxes(title="Fraud Probability (Improved)")
        fig.update_yaxes(title="Claims Count")
        # Add threshold line
        fig.add_vline(x=0.7, line_dash="dash", line_color=COLORS["danger"],
                      annotation_text="High Risk Threshold (0.7)", annotation_font_color=COLORS["danger"])
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        payer_fraud = master.groupby("insurance").agg(
            total=("claim_id", "count"),
            flagged=("fraud_flag", "sum"),
            avg_score=("fraud_score", "mean")
        ).reset_index()
        payer_fraud["rate"] = (payer_fraud["flagged"] / payer_fraud["total"] * 100).round(2)
        payer_fraud = payer_fraud.sort_values("rate", ascending=True)

        fig = go.Figure(go.Bar(
            y=payer_fraud["insurance"], x=payer_fraud["rate"],
            orientation="h", marker_color=PALETTE[:len(payer_fraud)],
            text=payer_fraud["rate"].astype(str) + "%", textposition="outside"
        ))
        plotly_layout(fig, "🏢 Fraud Rate by Payer", 380)
        fig.update_xaxes(title="Fraud Rate %")
        st.plotly_chart(fig, use_container_width=True)

    col3, col4 = st.columns(2)

    with col3:
        visit_fraud = master.groupby("visit_type").agg(
            total=("claim_id", "count"),
            flagged=("fraud_flag", "sum"),
            avg_score=("fraud_score", "mean")
        ).reset_index()
        visit_fraud["rate"] = (visit_fraud["flagged"] / visit_fraud["total"] * 100).round(1)

        fig = go.Figure()
        fig.add_trace(go.Bar(x=visit_fraud["visit_type"], y=visit_fraud["rate"],
                             name="Fraud Rate %", marker_color=COLORS["danger"], opacity=0.8))
        fig.add_trace(go.Scatter(x=visit_fraud["visit_type"], y=visit_fraud["avg_score"] * 100,
                                 name="Avg Score × 100", mode="lines+markers",
                                 line=dict(color=COLORS["accent"], width=2.5)))
        plotly_layout(fig, "🏥 Fraud by Visit Type", 380)
        st.plotly_chart(fig, use_container_width=True)

    with col4:
        # Fraud vs Scrubbing correlation
        cross = master.groupby(["fraud_flag", "cpt_icd_mismatch"]).size().reset_index(name="count")
        cross["fraud_flag"] = cross["fraud_flag"].map({True: "Fraud Flagged", False: "Clean"})
        cross["cpt_icd_mismatch"] = cross["cpt_icd_mismatch"].map({True: "CPT-ICD Mismatch", False: "No Mismatch"})

        fig = px.sunburst(cross, path=["fraud_flag", "cpt_icd_mismatch"], values="count",
                          color_discrete_sequence=PALETTE)
        plotly_layout(fig, "🔗 Fraud × Scrubbing Overlap", 380)
        st.plotly_chart(fig, use_container_width=True)

    # High-risk claims table
    st.markdown("---")
    st.markdown("### 🚨 High-Risk Claims (Fraud Probability (Improved) > 0.7)")
    high_risk = fraud_enhanced[fraud_enhanced["fraud_probability_improved"] > 0.7][
        [
            "claim_id",
            "insurance",
            "claim_amount",
            "paid_amount",
            "fraud_score",
            "fraud_probability_improved",
            "visit_type",
            "icd_code",
        ]
    ].sort_values("fraud_probability_improved", ascending=False).head(20)
    st.dataframe(high_risk, use_container_width=True, hide_index=True)

    st.info(
        f"💡 **AI Insight:** {len(fraud_enhanced[fraud_enhanced['fraud_probability_improved']>0.7]):,} claims have "
        f"fraud probability (improved) above 0.7, totaling "
        f"**{fmt_dollar(fraud_enhanced[fraud_enhanced['fraud_probability_improved']>0.7]['claim_amount'].sum())}**. "
        f"Priority review recommended for these claims."
    )


# ══════════════════════════════════════════════
#  PAGE 5 — SMART SCRUBBING
# ══════════════════════════════════════════════
def page_scrubbing():
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
        # Clean vs flagged gauge
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
        # Flag breakdown
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
        fig.update_yaxes(title="Flagged Claims")
        st.plotly_chart(fig, use_container_width=True)

    # ──────────────────────────────────────────────
    # Phase 3 — Smart Claim Scrubbing (ML)
    # ──────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 🤖 Phase 3 — ML CPT-ICD Mismatch Detection")

    @st.cache_resource(show_spinner=False)
    def cached_denial_predictor():
        model, meta = ml_engine.load_model()
        if model is None:
            with st.spinner("🚀 Training denial predictor (for recommendations)..."):
                cpt_summary = get_cpt_summary()
                res = ml_engine.train_model(master, cpt_summary)
                meta = {"feature_cols": res["feature_cols"], "all_probabilities": res["all_probabilities"]}
                return res["model"], meta
        return model, meta

    @st.cache_resource(show_spinner=False)
    def cached_mismatch_predictor():
        model, meta = ml_engine.load_mismatch_model()
        if model is None:
            with st.spinner("🚀 Training CPT-ICD mismatch detector..."):
                cpt_summary = get_cpt_summary()
                res = ml_engine.train_mismatch_model(master, cpt_summary)
                return res["model"], res["meta"]
        return model, meta

    mismatch_model, mismatch_meta = cached_mismatch_predictor()
    denial_model, denial_meta = cached_denial_predictor()

    cpt_summary = get_cpt_summary()
    mismatch_probs = ml_engine.score_all_mismatch(master, cpt_summary, mismatch_model, mismatch_meta["feature_cols"])
    denial_probs = denial_meta.get("all_probabilities", np.zeros(len(master), dtype=float))

    ml_high_threshold = 0.7
    ml_high_rate = (mismatch_probs >= ml_high_threshold).mean() * 100
    st.metric("ML High-Risk CPT-ICD Mismatch Rate", f"{ml_high_rate:.1f}%")

    phase3_df = master[["claim_id", "insurance", "claim_amount", "visit_type", "icd_code", "cpt_icd_mismatch"]].copy()
    phase3_df["mismatch_probability"] = mismatch_probs
    phase3_df["denial_probability"] = denial_probs

    phase3_df["recommendation"] = np.where(
        (phase3_df["denial_probability"] >= 0.7) | (phase3_df["mismatch_probability"] >= ml_high_threshold),
        "Is claim mein Auth required add kar do",
        "Standard scrubbing + coding review",
    )
    high_phase3 = phase3_df.sort_values("mismatch_probability", ascending=False).head(20)
    st.dataframe(
        high_phase3[[
            "claim_id", "insurance", "claim_amount", "visit_type", "icd_code",
            "cpt_icd_mismatch", "mismatch_probability", "denial_probability", "recommendation"
        ]],
        use_container_width=True,
        hide_index=True,
    )

    col3, col4 = st.columns(2)

    with col3:
        # Scrubbing by payer
        payer_scrub = master.groupby("insurance").agg(
            clean=("is_clean_claim", "mean"),
            mismatch=("cpt_icd_mismatch", "mean"),
        ).reset_index()
        payer_scrub["clean"] = (payer_scrub["clean"] * 100).round(1)
        payer_scrub["mismatch"] = (payer_scrub["mismatch"] * 100).round(1)

        fig = go.Figure()
        fig.add_trace(go.Bar(x=payer_scrub["insurance"], y=payer_scrub["clean"],
                             name="Clean Rate %", marker_color=COLORS["success"], opacity=0.8))
        fig.add_trace(go.Bar(x=payer_scrub["insurance"], y=payer_scrub["mismatch"],
                             name="Mismatch Rate %", marker_color=COLORS["danger"], opacity=0.8))
        plotly_layout(fig, "🏢 Clean Claim Rate by Payer", 380)
        fig.update_layout(barmode="group")
        st.plotly_chart(fig, use_container_width=True)

    with col4:
        # Flag combinations — how many claims have multiple flags
        master["flag_count"] = master[["cpt_icd_mismatch", "high_amount_flag", "strict_insurance_flag"]].sum(axis=1)
        combo = master["flag_count"].value_counts().sort_index().reset_index()
        combo.columns = ["Flags", "Claims"]
        combo["Flags"] = combo["Flags"].astype(int).astype(str) + " flags"

        fig = px.pie(combo, values="Claims", names="Flags", hole=0.5,
                     color_discrete_sequence=[COLORS["success"], COLORS["warning"], COLORS["danger"], "#EC4899"])
        plotly_layout(fig, "🔢 Flag Count Distribution", 380)
        fig.update_traces(textinfo="percent+label")
        st.plotly_chart(fig, use_container_width=True)

    # Scrubbing vs Denial correlation
    st.markdown("---")
    st.markdown("### 🔗 Scrubbing Impact on Denials")
    scrub_denial = master.groupby("is_clean_claim").agg(
        claims=("claim_id", "count"),
        denial_rate=("is_denied", "mean"),
        avg_amount=("claim_amount", "mean"),
    ).reset_index()
    scrub_denial["denial_rate"] = (scrub_denial["denial_rate"] * 100).round(2)
    scrub_denial["is_clean_claim"] = scrub_denial["is_clean_claim"].map({True: "✅ Clean Claims", False: "🚩 Flagged Claims"})
    scrub_denial.columns = ["Claim Type", "Total Claims", "Denial Rate %", "Avg Claim Amount"]

    col5, col6 = st.columns([1, 2])
    with col5:
        st.dataframe(scrub_denial, use_container_width=True, hide_index=True)
    with col6:
        flagged_denial = master[~master["is_clean_claim"]]["is_denied"].mean() * 100
        clean_denial = master[master["is_clean_claim"]]["is_denied"].mean() * 100
        st.success(f"💡 **AI Finding:** Clean claims have a **{clean_denial:.1f}%** denial rate vs "
                   f"**{flagged_denial:.1f}%** for flagged claims.\n\n"
                   f"Improving your scrubbing rules could reduce overall denials significantly.")


# ══════════════════════════════════════════════
#  PAGE 6 — AR AGING & LIFECYCLE
# ══════════════════════════════════════════════
def page_ar_aging():
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
        # Aging buckets
        bins = [0, 7, 14, 30, 60, 999]
        labels = ["0-7d", "7-14d", "14-30d", "30-60d", "60+d"]
        tl["aging_bucket"] = pd.cut(tl["total_cycle_days"], bins=bins, labels=labels, right=True)
        aging = tl["aging_bucket"].value_counts().reindex(labels).reset_index()
        aging.columns = ["Bucket", "Claims"]

        colors_aging = [COLORS["success"], COLORS["accent"], COLORS["warning"], "#F97316", COLORS["danger"]]
        fig = go.Figure(go.Bar(
            x=aging["Bucket"], y=aging["Claims"],
            marker_color=colors_aging,
            text=aging["Claims"].apply(fmt_number), textposition="outside"
        ))
        plotly_layout(fig, "📊 AR Aging Distribution", 400)
        fig.update_yaxes(title="Claims")
        fig.update_xaxes(title="Days in Cycle")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Processing time distribution
        fig = px.histogram(tl, x="total_cycle_days", nbins=40,
                           color_discrete_sequence=[COLORS["accent"]])
        plotly_layout(fig, "📈 Total Cycle Days Distribution", 400)
        fig.update_xaxes(title="Days")
        fig.update_yaxes(title="Claims")
        fig.add_vline(x=tl["total_cycle_days"].mean(), line_dash="dash",
                      line_color=COLORS["warning"],
                      annotation_text=f"Avg: {tl['total_cycle_days'].mean():.1f}d",
                      annotation_font_color=COLORS["warning"])
        st.plotly_chart(fig, use_container_width=True)

    col3, col4 = st.columns(2)

    with col3:
        # Payer-wise processing
        merged_payer = tl.merge(master[["claim_id", "insurance"]].drop_duplicates(), on="claim_id", how="left")
        payer_time = merged_payer.groupby("insurance").agg(
            submit=("days_to_submit", "mean"),
            process=("days_to_process", "mean"),
            total=("total_cycle_days", "mean")
        ).reset_index().sort_values("total", ascending=True)

        fig = go.Figure()
        fig.add_trace(go.Bar(x=payer_time["insurance"], y=payer_time["submit"],
                             name="Submit Time", marker_color=COLORS["primary"], opacity=0.8))
        fig.add_trace(go.Bar(x=payer_time["insurance"], y=payer_time["process"],
                             name="Process Time", marker_color=COLORS["accent"], opacity=0.8))
        plotly_layout(fig, "🏢 Avg Processing Time by Payer", 400)
        fig.update_layout(barmode="stack")
        fig.update_yaxes(title="Days")
        st.plotly_chart(fig, use_container_width=True)

    with col4:
        # Lifecycle Sankey
        events_raw = raw["events"]
        event_pairs = []
        for claim_id, group in events_raw.groupby("claim_id"):
            sorted_events = group.sort_values("timestamp")["event"].tolist()
            for i in range(len(sorted_events) - 1):
                event_pairs.append((sorted_events[i], sorted_events[i+1]))

        pair_counts = pd.Series(event_pairs).value_counts().reset_index()
        pair_counts.columns = ["pair", "count"]
        pair_counts["source"] = pair_counts["pair"].apply(lambda x: x[0])
        pair_counts["target"] = pair_counts["pair"].apply(lambda x: x[1])

        all_nodes = list(set(pair_counts["source"].tolist() + pair_counts["target"].tolist()))
        node_map = {n: i for i, n in enumerate(all_nodes)}

        fig = go.Figure(go.Sankey(
            node=dict(
                pad=20, thickness=25,
                label=all_nodes,
                color=[COLORS["primary"], COLORS["accent"], COLORS["success"]][:len(all_nodes)] +
                      [COLORS["warning"]] * max(0, len(all_nodes) - 3)
            ),
            link=dict(
                source=[node_map[s] for s in pair_counts["source"]],
                target=[node_map[t] for t in pair_counts["target"]],
                value=pair_counts["count"].tolist(),
                color="rgba(108,99,255,0.25)"
            )
        ))
        plotly_layout(fig, "🔄 Claim Lifecycle Flow", 400)
        st.plotly_chart(fig, use_container_width=True)

    # Monthly processing trend
    st.markdown("---")
    st.markdown("### 📅 Monthly Processing Time Trend")
    tl_m = tl.copy()
    tl_m["month"] = tl_m["CREATED"].dt.to_period("M").astype(str)
    monthly_time = tl_m.groupby("month").agg(
        avg_submit=("days_to_submit", "mean"),
        avg_process=("days_to_process", "mean"),
        avg_total=("total_cycle_days", "mean")
    ).reset_index()

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=monthly_time["month"], y=monthly_time["avg_submit"],
                             name="Avg Submit Days", mode="lines+markers",
                             line=dict(color=COLORS["primary"], width=2.5)))
    fig.add_trace(go.Scatter(x=monthly_time["month"], y=monthly_time["avg_process"],
                             name="Avg Process Days", mode="lines+markers",
                             line=dict(color=COLORS["accent"], width=2.5)))
    fig.add_trace(go.Scatter(x=monthly_time["month"], y=monthly_time["avg_total"],
                             name="Avg Total Cycle", mode="lines+markers",
                             line=dict(color=COLORS["warning"], width=2.5, dash="dash")))
    plotly_layout(fig, "", 350)
    fig.update_yaxes(title="Days")
    st.plotly_chart(fig, use_container_width=True)

    st.info(f"💡 **AI Insight:** Average total cycle time is **{tl['total_cycle_days'].mean():.1f} days**. "
            f"Automating claim submission could reduce submission time by up to 60%, "
            f"bringing the total cycle closer to **{tl['total_cycle_days'].mean() * 0.7:.0f} days**.")


# ══════════════════════════════════════════════
#  PAGE 7 — AI DENIAL PREDICTOR
# ══════════════════════════════════════════════
def page_denial_predictor():
    st.markdown("# 🧠 AI Denial Predictor & Risk Analysis")
    st.caption("Predictive denial management powered by XGBoost & SHAP")

    # ── Load/Train Model ──
    @st.cache_resource(show_spinner=False)
    def cached_model():
        model, meta = ml_engine.load_model()
        if model is None:
            # First time training
            with st.spinner("🚀 Training AI Predictive Engine... (First time setup)"):
                m, c = build_master(), get_cpt_summary()
                results = ml_engine.train_model(m, c)
                return results["model"], results
        return model, meta

    model, meta = cached_model()

    tab1, tab2, tab3 = st.tabs(["🎯 Interactive Predictor", "📊 Model Performance", "📂 Portfolio Risk"])

    with tab1:
        st.markdown("### 🔍 Predict New Claim Denial Risk")
        st.info("Input claim details below to analyze the probability of denial.")

        with st.form("prediction_form"):
            c1, c2, c3 = st.columns(3)
            with c1:
                ins = st.selectbox("Payer / Insurance", ["BCBS", "Humana", "UHC", "Cigna", "Aetna"])
                visit = st.selectbox("Visit Type", ["ER", "OPD", "IPD"])
                icd = st.selectbox("ICD Code", ["K21", "I10", "N39", "J45", "E11", "M54"])
            with c2:
                amt = st.slider("Claim Amount ($)", 100, 25000, 5000)
                age = st.slider("Patient Age", 0, 100, 45)
                gender = st.selectbox("Patient Gender", ["M", "F"])
            with c3:
                scrub1 = st.checkbox("CPT-ICD Mismatch Detected")
                scrub2 = st.checkbox("High Amount Flagged")
                scrub3 = st.checkbox("Strict Insurance Rules Applied")
                fraud_f = st.slider("Initial Fraud Score", 0.0, 1.0, 0.3)

            submitted = st.form_submit_state = st.form_submit_button("⚡ Run AI Analysis", use_container_width=True)

        if submitted:
            # Prepare input
            claim_data = {
                "insurance": ins, "visit_type": visit, "icd_code": icd,
                "claim_amount": amt, "age": age, "gender": gender,
                "fraud_score": fraud_f, "num_cpt_codes": 1, "total_cpt_amount": amt,
                "cpt_icd_mismatch": scrub1, "high_amount_flag": scrub2, "strict_insurance_flag": scrub3
            }

            res = ml_engine.predict_single_claim(claim_data, model, meta["feature_cols"])
            prob = res["denial_probability"]
            risk = res["risk_level"]

            st.markdown("---")
            pc1, pc2 = st.columns([1, 1.2])

            with pc1:
                st.markdown(f"""
                <div class="prediction-result">
                    <div style="font-size:0.9rem; color:#94A3B8; text-transform:uppercase;">Denial Probability</div>
                    <div class="prediction-prob">{prob*100:.1f}%</div>
                    <div class="risk-badge risk-badge-{risk.lower()}">{risk} RISK</div>
                </div>
                """, unsafe_allow_html=True)

            with pc2:
                st.markdown("#### 🛠️ AI Recommendation")
                if risk == "HIGH":
                    st.error(f"⚠️ **Action Required:** This claim is highly likely to be denied. "
                             f"AI recommends reviewing {'**Mismatch** flags' if scrub1 else '**Auth documentation**'} "
                             f"before submission.")
                elif risk == "MEDIUM":
                    st.warning(f"🔔 **Caution:** Moderate risk detected. Verify that coverage details are up-to-date.")
                else:
                    st.success(f"✅ **Safe to Proceed:** Low probability of denial. Proceed with standard submission.")

                # SHAP Explanation
                st.markdown("#### 🧬 Top Risk Drivers")
                shap_df = res["shap_explanation"]
                fig = go.Figure(go.Bar(
                    x=shap_df["shap_value"],
                    y=shap_df["feature"],
                    orientation="h",
                    marker_color=[COLORS["danger"] if x > 0 else COLORS["success"] for x in shap_df["shap_value"]]
                ))
                plotly_layout(fig, "", 220)
                fig.update_layout(margin=dict(l=0, r=0, t=0, b=0))
                st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.markdown("### 📊 ML Model Performance Metrics")
        m = meta["metrics"]
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("ROC-AUC Score", f"{m['auc']:.3f}", delta="Predictive Power")
        k2.metric("Precision", f"{m['precision']*100:.1f}%", delta="Confidence")
        k3.metric("Recall", f"{m['recall']*100:.1f}%", delta="Coverage")
        k4.metric("F1-Score", f"{m['f1']:.3f}")

        st.markdown("---")
        colp1, colp2 = st.columns(2)

        with colp1:
            # ROC Curve
            roc = meta["roc_data"]
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=roc["fpr"], y=roc["tpr"], name="ROC", line=dict(color=COLORS["accent"], width=3)))
            fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], name="Random", line=dict(dash="dash", color=COLORS["muted"])))
            plotly_layout(fig, "📈 ROC Curve", 350)
            fig.update_xaxes(title="False Positive Rate")
            fig.update_yaxes(title="True Positive Rate")
            st.plotly_chart(fig, use_container_width=True)

        with colp2:
            # Feature Importance
            feat = meta["feature_importance"].head(10)
            fig = go.Figure(go.Bar(
                y=feat["feature"], x=feat["importance"],
                orientation="h", marker_color=COLORS["primary"]
            ))
            plotly_layout(fig, "🧬 Global Feature Importance (SHAP)", 350)
            st.plotly_chart(fig, use_container_width=True)

    with tab3:
        st.markdown("### 📂 Portfolio Risk Analysis")
        st.caption("Denied probability distribution across the entire claims portfolio (50,000 cases)")

        all_probs = meta["all_probabilities"]
        fig = px.histogram(all_probs, nbins=50, color_discrete_sequence=[COLORS["secondary"]])
        plotly_layout(fig, "📊 Denial Probability Distribution", 350)
        fig.update_xaxes(title="Denial Probability")
        fig.update_yaxes(title="Claims Count")
        # Threshold line
        fig.add_vline(x=0.7, line_dash="dash", line_color=COLORS["danger"],
                      annotation_text="High Risk Threshold", annotation_font_color=COLORS["danger"])
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")
        st.markdown("### 🚨 Top 20 High-Risk Claims (Unsubmitted)")

        # Create a table of risky claims
        risky_df = master.copy()
        risky_df["denial_probability"] = all_probs
        risky_df = risky_df[risky_df["denial_probability"] > 0.7].sort_values("denial_probability", ascending=False)

        st.dataframe(
            risky_df[["claim_id", "insurance", "claim_amount", "visit_type", "icd_code", "denial_probability"]].head(20),
            use_container_width=True, hide_index=True
        )

        projected_recovery = (risky_df["claim_amount"] * 0.4).sum()
        st.info(f"💡 **Portfolio Insight:** There are **{len(risky_df):,} claims** in the high-risk pool. "
                f"Proactive intervention on these could prevent approximately **{fmt_dollar(projected_recovery)}** in denials.")

#
# ══════════════════════════════════════════════
#  PAGE — AGENTIC RCM WORKFLOW
# ══════════════════════════════════════════════
def page_agentic_rcm_agent():
    st.markdown("# 🤖 Agentic RCM Agent (Demo)")
    st.caption("A starter multi-step agent that orchestrates ML predictions + action suggestions.")

    @st.cache_resource(show_spinner=False)
    def cached_models():
        # Train/load denial predictor
        denial_model, denial_meta = ml_engine.load_model()
        if denial_model is None:
            with st.spinner("🚀 Training denial predictor..."):
                m = build_master()
                c = get_cpt_summary()
                results = ml_engine.train_model(m, c)
                denial_model = results["model"]
                denial_meta = {"feature_cols": results["feature_cols"], "all_probabilities": results["all_probabilities"]}

        # Train/load CPT-ICD mismatch model
        mismatch_model, mismatch_meta = ml_engine.load_mismatch_model()
        if mismatch_model is None:
            with st.spinner("🚀 Training CPT-ICD mismatch detector..."):
                m = build_master()
                c = get_cpt_summary()
                results = ml_engine.train_mismatch_model(m, c)
                mismatch_model = results["model"]
                mismatch_meta = results["meta"]

        # Train/load appeals models
        appeals_success_model, appeals_success_meta = ml_engine.load_appeals_success_model()
        if appeals_success_model is None:
            with st.spinner("🚀 Training appeals success model..."):
                m = build_master()
                c = get_cpt_summary()
                res = ml_engine.train_appeals_success_model(m, c)
                appeals_success_model = res["model"]
                appeals_success_meta = res["meta"]

        appeals_recovery_model, appeals_recovery_meta = ml_engine.load_appeals_recovery_model()
        if appeals_recovery_model is None:
            with st.spinner("🚀 Training appeals recovery model..."):
                m = build_master()
                c = get_cpt_summary()
                res = ml_engine.train_appeals_recovery_model(m, c)
                appeals_recovery_model = res["model"]
                appeals_recovery_meta = res["meta"]

        # Train/load fraud models
        fraud_prob_model, fraud_prob_meta = ml_engine.load_fraud_probability_model()
        if fraud_prob_model is None:
            with st.spinner("🚀 Training fraud probability model..."):
                m = build_master()
                c = get_cpt_summary()
                res = ml_engine.train_fraud_probability_model(m, c)
                fraud_prob_model = res["model"]
                fraud_prob_meta = res["meta"]

        fraud_anomaly_model, fraud_anomaly_meta = ml_engine.load_fraud_anomaly_model()
        if fraud_anomaly_model is None:
            with st.spinner("🚀 Training fraud anomaly model..."):
                m = build_master()
                res = ml_engine.train_fraud_anomaly_model(m)
                fraud_anomaly_model = res["model"]
                fraud_anomaly_meta = res["meta"]

        return (
            denial_model,
            denial_meta,
            mismatch_model,
            mismatch_meta,
            appeals_success_model,
            appeals_success_meta,
            appeals_recovery_model,
            appeals_recovery_meta,
            fraud_prob_model,
            fraud_prob_meta,
            fraud_anomaly_model,
            fraud_anomaly_meta,
        )

    @st.cache_data(show_spinner=False)
    def cached_claim_predictions():
        cpt_summary = get_cpt_summary()
        master_reset = master.reset_index(drop=True)
        raw_local = load_all()
        coding_knowledge = build_cpt_icd_knowledge(
            raw_local["cpt_lines"],
            raw_local["icd"],
            raw_local["claims"],
        )
        nlp_model_bundle = train_notes_to_icd_model(
            raw_local["encounters"],
            raw_local["claims"],
            raw_local["icd"],
            min_samples=50,
        )
        encounter_notes_map = raw_local["encounters"].set_index("encounter_id")["clinical_notes"].to_dict()
        notes_batch = [encounter_notes_map.get(eid, "") for eid in master_reset["encounter_id"].tolist()]
        note_preds_batch = predict_icd_from_notes_batch(notes_batch, nlp_model_bundle, top_k=3)

        # denial_probability per claim (from denial_meta if available)
        _, denial_meta_local = ml_engine.load_model()
        denial_probs = None
        if denial_meta_local is not None and isinstance(denial_meta_local, dict) and "all_probabilities" in denial_meta_local:
            denial_probs = np.array(denial_meta_local["all_probabilities"], dtype=float)

        # mismatch_probability per claim
        mismatch_model_local, mismatch_meta_local = ml_engine.load_mismatch_model()
        mismatch_probs = None
        if mismatch_model_local is not None:
            mismatch_probs = ml_engine.score_all_mismatch(
                master_reset,
                cpt_summary,
                mismatch_model_local,
                mismatch_meta_local["feature_cols"],
            )

        # fraud_probability_improved per claim
        fraud_prob_model_local, fraud_prob_meta_local = ml_engine.load_fraud_probability_model()
        fraud_anomaly_model_local, fraud_anomaly_meta_local = ml_engine.load_fraud_anomaly_model()
        fraud_df = None
        if fraud_prob_model_local is not None and fraud_anomaly_model_local is not None:
            fraud_df = ml_engine.score_fraud_enhanced(
                master_reset,
                cpt_summary,
                fraud_prob_model_local,
                fraud_prob_meta_local["feature_cols"],
                fraud_anomaly_model_local,
                fraud_anomaly_meta_local,
                alpha=0.6,
            )

        # reconciliation risk per claim
        recon_model_local, recon_meta_local = ml_engine.load_reconciliation_risk_model()
        recon_probs = None
        if recon_model_local is not None and recon_meta_local is not None:
            recon_probs = ml_engine.score_reconciliation_risk(master_reset, recon_model_local, recon_meta_local["feature_cols"])

        # appeal ranking (only for denied & not appealed)
        denied = master_reset[master_reset["is_denied"]].copy()
        not_appealed = denied[~denied["is_appealed"]].copy()
        appeals_success_model_local, appeals_success_meta_local = ml_engine.load_appeals_success_model()
        appeals_recovery_model_local, appeals_recovery_meta_local = ml_engine.load_appeals_recovery_model()
        ranked = None
        if appeals_success_model_local is not None and appeals_recovery_model_local is not None and len(not_appealed) > 0:
            ranked = ml_engine.predict_appeals_ranked_candidates(
                not_appealed,
                cpt_summary,
                appeals_success_model_local,
                appeals_success_meta_local["feature_cols"],
                appeals_recovery_model_local,
                appeals_recovery_meta_local["feature_cols"],
            )
            ranked = ranked.sort_values("expected_recovery", ascending=False)

        # Assemble a lookup dict
        by_id = {}
        for i, row in master_reset.iterrows():
            cid = int(row["claim_id"])
            cpt_row = cpt_summary[cpt_summary["claim_id"] == cid]
            cpt_codes = []
            if len(cpt_row) > 0:
                cpt_codes_str = str(cpt_row.iloc[0].get("cpt_codes_list", ""))
                cpt_codes = [x.strip() for x in cpt_codes_str.split(",") if x.strip()]

            coding_reco = build_coding_recommendation(
                row.to_dict(),
                cpt_codes,
                coding_knowledge,
                float(mismatch_probs[i]) if mismatch_probs is not None else 0.0,
                min_support=20,
            )
            nlp_coding_reco = build_nlp_coding_recommendation(
                row.to_dict(),
                note_preds_batch[i] if i < len(note_preds_batch) else {},
                float(mismatch_probs[i]) if mismatch_probs is not None else 0.0,
            )

            by_id[int(row["claim_id"])] = {
                "denial_probability": float(denial_probs[i]) if denial_probs is not None else float(np.nan),
                "mismatch_probability": float(mismatch_probs[i]) if mismatch_probs is not None else float(np.nan),
                "fraud_probability_improved": float(fraud_df.loc[i, "fraud_probability_improved"]) if fraud_df is not None else float(np.nan),
                "reconciliation_risk_probability": float(recon_probs[i]) if recon_probs is not None else float(np.nan),
                "coding_recommendation": coding_reco,
                "nlp_coding_recommendation": nlp_coding_reco,
            }
        if ranked is not None:
            for _, r in ranked.iterrows():
                cid = int(r["claim_id"])
                by_id[cid]["p_appeal_success"] = float(r["p_appeal_success"])
                by_id[cid]["expected_recovery"] = float(r["expected_recovery"])
        return by_id

    _ = cached_models()  # Ensure models exist/trained (trigger cache initialization)
    predictions_by_id = cached_claim_predictions()

    # Input
    col_i1, col_i2 = st.columns([2, 3])
    with col_i1:
        claim_id = st.number_input("Enter Claim ID", min_value=1, step=1, value=int(master["claim_id"].iloc[0]))
    with col_i2:
        st.markdown("<br>", unsafe_allow_html=True)
        force_appeal = st.checkbox("Draft Appeal Letter even if predicted success is low", value=False)

    if claim_id not in predictions_by_id:
        st.error("Claim ID not found in dataset.")
        return

    # Extract claim row
    claim_row = master[master["claim_id"] == claim_id].iloc[0].to_dict()
    pred = predictions_by_id[claim_id]

    # Run agent
    agent = CoordinatorAgent()
    agent_out = agent.run(claim=claim_row, predictions=pred, force_appeal=force_appeal)

    st.markdown("---")
    st.markdown("### Final Recommendation")
    st.success(agent_out.recommendation)

    st.markdown("### Agent Steps (Demo Log)")
    for s in agent_out.steps:
        st.write(f"**{s.agent} | {s.step}:** {s.summary}")

    st.markdown("### Action Items")
    if agent_out.action_items:
        for a in agent_out.action_items:
            st.write(f"- {a}")
    else:
        st.write("No actions required.")

    coding_reco = pred.get("coding_recommendation") if isinstance(pred, dict) else None
    nlp_coding_reco = pred.get("nlp_coding_recommendation") if isinstance(pred, dict) else None
    if isinstance(coding_reco, dict):
        st.markdown("### Custom Coding Agent Output")
        st.info(coding_reco.get("recommendation", "No coding recommendation."))
        st.write(
            f"Source: `{coding_reco.get('source_used', 'unknown')}` | "
            f"Payer: `{coding_reco.get('payer', 'UNKNOWN')}` | "
            f"Visit Type: `{coding_reco.get('visit_type', 'UNKNOWN')}` | "
            f"Support: `{coding_reco.get('support_used', 0)}` (min `{coding_reco.get('min_support', 20)}`)"
        )
        suggestions = coding_reco.get("suggestions", [])
        if suggestions:
            suggestion_text = ", ".join([f"{icd} ({score:.2f})" for icd, score in suggestions[:3]])
            st.write(f"Suggested ICD candidates: {suggestion_text}")
    if isinstance(nlp_coding_reco, dict):
        st.markdown("### NLP Clinical Notes Coding Support")
        st.info(nlp_coding_reco.get("recommendation", "No NLP coding recommendation."))
        nlp_suggestions = nlp_coding_reco.get("suggestions", [])
        if nlp_suggestions:
            nlp_text = ", ".join([f"{icd} ({score:.2f})" for icd, score in nlp_suggestions[:3]])
            st.write(f"NLP ICD candidates: {nlp_text}")

    if agent_out.appeal_letter:
        st.markdown("### Draft Appeal Letter (Professional Template)")
        st.text_area("Appeal Letter", agent_out.appeal_letter, height=400)
    elif claim_row.get("is_denied"):
        st.markdown("### Draft Appeal Letter")
        st.warning(
            f"ML model predicts a low success probability ({pred.get('p_appeal_success', 0.0):.1%}) "
            f"for this claim. Draft skipped to prioritize high-yield recovery."
        )
        st.info("Tip: Use the checkbox above to 'Force Draft' if you still wish to proceed.")

    st.markdown("---")
    st.markdown("### Generate 2-page Summary (Groq)")
    st.caption("Agent output se 2-page executive summary banata hai. Groq ke liye `GROQ_API_KEY` set hona chahiye.")

    if st.button("Generate with Groq", use_container_width=True):
        try:
            with st.spinner("Calling Groq to generate 2-page summary..."):
                page1_md, page2_md = generate_two_page_agent_summary(agent_out)

            st.success("Summary generated.")

            tab_p1, tab_p2 = st.tabs(["Page 1", "Page 2"])
            with tab_p1:
                st.markdown(page1_md)
            with tab_p2:
                st.markdown(page2_md)

            # Save artifacts for download / future PDF printing.
            out_dir = os.path.join(os.path.dirname(__file__), "outputs")
            os.makedirs(out_dir, exist_ok=True)
            page1_path = os.path.join(out_dir, "agent_summary_page1.md")
            page2_path = os.path.join(out_dir, "agent_summary_page2.md")

            with open(page1_path, "w", encoding="utf-8") as f:
                f.write(page1_md)
            with open(page2_path, "w", encoding="utf-8") as f:
                f.write(page2_md)

            page1_html = markdown_to_html_page(page1_md, title="Agent Summary - Page 1")
            page2_html = markdown_to_html_page(page2_md, title="Agent Summary - Page 2")
            page1_html_path = os.path.join(out_dir, "agent_summary_page1.html")
            page2_html_path = os.path.join(out_dir, "agent_summary_page2.html")

            with open(page1_html_path, "w", encoding="utf-8") as f:
                f.write(page1_html)
            with open(page2_html_path, "w", encoding="utf-8") as f:
                f.write(page2_html)

            st.markdown("#### Download")
            st.download_button(
                label="Download Page 1 (MD)",
                data=page1_md,
                file_name="agent_summary_page1.md",
                mime="text/markdown",
                use_container_width=True,
            )
            st.download_button(
                label="Download Page 2 (MD)",
                data=page2_md,
                file_name="agent_summary_page2.md",
                mime="text/markdown",
                use_container_width=True,
            )
            st.info(
                "PDF ke liye: generated `.html` ko browser me open karo aur Print -> Save as PDF karo."
            )
        except Exception as e:
            st.error(f"Groq summary generation failed: {e}")

    st.markdown("---")
    st.markdown("### Generate 2-page Summary (Ollama)")
    st.caption("Local Ollama se 2-page executive summary banata hai (no API key).")

    ollama_model = st.text_input("Ollama Model", value=os.getenv("OLLAMA_MODEL", "llama3:70b"))
    if st.button("Generate with Ollama", use_container_width=True):
        try:
            with st.spinner("Calling Ollama to generate 2-page summary..."):
                page1_md, page2_md = generate_two_page_agent_summary_ollama(
                    agent_out, model=ollama_model
                )

            st.success("Ollama summary generated.")

            tab_p1, tab_p2 = st.tabs(["Page 1", "Page 2"])
            with tab_p1:
                st.markdown(page1_md)
            with tab_p2:
                st.markdown(page2_md)

            out_dir = os.path.join(os.path.dirname(__file__), "outputs")
            os.makedirs(out_dir, exist_ok=True)
            page1_path = os.path.join(out_dir, "agent_summary_page1_ollama.md")
            page2_path = os.path.join(out_dir, "agent_summary_page2_ollama.md")

            with open(page1_path, "w", encoding="utf-8") as f:
                f.write(page1_md)
            with open(page2_path, "w", encoding="utf-8") as f:
                f.write(page2_md)

            page1_html = markdown_to_html_page(page1_md, title="Agent Summary - Page 1 (Ollama)")
            page2_html = markdown_to_html_page(page2_md, title="Agent Summary - Page 2 (Ollama)")
            page1_html_path = os.path.join(out_dir, "agent_summary_page1_ollama.html")
            page2_html_path = os.path.join(out_dir, "agent_summary_page2_ollama.html")

            with open(page1_html_path, "w", encoding="utf-8") as f:
                f.write(page1_html)
            with open(page2_html_path, "w", encoding="utf-8") as f:
                f.write(page2_html)

            st.info("PDF ke liye: generated `.html` ko browser me open karo aur Print -> Save as PDF karo.")

            st.download_button(
                label="Download Page 1 (MD) - Ollama",
                data=page1_md,
                file_name="agent_summary_page1_ollama.md",
                mime="text/markdown",
                use_container_width=True,
            )
            st.download_button(
                label="Download Page 2 (MD) - Ollama",
                data=page2_md,
                file_name="agent_summary_page2_ollama.md",
                mime="text/markdown",
                use_container_width=True,
            )
        except Exception as e:
            st.error(f"Ollama summary generation failed: {e}")

    st.markdown("---")
    st.markdown("### Metrics (Model Outputs)")
    st.json(agent_out.metrics)

#
# ══════════════════════════════════════════════
#  Page — LangGraph Chatbot
# ══════════════════════════════════════════════
def page_langgraph_chatbot():
    st.markdown("# 💬 LangGraph Chatbot (Demo)")
    st.caption("Claim-level insights orchestrate using LangGraph nodes: denial/scrubbing/appeals/fraud.")

    # Load/train all models once (cached).
    @st.cache_resource(show_spinner=False)
    def cached_langgraph_models():
        # Denial predictor
        denial_model, denial_meta = ml_engine.load_model()
        if denial_model is None:
            with st.spinner("🚀 Training denial predictor..."):
                results = ml_engine.train_model(master, get_cpt_summary())
                denial_model = results["model"]
                denial_meta = {"feature_cols": results["feature_cols"], "all_probabilities": results["all_probabilities"]}

        # CPT-ICD mismatch
        mismatch_model, mismatch_meta = ml_engine.load_mismatch_model()
        if mismatch_model is None:
            with st.spinner("🚀 Training CPT-ICD mismatch detector..."):
                results = ml_engine.train_mismatch_model(master, get_cpt_summary())
                mismatch_model = results["model"]
                mismatch_meta = results["meta"]

        # Appeals success + recovery
        appeals_success_model, appeals_success_meta = ml_engine.load_appeals_success_model()
        if appeals_success_model is None:
            with st.spinner("🚀 Training appeals success model..."):
                results = ml_engine.train_appeals_success_model(master, get_cpt_summary())
                appeals_success_model = results["model"]
                appeals_success_meta = results["meta"]

        appeals_recovery_model, appeals_recovery_meta = ml_engine.load_appeals_recovery_model()
        if appeals_recovery_model is None:
            with st.spinner("🚀 Training appeals recovery model..."):
                results = ml_engine.train_appeals_recovery_model(master, get_cpt_summary())
                appeals_recovery_model = results["model"]
                appeals_recovery_meta = results["meta"]

        # Fraud probability + anomaly
        fraud_prob_model, fraud_prob_meta = ml_engine.load_fraud_probability_model()
        if fraud_prob_model is None:
            with st.spinner("🚀 Training fraud probability model..."):
                results = ml_engine.train_fraud_probability_model(master, get_cpt_summary())
                fraud_prob_model = results["model"]
                fraud_prob_meta = results["meta"]

        fraud_anomaly_model, fraud_anomaly_meta = ml_engine.load_fraud_anomaly_model()
        if fraud_anomaly_model is None:
            with st.spinner("🚀 Training fraud anomaly model..."):
                results = ml_engine.train_fraud_anomaly_model(master)
                fraud_anomaly_model = results["model"]
                fraud_anomaly_meta = results["meta"]

        return (
            denial_model,
            denial_meta,
            mismatch_model,
            mismatch_meta,
            appeals_success_model,
            appeals_success_meta,
            appeals_recovery_model,
            appeals_recovery_meta,
            fraud_prob_model,
            fraud_prob_meta,
            fraud_anomaly_model,
            fraud_anomaly_meta,
        )

    (
        denial_model,
        denial_meta,
        mismatch_model,
        mismatch_meta,
        appeals_success_model,
        appeals_success_meta,
        appeals_recovery_model,
        appeals_recovery_meta,
        fraud_prob_model,
        fraud_prob_meta,
        fraud_anomaly_model,
        fraud_anomaly_meta,
    ) = cached_langgraph_models()

    # Precompute insights for all claims once per session.
    if "lg_insights_by_id" not in st.session_state:
        with st.spinner("🧠 Computing claim insights for LangGraph chatbot..."):
            cpt_summary = get_cpt_summary()
            master_reset = master.reset_index(drop=True)
            raw_local = load_all()
            coding_knowledge = build_cpt_icd_knowledge(
                raw_local["cpt_lines"],
                raw_local["icd"],
                raw_local["claims"],
            )
            nlp_model_bundle = train_notes_to_icd_model(
                raw_local["encounters"],
                raw_local["claims"],
                raw_local["icd"],
                min_samples=50,
            )
            encounter_notes_map = raw_local["encounters"].set_index("encounter_id")["clinical_notes"].to_dict()
            notes_batch = [encounter_notes_map.get(eid, "") for eid in master_reset["encounter_id"].tolist()]
            note_preds_batch = predict_icd_from_notes_batch(notes_batch, nlp_model_bundle, top_k=3)

            denial_probs = denial_meta.get("all_probabilities", None)
            if denial_probs is None:
                tmp_results = ml_engine.train_model(master, cpt_summary)
                denial_probs = tmp_results["all_probabilities"]

            mismatch_probs = ml_engine.score_all_mismatch(
                master_reset,
                cpt_summary,
                mismatch_model,
                mismatch_meta["feature_cols"],
            )

            fraud_df = ml_engine.score_fraud_enhanced(
                master_reset,
                cpt_summary,
                fraud_prob_model,
                fraud_prob_meta["feature_cols"],
                fraud_anomaly_model,
                fraud_anomaly_meta,
                alpha=0.6,
            )

            denied = master_reset[master_reset["is_denied"]].copy()
            not_appealed = denied[~denied["is_appealed"]].copy()
            ranked = None
            if len(not_appealed) > 0:
                ranked = ml_engine.predict_appeals_ranked_candidates(
                    not_appealed,
                    cpt_summary,
                    appeals_success_model,
                    appeals_success_meta["feature_cols"],
                    appeals_recovery_model,
                    appeals_recovery_meta["feature_cols"],
                ).sort_values("expected_recovery", ascending=False)

            appeal_map = {}
            if ranked is not None:
                for _, r in ranked.iterrows():
                    cid = int(r["claim_id"])
                    appeal_map[cid] = {
                        "p_appeal_success": float(r["p_appeal_success"]),
                        "expected_recovery": float(r["expected_recovery"]),
                    }

            claim_ids = master_reset["claim_id"].astype(int).values
            insights_by_id = {}
            for i, cid in enumerate(claim_ids):
                cpt_row = cpt_summary[cpt_summary["claim_id"] == int(cid)]
                cpt_codes = []
                if len(cpt_row) > 0:
                    cpt_codes_str = str(cpt_row.iloc[0].get("cpt_codes_list", ""))
                    cpt_codes = [x.strip() for x in cpt_codes_str.split(",") if x.strip()]

                coding_reco = build_coding_recommendation(
                    master_reset.iloc[i].to_dict(),
                    cpt_codes,
                    coding_knowledge,
                    float(mismatch_probs[i]),
                    min_support=20,
                )
                nlp_coding_reco = build_nlp_coding_recommendation(
                    master_reset.iloc[i].to_dict(),
                    note_preds_batch[i] if i < len(note_preds_batch) else {},
                    float(mismatch_probs[i]),
                )

                ins = {
                    "denial_probability": float(denial_probs[i]),
                    "mismatch_probability": float(mismatch_probs[i]),
                    "fraud_probability_improved": float(fraud_df.loc[i, "fraud_probability_improved"]),
                    "p_appeal_success": None,
                    "expected_recovery": None,
                    "coding_recommendation": coding_reco,
                    "nlp_coding_recommendation": nlp_coding_reco,
                }
                if int(cid) in appeal_map:
                    ins["p_appeal_success"] = appeal_map[int(cid)]["p_appeal_success"]
                    ins["expected_recovery"] = appeal_map[int(cid)]["expected_recovery"]
                insights_by_id[int(cid)] = ins

            st.session_state["lg_insights_by_id"] = insights_by_id
            st.session_state["lg_master_reset"] = master_reset
            st.session_state["lg_cpt_summary"] = cpt_summary

    master_reset = st.session_state["lg_master_reset"]
    cpt_summary = st.session_state["lg_cpt_summary"]
    insights_by_id = st.session_state["lg_insights_by_id"]

    # Claim selector
    default_claim_id = int(master_reset["claim_id"].iloc[0])
    claim_id = int(st.number_input("Enter Claim ID for LangGraph chat", min_value=1, step=1, value=default_claim_id))
    if claim_id not in insights_by_id:
        st.error("Claim ID not found in dataset.")
        return

    # Reset chat on claim change
    if st.session_state.get("lg_selected_claim_id") != claim_id:
        st.session_state["lg_selected_claim_id"] = claim_id
        st.session_state["lg_messages"] = [{
            "role": "assistant",
            "content": f"Claim {claim_id} selected. Ask denial risk, scrubbing/coding (CPT-ICD), appeals, or fraud."
        }]

    # Render history
    if "lg_messages" not in st.session_state:
        st.session_state["lg_messages"] = []
    for msg in st.session_state["lg_messages"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_prompt = st.chat_input("Type: denial / scrub / appeal / fraud / why (SHAP)...")
    if user_prompt:
        st.session_state["lg_messages"].append({"role": "user", "content": user_prompt})
        with st.chat_message("user"):
            st.markdown(user_prompt)

        prompt_lower = user_prompt.lower()
        include_shap = any(k in prompt_lower for k in ["why", "shap", "driver", "reason", "denial", "risk"])

        denial_shap_top_drivers = None
        if include_shap:
            claim_row = master_reset[master_reset["claim_id"] == claim_id].iloc[0].to_dict()
            cs_row = cpt_summary[cpt_summary["claim_id"] == claim_id].iloc[0]
            claim_dict = {
                "insurance": claim_row.get("insurance"),
                "visit_type": claim_row.get("visit_type"),
                "icd_code": claim_row.get("icd_code"),
                "gender": claim_row.get("gender"),
                "claim_amount": claim_row.get("claim_amount"),
                "age": claim_row.get("age"),
                "fraud_score": claim_row.get("fraud_score"),
                "num_cpt_codes": float(cs_row.get("num_cpt_codes", 1.0)),
                "total_cpt_amount": float(cs_row.get("total_cpt_amount", claim_row.get("claim_amount", 0.0))),
                "cpt_icd_mismatch": bool(claim_row.get("cpt_icd_mismatch", False)),
                "high_amount_flag": bool(claim_row.get("high_amount_flag", False)),
                "strict_insurance_flag": bool(claim_row.get("strict_insurance_flag", False)),
            }
            pred = ml_engine.predict_single_claim(claim_dict, denial_model, denial_meta["feature_cols"])
            shap_df = pred.get("shap_explanation")
            if shap_df is not None and hasattr(shap_df, "head"):
                denial_shap_top_drivers = [
                    {"feature": str(r["feature"]), "shap_value": float(r["shap_value"])}
                    for _, r in shap_df.head(6).iterrows()
                ]

        claim_row = master_reset[master_reset["claim_id"] == claim_id].iloc[0].to_dict()
        graph_state = {
            "claim_id": claim_id,
            "user_message": user_prompt,
            "claim": claim_row,
            "insights": insights_by_id[claim_id],
            "denial_shap_top_drivers": denial_shap_top_drivers,
        }
        result = rcm_langgraph_app.invoke(graph_state)
        assistant_text = result.get("response", "Sorry, something went wrong.")

        st.session_state["lg_messages"].append({"role": "assistant", "content": assistant_text})
        with st.chat_message("assistant"):
            st.markdown(assistant_text)


# ══════════════════════════════════════════════
#  PAGE — PATIENT ACCESS & ELIGIBILITY
# ══════════════════════════════════════════════
def page_patient_access_eligibility():
    st.markdown("# 🧾 Patient Access & Eligibility")
    st.caption("Front-end automation for registration quality, eligibility checks, and patient support.")

    df = master.copy()
    # Patient insurance is kept from patients merge as `insurance_pat` in build_master.
    if "insurance_pat" not in df.columns:
        df["insurance_pat"] = df["insurance"]
    df["insurance_pat"] = df["insurance_pat"].fillna("Unknown")
    df["insurance_match"] = (df["insurance"].astype(str) == df["insurance_pat"].astype(str))
    @st.cache_resource(show_spinner=False)
    def cached_eligibility_model():
        m, meta = ml_engine.load_eligibility_risk_model()
        if m is None:
            with st.spinner("🚀 Training eligibility risk model..."):
                res = ml_engine.train_eligibility_risk_model(df)
                return res["model"], res["meta"]
        return m, meta

    elig_model, elig_meta = cached_eligibility_model()
    df["eligibility_risk"] = ml_engine.score_eligibility_risk(df, elig_model, elig_meta["feature_cols"])

    match_rate = df["insurance_match"].mean() * 100
    high_risk_count = int((df["eligibility_risk"] >= 0.65).sum())
    est_avoidable = float(df[df["eligibility_risk"] >= 0.65]["claim_amount"].sum() * 0.15)

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Patients Covered", fmt_number(df["patient_id"].nunique()))
    k2.metric("Insurance Match Rate", f"{match_rate:.1f}%")
    k3.metric("High Eligibility Risk", fmt_number(high_risk_count))
    k4.metric("Avoidable Leakage (Est.)", fmt_dollar(est_avoidable))

    st.markdown("---")
    c1, c2 = st.columns(2)
    with c1:
        payer_match = df.groupby("insurance").agg(
            total=("claim_id", "count"),
            match=("insurance_match", "mean"),
        ).reset_index()
        payer_match["match"] = payer_match["match"] * 100
        fig = go.Figure(go.Bar(
            x=payer_match["insurance"],
            y=payer_match["match"],
            marker_color=COLORS["accent"],
            text=payer_match["match"].round(1).astype(str) + "%",
            textposition="outside"
        ))
        plotly_layout(fig, "Insurance Match Rate by Payer", 350)
        fig.update_yaxes(title="Match %", range=[0, 100])
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        high_elig = df[df["eligibility_risk"] >= 0.65][
            ["claim_id", "patient_id", "insurance", "insurance_pat", "claim_amount", "eligibility_risk"]
        ].sort_values("eligibility_risk", ascending=False).head(15)
        st.markdown("### ⚠️ High-Risk Eligibility Cases")
        st.dataframe(high_elig, use_container_width=True, hide_index=True)
        if isinstance(elig_meta, dict) and "metrics" in elig_meta:
            st.caption(f"Model metrics: {elig_meta['metrics']}")

    st.markdown("---")
    st.markdown("### 💬 Patient Access Assistant (Demo)")
    st.caption("Virtual assistant for eligibility, registration, and patient-access triage (data-aware prototype).")
    chat_key = "pa_chat_messages"
    if chat_key not in st.session_state:
        st.session_state[chat_key] = [
            {
                "role": "assistant",
                "content": (
                    "Hi! I can help with eligibility checks, required registration docs, copay guidance, "
                    "and patient/claim risk lookups.\n\n"
                    "Try: `show high risk cases`, `patient 12345`, `claim 67890`, or `what is match rate?`"
                ),
            }
        ]

    top_risk_df = df.sort_values("eligibility_risk", ascending=False).copy()

    def _extract_first_int(text: str):
        for tok in text.replace(",", " ").split():
            cleaned = "".join(ch for ch in tok if ch.isdigit())
            if cleaned:
                try:
                    return int(cleaned)
                except Exception:
                    pass
        return None

    def _patient_access_reply(user_text: str) -> str:
        ql = user_text.lower().strip()
        entity_id = _extract_first_int(user_text)

        if any(k in ql for k in ["kpi", "summary", "match rate", "dashboard", "overall"]):
            return (
                f"Current patient-access snapshot:\n"
                f"- Insurance match rate: **{match_rate:.1f}%**\n"
                f"- High eligibility risk cases: **{high_risk_count}**\n"
                f"- Avoidable leakage estimate: **{fmt_dollar(est_avoidable)}**\n\n"
                f"Focus first on high-risk claims where `insurance != insurance_pat` or auth is likely missing."
            )

        if ("patient" in ql or "pt" in ql) and entity_id is not None:
            p = top_risk_df[top_risk_df["patient_id"].astype(int) == int(entity_id)]
            if len(p) == 0:
                return f"I could not find patient `{entity_id}` in the current dataset."
            p_top = p.sort_values("eligibility_risk", ascending=False).iloc[0]
            level = "HIGH" if p_top["eligibility_risk"] >= 0.65 else ("MEDIUM" if p_top["eligibility_risk"] >= 0.4 else "LOW")
            return (
                f"Patient `{entity_id}` latest risk snapshot:\n"
                f"- Claim ID: **{int(p_top['claim_id'])}**\n"
                f"- Eligibility risk: **{float(p_top['eligibility_risk'])*100:.1f}% ({level})**\n"
                f"- Insurance (claim vs patient): **{p_top['insurance']} vs {p_top['insurance_pat']}**\n"
                f"- Claim amount: **{fmt_dollar(float(p_top['claim_amount']))}**\n\n"
                f"Recommended next step: verify insurance details + authorization requirement before submission."
            )

        if ("claim" in ql or "clm" in ql) and entity_id is not None:
            c = top_risk_df[top_risk_df["claim_id"].astype(int) == int(entity_id)]
            if len(c) == 0:
                return f"I could not find claim `{entity_id}` in the current dataset."
            r = c.iloc[0]
            level = "HIGH" if r["eligibility_risk"] >= 0.65 else ("MEDIUM" if r["eligibility_risk"] >= 0.4 else "LOW")
            return (
                f"Claim `{entity_id}` eligibility triage:\n"
                f"- Risk: **{float(r['eligibility_risk'])*100:.1f}% ({level})**\n"
                f"- Patient ID: **{int(r['patient_id'])}**\n"
                f"- Insurance match: **{'Yes' if bool(r['insurance_match']) else 'No'}**\n"
                f"- Amount: **{fmt_dollar(float(r['claim_amount']))}**\n\n"
                f"Action: run registration quality check, then confirm policy + auth before final submission."
            )

        if any(k in ql for k in ["high risk", "top risk", "priority", "which cases"]):
            top5 = top_risk_df.head(5)[["claim_id", "patient_id", "eligibility_risk", "insurance", "insurance_pat"]]
            lines = [
                f"- Claim {int(r.claim_id)} | Patient {int(r.patient_id)} | Risk {float(r.eligibility_risk)*100:.1f}% | {r.insurance} vs {r.insurance_pat}"
                for r in top5.itertuples(index=False)
            ]
            return "Top 5 eligibility-priority cases:\n" + "\n".join(lines)

        if any(k in ql for k in ["document", "docs", "eligibility", "registration"]):
            return (
                "Eligibility and registration checklist:\n"
                "- Insurance card (front/back)\n"
                "- Patient government ID\n"
                "- DOB + policy holder verification\n"
                "- Active coverage + plan effective dates\n"
                "- Referral/Auth requirement (if payer requires)\n\n"
                "Best practice: complete this before appointment confirmation to reduce front-desk denials."
            )

        if any(k in ql for k in ["copay", "co-pay", "payment", "estimate"]):
            return (
                "Copay flow:\n"
                "- Verify eligibility first\n"
                "- Estimate patient responsibility by plan\n"
                "- Share pre-visit payment options\n"
                "- Capture payment confirmation notes\n\n"
                "This reduces post-visit AR and improves collection quality."
            )

        if any(k in ql for k in ["schedule", "appointment", "booking"]):
            return (
                "Scheduling recommendation:\n"
                "1) Quick registration quality check\n"
                "2) Eligibility and auth pre-check\n"
                "3) Confirm appointment\n"
                "4) Send patient prep checklist\n\n"
                "This sequence minimizes same-day denial risk."
            )

        return (
            "I can help with:\n"
            "- `show high risk cases`\n"
            "- `patient <id>` or `claim <id>` lookups\n"
            "- eligibility docs checklist\n"
            "- copay/payment and scheduling workflow\n"
            "- dashboard KPI summary"
        )

    for m in st.session_state[chat_key]:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    q = st.chat_input("Example: What documents are required for eligibility verification?")
    if q:
        st.session_state[chat_key].append({"role": "user", "content": q})
        with st.chat_message("user"):
            st.markdown(q)
        ans = _patient_access_reply(q)
        st.session_state[chat_key].append({"role": "assistant", "content": ans})
        with st.chat_message("assistant"):
            st.markdown(ans)

    st.markdown("---")
    st.markdown("### ⚡ Realtime Claim Scoring (Patient Access + Agent)")
    st.caption("Score a new incoming claim payload directly and view CoordinatorAgent output.")

    with st.form("patient_access_realtime_score_form"):
        f1, f2, f3 = st.columns(3)
        with f1:
            claim_id_in = int(st.number_input("Claim ID", min_value=1, value=990001, step=1))
            patient_id_in = int(st.number_input("Patient ID", min_value=1, value=770001, step=1))
            encounter_id_in = int(st.number_input("Encounter ID", min_value=1, value=880001, step=1))
            age_in = int(st.number_input("Age", min_value=0, max_value=120, value=54, step=1))
        with f2:
            insurance_in = st.selectbox("Insurance", sorted(df["insurance"].astype(str).dropna().unique().tolist()))
            visit_type_in = st.selectbox("Visit Type", sorted(df["visit_type"].astype(str).dropna().unique().tolist()))
            gender_in = st.selectbox("Gender", ["F", "M", "U"])
            icd_in = st.text_input("ICD Code", value="E11.9")
        with f3:
            claim_amt_in = float(st.number_input("Claim Amount", min_value=0.0, value=1850.0, step=50.0))
            paid_amt_in = float(st.number_input("Paid Amount", min_value=0.0, value=0.0, step=50.0))
            fraud_score_in = float(st.slider("Fraud Score (input signal)", min_value=0.0, max_value=1.0, value=0.32, step=0.01))
            denial_reason_in = st.selectbox(
                "Denial Reason",
                sorted(df["denial_reason"].fillna("None").astype(str).unique().tolist()),
            )

        cpt_codes_in = st.text_input("CPT Codes (comma-separated)", value="99213,80053")
        clinical_notes_in = st.text_area(
            "Clinical Notes",
            value="Diabetes follow-up, lab review, medication adjustment.",
            height=90,
        )

        b1, b2, b3 = st.columns(3)
        with b1:
            is_denied_in = st.checkbox("Is Denied", value=True)
        with b2:
            is_appealed_in = st.checkbox("Is Appealed", value=False)
        with b3:
            appeal_success_in = st.checkbox("Appeal Success", value=False)

        run_rt = st.form_submit_button("Run Realtime Agent Score", type="primary")

    if run_rt:
        payload = {
            "claim_id": claim_id_in,
            "encounter_id": encounter_id_in,
            "patient_id": patient_id_in,
            "insurance": insurance_in,
            "visit_type": visit_type_in,
            "icd_code": icd_in.strip(),
            "gender": gender_in,
            "age": age_in,
            "claim_amount": claim_amt_in,
            "paid_amount": paid_amt_in,
            "fraud_score": fraud_score_in,
            "denial_reason": denial_reason_in,
            "is_denied": is_denied_in,
            "is_appealed": is_appealed_in,
            "appeal_success": appeal_success_in,
            "cpt_codes": [x.strip() for x in cpt_codes_in.split(",") if x.strip()],
            "total_cpt_amount": claim_amt_in,
            "clinical_notes": clinical_notes_in,
        }
        with st.spinner("Scoring runtime claim and running coordinator agent..."):
            try:
                rt_res = build_realtime_agent_payload(payload)
            except Exception as e:
                rt_res = {"ok": False, "error": str(e)}

        if not rt_res.get("ok", False):
            st.error(f"Realtime scoring failed: {rt_res.get('error', 'Unknown error')}")
        else:
            st.success(f"Realtime claim `{rt_res.get('claim_id')}` scored successfully.")
            preds = rt_res.get("predictions", {})
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Denial Probability", f"{float(preds.get('denial_probability', 0))*100:.1f}%")
            m2.metric("Mismatch Probability", f"{float(preds.get('mismatch_probability', 0))*100:.1f}%")
            m3.metric("Fraud (Improved)", f"{float(preds.get('fraud_probability_improved', 0))*100:.1f}%")
            m4.metric("Reconciliation Risk", f"{float(preds.get('reconciliation_risk_probability', 0))*100:.1f}%")

            st.markdown("#### Coordinator Recommendation")
            st.info(rt_res.get("recommendation", "No recommendation generated."))

            a1, a2 = st.columns([1.2, 1])
            with a1:
                st.markdown("#### Action Items")
                for item in rt_res.get("action_items", []):
                    st.markdown(f"- {item}")
            with a2:
                st.markdown("#### Key Metrics")
                st.json(rt_res.get("metrics", {}))

            with st.expander("Agent Steps", expanded=False):
                st.json(rt_res.get("steps", []))


# ══════════════════════════════════════════════
#  PAGE — PAYMENT RECONCILIATION
# ══════════════════════════════════════════════
def page_payment_reconciliation():
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
    df["recon_risk_probability"] = ml_engine.score_reconciliation_risk(
        df, recon_model, recon_meta["feature_cols"]
    )
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
        fig = go.Figure(go.Bar(
            x=payer_recon["insurance"],
            y=payer_recon["auto_match"],
            marker_color=COLORS["success"],
            text=payer_recon["auto_match"].round(1).astype(str) + "%",
            textposition="outside",
        ))
        plotly_layout(fig, "Auto-Match Rate by Payer", 330)
        fig.update_yaxes(range=[0, 100], title="Auto-Match %")
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        top_review = df[df["recon_status"] == "Needs Review"][
            ["claim_id", "insurance", "claim_amount", "paid_amount", "posting_gap", "recon_risk_probability"]
        ].sort_values("recon_risk_probability", ascending=False).head(15)
        st.dataframe(top_review, use_container_width=True, hide_index=True)
        if isinstance(recon_meta, dict) and "metrics" in recon_meta:
            st.caption(f"Model metrics: {recon_meta['metrics']}")


# ══════════════════════════════════════════════
#  PAGE — REVENUE FORECASTING
# ══════════════════════════════════════════════
def page_revenue_forecasting():
    st.markdown("# 📈 Revenue Forecasting & What-If")
    st.caption("Trend projection and denial-reduction scenario planning.")

    merged = master.merge(events_tl[["claim_id", "CREATED"]], on="claim_id", how="left")
    merged = merged.dropna(subset=["CREATED"]).copy()
    merged["month"] = merged["CREATED"].dt.to_period("M").astype(str)
    monthly = merged.groupby("month").agg(
        billed=("claim_amount", "sum"),
        collected=("paid_amount", "sum"),
        denied=("is_denied", "mean"),
    ).reset_index()
    monthly["denied"] = monthly["denied"] * 100

    forecast_res = ml_engine.fit_revenue_forecast_model(monthly, value_col="collected")
    forecast_collected = float(forecast_res["next_forecast"])

    k1, k2 = st.columns(2)
    k1.metric("Next-Month Revenue Forecast", fmt_dollar(forecast_collected))
    k2.metric("Avg Monthly Denial Rate", f"{monthly['denied'].mean():.1f}%")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=monthly["month"], y=monthly["billed"], name="Billed", line=dict(color=COLORS["primary"])))
    fig.add_trace(go.Scatter(x=monthly["month"], y=monthly["collected"], name="Collected", line=dict(color=COLORS["accent"])))
    if len(monthly) == len(forecast_res["fitted"]):
        fig.add_trace(go.Scatter(
            x=monthly["month"],
            y=forecast_res["fitted"],
            name="Forecast Trend (Model)",
            line=dict(color=COLORS["warning"], dash="dash"),
        ))
    plotly_layout(fig, "Monthly Revenue Trend", 340)
    st.plotly_chart(fig, use_container_width=True)
    if "metrics" in forecast_res:
        st.caption(f"Forecast model metrics: {forecast_res['metrics']}")

    st.markdown("### What-If Analysis")
    reduction = st.slider("If 'Missing docs' denials reduce by (%)", 0, 80, 50)
    missing_docs_amt = master[master["denial_reason"] == "Missing docs"]["claim_amount"].sum()
    uplift = missing_docs_amt * (reduction / 100.0) * 0.5
    st.success(f"Estimated additional recoverable revenue: **{fmt_dollar(uplift)}**")


# ══════════════════════════════════════════════
#  PAGE — MONITORING & ALERTS
# ══════════════════════════════════════════════
def page_monitoring_alerts():
    st.markdown("# 🖥️ Monitoring & Alerts")
    st.caption("Continuous KPI watchlist for denial, fraud, and payment risk (near real-time simulation).")

    denial_rate = master["is_denied"].mean() * 100
    fraud_high = (master["fraud_score"] > 0.8).mean() * 100
    recon_backlog = ((master["claim_amount"] - master["paid_amount"]).clip(lower=0) > 50).sum()

    alerts = []
    if denial_rate > 8:
        alerts.append(("Denial rate above threshold", "danger"))
    if fraud_high > 3:
        alerts.append(("High fraud-risk pool increased", "warning"))
    if recon_backlog > 5000:
        alerts.append(("Payment reconciliation backlog high", "warning"))
    if not alerts:
        alerts.append(("All core KPIs within expected bands", "success"))

    for txt, lvl in alerts:
        if lvl == "danger":
            st.error(f"🚨 {txt}")
        elif lvl == "warning":
            st.warning(f"⚠️ {txt}")
        else:
            st.success(f"✅ {txt}")

    st.markdown("---")
    monitor_df = pd.DataFrame({
        "Metric": ["Denial Rate %", "Fraud High-Risk %", "Reconciliation Backlog"],
        "Current": [round(denial_rate, 2), round(fraud_high, 2), int(recon_backlog)],
        "Threshold": [8.0, 3.0, 5000],
        "Status": [
            "ALERT" if denial_rate > 8 else "OK",
            "ALERT" if fraud_high > 3 else "OK",
            "ALERT" if recon_backlog > 5000 else "OK",
        ]
    })
    st.dataframe(monitor_df, use_container_width=True, hide_index=True)

# ══════════════════════════════════════════════
#  Page Router
# ══════════════════════════════════════════════
PAGES = {
    "🏠 Executive Summary": page_executive_summary,
    "🧾 Patient Access & Eligibility": page_patient_access_eligibility,
    "🚫 Denial Intelligence": page_denial_intelligence,
    "📋 Appeals Analytics": page_appeals_analytics,
    "🔍 Fraud Detection": page_fraud_detection,
    "🧹 Smart Scrubbing": page_scrubbing,
    "💳 Payment Reconciliation": page_payment_reconciliation,
    "📈 Revenue Forecasting": page_revenue_forecasting,
    "🖥️ Monitoring & Alerts": page_monitoring_alerts,
    "⏱️ AR Aging & Lifecycle": page_ar_aging,
    "🧠 AI Denial Predictor": page_denial_predictor,
    "🤖 Agentic RCM Agent": page_agentic_rcm_agent,
    "💬 LangGraph Chatbot": page_langgraph_chatbot,
}

PAGES[page]()
