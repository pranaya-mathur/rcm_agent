"""
utils_ui.py — Shared UI components, helpers, and data loading for Streamlit pages.
Revised for premium enterprise aesthetics.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import os
import sys

# Ensure the root directory is on the path for package-style imports
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.core.data_loader import load_all, build_master, build_events_timeline, get_cpt_summary

# ──────────────────────────────────────────────
#  UI Colors & Theme
# ──────────────────────────────────────────────
COLORS = {
    "primary": "#6366f1",
    "secondary": "#0ea5e9",
    "accent": "#22d3ee",
    "success": "#10b981",
    "warning": "#f59e0b",
    "danger": "#ef4444",
    "bg": "#0f172a",
    "card": "rgba(30, 41, 59, 0.7)",
    "text": "#f1f5f9",
    "muted": "#94a3b8",
}

PALETTE = ["#6366f1", "#0ea5e9", "#10b981", "#f59e0b", "#ef4444", "#8b5cf6", "#ec4899"]

# ──────────────────────────────────────────────
#  Helpers
# ──────────────────────────────────────────────

def load_css():
    css_path = os.path.join(os.path.dirname(__file__), "style.css")
    if os.path.exists(css_path):
        with open(css_path) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

def plotly_layout(fig, title="", height=400):
    fig.update_layout(
        title=dict(text=title, font=dict(size=16, color=COLORS["text"], family="Outfit")),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter", color=COLORS["muted"], size=12),
        height=height,
        margin=dict(l=40, r=20, t=50, b=40),
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            font=dict(color=COLORS["muted"], size=11),
        ),
        xaxis=dict(gridcolor="rgba(148, 163, 184, 0.1)", zerolinecolor="rgba(148, 163, 184, 0.1)"),
        yaxis=dict(gridcolor="rgba(148, 163, 184, 0.1)", zerolinecolor="rgba(148, 163, 184, 0.1)"),
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

def render_page_header(title: str, subtitle: str):
    st.markdown(
        f"""
        <div class="command-header animate-fade">
            <div class="header-title">{title}</div>
            <div class="header-subtitle">{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

def metric_card(label: str, value: str, delta: str = "", delta_up: bool = True):
    delta_class = "delta-up" if delta_up else "delta-down"
    delta_arrow = "↑" if delta_up else "↓"
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-label">{label}</div>
            <div class="kpi-value">{value}</div>
            {f'<div class="kpi-delta {delta_class}">{delta_arrow} {delta}</div>' if delta else ""}
        </div>
        """,
        unsafe_allow_html=True,
    )

@st.cache_data(show_spinner=False)
def get_data():
    master = build_master()
    events = build_events_timeline()
    raw = load_all()
    return master, events, raw

def sidebar_status(master):
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"""
    <div style="padding:1rem; background:rgba(99, 102, 241, 0.08); border-radius:16px;
                border:1px solid rgba(99, 102, 241, 0.2); font-size:12px; color:#94A3B8;">
        <div style="margin-bottom:10px;">
            <span class="pulse-dot"></span> 
            <strong style="color:#f1f5f9; margin-left:8px;">System Integrity Active</strong>
        </div>
        <div style="display:flex; justify-content:space-between; margin-top:8px;">
            <span>Claims Logged:</span>
            <span style="color:#f1f5f9; font-weight:600;">{fmt_number(len(master))}</span>
        </div>
        <div style="display:flex; justify-content:space-between;">
            <span>Active Payers:</span>
            <span style="color:#f1f5f9; font-weight:600;">{master['insurance'].nunique()}</span>
        </div>
        <div style="margin-top:12px; height:4px; background:rgba(255,255,255,0.05); border-radius:2px;">
            <div style="width:100%; height:100%; background:var(--success); border-radius:2px;"></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

def sidebar_branding():
    st.sidebar.markdown("""
    <div style="text-align:center; padding: 1rem 0 0.5rem 0;">
        <div style="font-size:3rem; margin-bottom: 0.5rem;">🏢</div>
        <h2 style="margin:0.2rem 0; font-family:'Outfit'; font-size:1.6rem; border:none; padding:0;
            background: linear-gradient(135deg, #f1f5f9, #94a3b8);
            -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
            Smart RCM
        </h2>
        <div style="height:1px; background:linear-gradient(90deg, transparent, rgba(99, 102, 241, 0.4), transparent); margin: 0.5rem 0;"></div>
        <p style="color:#64748B; font-size:0.75rem; letter-spacing:0.1em; text-transform:uppercase;">Enterprise Intelligence</p>
    </div>
    """, unsafe_allow_html=True)

def shared_page_init():
    load_css()
    sidebar_branding()
    with st.spinner("🔄 Synchronizing RCM Assets..."):
        master, events_tl, raw = get_data()
    sidebar_status(master)
    return master, events_tl, raw
