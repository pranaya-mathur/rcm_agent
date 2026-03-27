"""
3_AI_Denial_Predictor.py — AI Denial Predictor & Risk Analysis with Premium UI
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from utils_ui import (
    shared_page_init, plotly_layout, COLORS, 
    fmt_dollar, render_page_header, metric_card
)
from src.core import ml_engine
from src.core.data_loader import get_cpt_summary

def page_denial_predictor():
    master, _, _ = shared_page_init()
    
    render_page_header(
        "🧠 AI Denial Predictor & Risk Analysis",
        "Predictive denial management powered by Gradient Boosted Trees (XGBoost) and SHAP explainability."
    )

    @st.cache_resource(show_spinner=False)
    def cached_model():
        model, meta = ml_engine.load_model()
        if model is None:
            with st.spinner("🚀 Training AI Predictive Engine..."):
                results = ml_engine.train_model(master, get_cpt_summary())
                return results["model"], results
        return model, meta

    model, meta = cached_model()

    st.markdown("<br>", unsafe_allow_html=True)
    tab1, tab2, tab3 = st.tabs(["🎯 Interactive Predictor", "📈 Portfolio Risk", "⚙️ Model Health"])

    with tab1:
        st.markdown("### 🔍 Real-Time Claim Simulation")
        with st.form("prediction_form", border=False):
            c1, c2, c3 = st.columns(3)
            with c1:
                ins = st.selectbox("Payer", ["BCBS", "Humana", "UHC", "Cigna", "Aetna"])
                visit = st.selectbox("Visit Type", ["ER", "OPD", "IPD"])
                icd = st.selectbox("ICD Code", ["K21", "I10", "N39", "J45", "E11", "M54"])
            with c2:
                amt = st.slider("Claim Amount ($)", 100, 25000, 5000)
                age = st.slider("Patient Age", 0, 100, 45)
                gender = st.selectbox("Gender", ["M", "F"])
            with c3:
                scrub1 = st.checkbox("CPT-ICD Mismatch Detected")
                scrub2 = st.checkbox("High Amount Flagged")
                scrub3 = st.checkbox("Strict Insurance Rules Applied")
                fraud_f = st.slider("Initial Fraud Score", 0.0, 1.0, 0.3)

            submitted = st.form_submit_button("⚡ Execute AI Risk Analysis", use_container_width=True)

        if submitted:
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
            pc1, pc2 = st.columns([1, 1.5])
            with pc1:
                metric_card("Analysis Result", f"{prob*100:.1f}%", f"Risk: {risk}", delta_up=False if prob > 0.5 else True)
            with pc2:
                st.markdown("#### 🧬 SHAP Genetic Drivers")
                shap_df = res["shap_explanation"]
                fig = go.Figure(go.Bar(
                    x=shap_df["shap_value"], y=shap_df["feature"], orientation="h",
                    marker_color=[COLORS["danger"] if x > 0 else COLORS["success"] for x in shap_df["shap_value"]]
                ))
                plotly_layout(fig, "", 250)
                st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.markdown("### 📊 Portfolio Probability Distribution")
        all_probs = meta["all_probabilities"]
        fig = px.histogram(all_probs, nbins=50, color_discrete_sequence=[COLORS["secondary"]])
        plotly_layout(fig, "Claim Denial Dispersion", 400)
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Distribution of predicted denial probabilities across the active claim portfolio.")

    with tab3:
        st.markdown("### 🛠️ Model Performance Metrics")
        m = meta["metrics"]
        mc1, mc2, mc3 = st.columns(3)
        with mc1:
             metric_card("ROC-AUC", f"{m['auc']:.3f}", "Classification Power")
        with mc2:
             metric_card("Precision", f"{m['precision']*100:.1f}%", "Avoidance Accuracy")
        with mc3:
             metric_card("Recall", f"{m['recall']*100:.1f}%", "Sensitivity")

if __name__ == "__main__":
    page_denial_predictor()
