"""
11_Agentic_RCM_Agent.py — Agentic RCM Orchestrator with real ML-backed predictions.
"""

import streamlit as st
from utils_ui import shared_page_init, render_page_header, metric_card, fmt_dollar
from src.core.claim_scoring import run_coordinator_for_claim, get_or_train_models
from src.core.data_loader import build_master, get_cpt_summary
from src.core.demo_db import db_signature


def page_agentic_rcm_agent():
    master, _, _ = shared_page_init()

    render_page_header(
        "🤖 Agentic RCM Orchestrator",
        "Multi-stage coordinator over real model scores (denial, scrubbing, fraud, appeals, reconciliation). "
        "Synthetic demo data — recommendations require human review; no payer submission."
    )
    st.caption(
        "Prototype notice: model outputs are experimental estimates on synthetic data. "
        "Appeal drafts are for human review only. The Review Gate step does **not** perform online learning."
    )

    @st.cache_resource(show_spinner="Loading / verifying ML models...")
    def cached_models(db_sig: float):
        m = build_master(db_sig=db_sig)
        cpt = get_cpt_summary(db_sig=db_sig)
        return get_or_train_models(m, cpt, train_if_missing=True)

    try:
        cached_models(db_signature())
    except Exception as e:
        st.error(f"Model preflight failed: {e}")
        return

    st.sidebar.markdown("### 🔍 Execution Context")
    default_id = int(master["claim_id"].iloc[0])
    # Prefer a denied, not-yet-appealed claim when available for a richer demo.
    denied_open = master[(master["is_denied"] == True) & (master["is_appealed"] == False)]  # noqa: E712
    if len(denied_open):
        default_id = int(denied_open["claim_id"].iloc[0])

    claim_id = st.sidebar.number_input("Target Claim ID", min_value=1, step=1, value=default_id)
    force_appeal = st.sidebar.checkbox(
        "Manual demo override: force appeal draft (denied+open claims only)",
        value=False,
        help=(
            "Unauthenticated demo switch — bypasses the model success gate only for denied, "
            "not-yet-appealed claims. Cannot draft for non-denied or resolved claims. "
            "Still requires human review; does not submit to payers."
        ),
    )

    if claim_id not in set(master["claim_id"].astype(int).tolist()):
        st.error("Claim ID not found in the current dataset (historical + live).")
        return

    claim_row = master[master["claim_id"].astype(int) == int(claim_id)].iloc[0].to_dict()

    m1, m2, m3 = st.columns(3)
    with m1:
        metric_card("Claim Amount", fmt_dollar(claim_row.get("claim_amount", 0)))
    with m2:
        status_text = (
            "🔴 Denied"
            if claim_row.get("is_denied")
            else ("🟡 Scrubbing" if not claim_row.get("is_clean_claim") else "🟢 Clean")
        )
        metric_card("Current Status", status_text, "Synthetic dataset flag")
    with m3:
        metric_card("Payer", str(claim_row.get("insurance", "Unknown")), "From claim record")

    with st.expander("Claim context (source of truth for this run)", expanded=False):
        st.json(
            {
                "claim_id": claim_row.get("claim_id"),
                "encounter_id": claim_row.get("encounter_id"),
                "patient_id": claim_row.get("patient_id"),
                "insurance": claim_row.get("insurance"),
                "visit_type": claim_row.get("visit_type"),
                "icd_code": claim_row.get("icd_code"),
                "age": claim_row.get("age"),
                "gender": claim_row.get("gender"),
                "is_denied": claim_row.get("is_denied"),
                "denial_reason": claim_row.get("denial_reason"),
                "is_appealed": claim_row.get("is_appealed"),
                "appeal_success": claim_row.get("appeal_success"),
                "cpt_icd_mismatch": claim_row.get("cpt_icd_mismatch"),
                "high_amount_flag": claim_row.get("high_amount_flag"),
                "strict_insurance_flag": claim_row.get("strict_insurance_flag"),
            }
        )

    if st.button("🚀 Execute Agentic Workflow", use_container_width=True):
        with st.status("🏗️ Orchestrating RCM Agents with live model scores...", expanded=True) as status:
            st.write("Scoring selected claim with loaded models...")
            try:
                result = run_coordinator_for_claim(
                    int(claim_id),
                    force_appeal=bool(force_appeal),
                    train_if_missing=True,
                )
            except Exception as e:
                status.update(label="Workflow failed", state="error")
                st.error(f"Inference / orchestration error: {e}")
                return

            if not result.get("ok"):
                status.update(label="Workflow failed", state="error")
                st.error(result.get("error", "Unknown scoring error"))
                return

            st.write("Coordinator stages: Front-End → Mid-Cycle → Back-End → Review Gate")
            status.update(label="✅ Workflow complete (advisory — human review required)", state="complete")

        preds = result.get("predictions", {})
        p1, p2, p3, p4 = st.columns(4)
        with p1:
            metric_card("Denial Prob", f"{float(preds.get('denial_probability', 0))*100:.1f}%", preds.get("risk_level", ""))
        with p2:
            metric_card("Mismatch Prob", f"{float(preds.get('mismatch_probability', 0))*100:.1f}%")
        with p3:
            metric_card("Fraud (blended)", f"{float(preds.get('fraud_probability_improved', 0))*100:.1f}%")
        with p4:
            metric_card("Recon Risk", f"{float(preds.get('reconciliation_risk_probability', 0))*100:.1f}%")

        st.info(
            "Appeals success / expected recovery are **experimental synthetic-data estimates**, "
            "not externally validated payer outcomes. "
            + str(result.get("persistence_note", ""))
        )

        st.markdown("### 📋 Final Agentic Recommendation")
        st.success(result.get("recommendation", ""))

        st.markdown("#### Action items")
        for item in result.get("action_items", []):
            st.write(f"- {item}")

        coding = preds.get("coding_recommendation") or {}
        nlp = preds.get("nlp_coding_recommendation") or {}
        if coding or nlp:
            st.markdown("#### Coding recommendations (where applicable)")
            c1, c2 = st.columns(2)
            with c1:
                st.caption("Statistical CPT→ICD")
                st.json(coding if coding else {"note": "No coding recommendation"})
            with c2:
                st.caption("Clinical NLP note→ICD")
                st.json(nlp if nlp else {"note": "No NLP recommendation"})

        appeal = result.get("appeal_letter")
        if appeal:
            st.markdown("#### Appeal draft (human review only)")
            st.code(appeal, language="text")
        else:
            st.caption("No appeal draft for this claim under current eligibility rules.")

        st.markdown("#### 🔄 Trace log")
        for s in result.get("steps", []):
            with st.expander(f"📌 {s.get('agent')} | {s.get('step')}"):
                st.write(s.get("summary"))


if __name__ == "__main__":
    page_agentic_rcm_agent()
