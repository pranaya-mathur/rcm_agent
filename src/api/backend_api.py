"""
backend_api.py — Lightweight JSON API for React demo frontend.

Run:
  python3 backend_api.py

Endpoints:
  GET /api/health
  GET /api/summary
  GET /api/agent/claim/<claim_id>
  POST /api/score-claim
"""

from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from src.core.data_loader import build_master, get_cpt_summary, load_all
from src.core.demo_db import upsert_live_claim
from src.core import ml_engine
from src.agents.rcm_agent import CoordinatorAgent
from src.agents.custom_coding_agent import build_cpt_icd_knowledge, build_coding_recommendation
from src.agents.clinical_nlp_agent import train_notes_to_icd_model, predict_icd_from_notes_batch, build_nlp_coding_recommendation


HOST = "0.0.0.0"
PORT = 8001


def _fmt_money(v: float) -> str:
    if v >= 1_000_000:
        return f"${v/1_000_000:.2f}M"
    if v >= 1_000:
        return f"${v/1_000:.1f}K"
    return f"${v:,.0f}"


def _risk_label(p: float) -> str:
    if p >= 0.7:
        return "HIGH"
    if p >= 0.4:
        return "MEDIUM"
    return "LOW"


def build_summary_payload() -> Dict[str, Any]:
    master = build_master().reset_index(drop=True).copy()

    # KPI-like fields from available data
    denial_rate = float(master["is_denied"].mean() * 100)
    clean_claim_rate = float(master["is_clean_claim"].mean() * 100)
    recovered = float(master[master["appeal_success"] == True]["paid_amount"].sum())  # noqa: E712

    fraud_high = float((master["fraud_score"] > 0.8).mean() * 100)

    # Reconciliation risk from trained model if available; fallback to posting gap proxy.
    recon_model, recon_meta = ml_engine.load_reconciliation_risk_model()
    if recon_model is not None and recon_meta is not None:
        recon_probs = ml_engine.score_reconciliation_risk(master, recon_model, recon_meta["feature_cols"])
        recon_high = float((recon_probs >= 0.65).mean() * 100)
    else:
        posting_gap = (master["claim_amount"] - master["paid_amount"]).clip(lower=0)
        recon_high = float((posting_gap > 50).mean() * 100)

    # Scrub bars
    scrub_bars = [
        {"label": "CPT-ICD Mismatch", "v": float(master["cpt_icd_mismatch"].mean() * 100)},
        {"label": "Auth Required Missing", "v": float((master["denial_reason"] == "Auth required").mean() * 100)},
        {"label": "High Amount Flags", "v": float(master["high_amount_flag"].mean() * 100)},
    ]

    # Denial table using historical denial rate by payer as proxy probability
    payer_den = master.groupby("insurance")["is_denied"].mean().to_dict()
    denial_rows: List[Dict[str, Any]] = []
    sample_denied = master.sort_values("claim_amount", ascending=False).head(20)
    for _, r in sample_denied.head(3).iterrows():
        p = float(payer_den.get(r["insurance"], master["is_denied"].mean()))
        risk = _risk_label(p)
        denial_rows.append(
            {
                "claim": f"CLM-{int(r['claim_id'])}",
                "payer": str(r["insurance"]),
                "risk": risk,
                "prob": f"{p*100:.1f}%",
                "action": "Add Auth + Coding QA" if risk == "HIGH" else ("Check docs before submit" if risk == "MEDIUM" else "Proceed standard"),
            }
        )

    # Appeals table from denied and not appealed population (proxy scoring)
    denied = master[master["is_denied"] == True].copy()  # noqa: E712
    not_appealed = denied[denied["is_appealed"] == False].copy()  # noqa: E712
    appeal_rows: List[Dict[str, Any]] = []
    if len(not_appealed) > 0:
        # Simple proxy expected recovery for front-end demo endpoint
        tmp = not_appealed.copy()
        base_success = master[master["is_appealed"] == True]["appeal_success"].mean() if (master["is_appealed"] == True).any() else 0.5  # noqa: E712
        tmp["p_success"] = np.clip(base_success + (tmp["claim_amount"] / tmp["claim_amount"].max()) * 0.15, 0.05, 0.95)
        tmp["expected_recovery"] = tmp["p_success"] * tmp["claim_amount"] * 0.5
        tmp = tmp.sort_values("expected_recovery", ascending=False).head(3)
        for _, r in tmp.iterrows():
            p = float(r["p_success"])
            appeal_rows.append(
                {
                    "claim": f"CLM-{int(r['claim_id'])}",
                    "success": f"{p*100:.0f}%",
                    "recovery": _fmt_money(float(r["expected_recovery"])),
                    "priority": "P1" if p >= 0.7 else "P2",
                }
            )

    payload = {
        "kpis": [
            {"label": "Denial Risk (Avg)", "value": f"{denial_rate:.1f}%", "delta": "-20% target path"},
            {"label": "Clean Claim Rate", "value": f"{clean_claim_rate:.1f}%", "delta": "towards 95%+"},
            {"label": "Appeal Recovery", "value": _fmt_money(recovered), "delta": "ML-prioritized opportunity"},
            {"label": "Reconciliation High-Risk", "value": f"{recon_high:.1f}%", "delta": "model-based posting risk"},
        ],
        "scrubBars": scrub_bars,
        "denialRows": denial_rows,
        "appealRows": appeal_rows,
        "meta": {"claims_count": int(len(master)), "fraud_high_risk_pct": round(fraud_high, 2)},
    }
    return payload


def _safe_float(v: Any, default: float = 0.0) -> float:
    try:
        if v is None:
            return default
        f = float(v)
        if np.isnan(f):
            return default
        return f
    except Exception:
        return default


def _build_runtime_claim_row(payload: Dict[str, Any], master: pd.DataFrame) -> Dict[str, Any]:
    """
    Build a model-compatible claim row by starting from an existing template row.
    This allows real-time payload scoring while preserving required columns.
    """
    template = master.iloc[0].to_dict() if len(master) > 0 else {}
    claim = dict(template)

    # Keep defaults from historical template, but allow explicit runtime overrides.
    claim["claim_id"] = int(payload.get("claim_id", 99_999_999))
    claim["encounter_id"] = int(payload.get("encounter_id", claim.get("encounter_id", 99_999_999)))
    claim["patient_id"] = int(payload.get("patient_id", claim.get("patient_id", 99_999_999)))
    claim["insurance"] = str(payload.get("insurance", claim.get("insurance", "Unknown")))
    claim["visit_type"] = str(payload.get("visit_type", claim.get("visit_type", "OP")))
    claim["icd_code"] = str(payload.get("icd_code", claim.get("icd_code", "Unknown")))
    claim["gender"] = str(payload.get("gender", claim.get("gender", "U")))
    claim["age"] = int(payload.get("age", claim.get("age", 40)))
    claim["claim_amount"] = _safe_float(payload.get("claim_amount", claim.get("claim_amount", 0.0)), 0.0)
    claim["paid_amount"] = _safe_float(payload.get("paid_amount", claim.get("paid_amount", 0.0)), 0.0)
    claim["fraud_score"] = _safe_float(payload.get("fraud_score", claim.get("fraud_score", 0.0)), 0.0)
    claim["denial_reason"] = str(payload.get("denial_reason", claim.get("denial_reason", "None")))

    claim["is_denied"] = bool(payload.get("is_denied", claim.get("is_denied", False)))
    claim["is_appealed"] = bool(payload.get("is_appealed", claim.get("is_appealed", False)))
    claim["appeal_success"] = bool(payload.get("appeal_success", claim.get("appeal_success", False)))
    claim["is_clean_claim"] = bool(payload.get("is_clean_claim", claim.get("is_clean_claim", True)))
    claim["cpt_icd_mismatch"] = bool(payload.get("cpt_icd_mismatch", claim.get("cpt_icd_mismatch", False)))
    claim["high_amount_flag"] = bool(payload.get("high_amount_flag", claim.get("high_amount_flag", claim["claim_amount"] > 1500)))
    claim["strict_insurance_flag"] = bool(payload.get("strict_insurance_flag", claim.get("strict_insurance_flag", False)))

    return claim


def _build_runtime_cpt_summary_row(
    payload: Dict[str, Any],
    claim_id: int,
    claim_amount: float,
    cpt_summary: pd.DataFrame,
) -> Dict[str, Any]:
    """
    Build a cpt_summary-like row for a runtime claim payload.
    """
    template = cpt_summary.iloc[0].to_dict() if len(cpt_summary) > 0 else {}
    row = dict(template)
    row["claim_id"] = int(claim_id)

    cpt_codes_raw = payload.get("cpt_codes", [])
    cpt_codes = [str(x).strip() for x in cpt_codes_raw if str(x).strip()]
    num_cpt_codes = int(payload.get("num_cpt_codes", len(cpt_codes) if len(cpt_codes) > 0 else 1))
    total_cpt_amount = _safe_float(payload.get("total_cpt_amount", claim_amount), claim_amount)

    row["num_cpt_codes"] = num_cpt_codes
    row["total_cpt_amount"] = total_cpt_amount
    row["cpt_codes_list"] = ",".join(cpt_codes)

    # Populate known dynamic cpt_<code>_count columns if present in current summary schema.
    for code in cpt_codes:
        col = f"cpt_{code}_count"
        if col in cpt_summary.columns:
            row[col] = row.get(col, 0) + 1

    return row


def build_realtime_agent_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Score a runtime claim payload (without requiring it in CSV data) and run coordinator agent.

    Expected payload keys (minimum practical):
      claim_id, insurance, visit_type, icd_code, gender, age, claim_amount
    Optional:
      paid_amount, fraud_score, denial_reason, is_denied, is_appealed, appeal_success,
      cpt_codes (list[str]), total_cpt_amount, num_cpt_codes, clinical_notes
    """
    master = build_master().reset_index(drop=True).copy()
    cpt_summary = get_cpt_summary().reset_index(drop=True).copy()
    raw = load_all()

    if len(master) == 0:
        return {"ok": False, "error": "Master dataset is empty. Cannot score runtime claim."}

    runtime_claim = _build_runtime_claim_row(payload, master)
    runtime_cpt = _build_runtime_cpt_summary_row(
        payload=payload,
        claim_id=int(runtime_claim["claim_id"]),
        claim_amount=_safe_float(runtime_claim["claim_amount"], 0.0),
        cpt_summary=cpt_summary,
    )

    # Append runtime row to existing batch so feature pipelines stay unchanged.
    runtime_claim_df = pd.DataFrame([runtime_claim]).reindex(columns=master.columns)
    runtime_cpt_df_full = pd.DataFrame([runtime_cpt]).reindex(columns=cpt_summary.columns)
    master_ext = master.copy()
    master_ext.loc[len(master_ext)] = runtime_claim_df.iloc[0]
    cpt_ext = cpt_summary.copy()
    cpt_ext.loc[len(cpt_ext)] = runtime_cpt_df_full.iloc[0]
    sel_idx = len(master_ext) - 1

    denial_model, denial_meta = ml_engine.load_model()
    if denial_model is None:
        tmp = ml_engine.train_model(master, cpt_summary)
        denial_model = tmp["model"]
        denial_meta = {"feature_cols": tmp["feature_cols"]}
    denial_pred = ml_engine.predict_single_claim(runtime_claim, denial_model, denial_meta["feature_cols"])

    mismatch_model, mismatch_meta = ml_engine.load_mismatch_model()
    if mismatch_model is None:
        tmp = ml_engine.train_mismatch_model(master, cpt_summary)
        mismatch_model, mismatch_meta = tmp["model"], tmp["meta"]
    mismatch_probs = ml_engine.score_all_mismatch(master_ext, cpt_ext, mismatch_model, mismatch_meta["feature_cols"])

    appeals_success_model, appeals_success_meta = ml_engine.load_appeals_success_model()
    if appeals_success_model is None:
        tmp = ml_engine.train_appeals_success_model(master, cpt_summary)
        appeals_success_model, appeals_success_meta = tmp["model"], tmp["meta"]
    appeals_recovery_model, appeals_recovery_meta = ml_engine.load_appeals_recovery_model()
    if appeals_recovery_model is None:
        tmp = ml_engine.train_appeals_recovery_model(master, cpt_summary)
        appeals_recovery_model, appeals_recovery_meta = tmp["model"], tmp["meta"]

    fraud_prob_model, fraud_prob_meta = ml_engine.load_fraud_probability_model()
    if fraud_prob_model is None:
        tmp = ml_engine.train_fraud_probability_model(master, cpt_summary)
        fraud_prob_model, fraud_prob_meta = tmp["model"], tmp["meta"]
    fraud_anomaly_model, fraud_anomaly_meta = ml_engine.load_fraud_anomaly_model()
    if fraud_anomaly_model is None:
        tmp = ml_engine.train_fraud_anomaly_model(master)
        fraud_anomaly_model, fraud_anomaly_meta = tmp["model"], tmp["meta"]
    fraud_df = ml_engine.score_fraud_enhanced(
        master_ext,
        cpt_ext,
        fraud_prob_model,
        fraud_prob_meta["feature_cols"],
        fraud_anomaly_model,
        fraud_anomaly_meta,
        alpha=0.6,
    )

    recon_model, recon_meta = ml_engine.load_reconciliation_risk_model()
    if recon_model is None:
        tmp = ml_engine.train_reconciliation_risk_model(master)
        recon_model, recon_meta = tmp["model"], tmp["meta"]
    recon_probs = ml_engine.score_reconciliation_risk(master_ext, recon_model, recon_meta["feature_cols"])

    # Appeals inference for this one claim by passing single-row frame.
    runtime_df = runtime_claim_df.copy()
    runtime_cpt_df = runtime_cpt_df_full.copy()
    ranked = ml_engine.predict_appeals_ranked_candidates(
        runtime_df,
        runtime_cpt_df,
        appeals_success_model,
        appeals_success_meta["feature_cols"],
        appeals_recovery_model,
        appeals_recovery_meta["feature_cols"],
    )
    p_appeal_success = _safe_float(ranked.iloc[0]["p_appeal_success"], 0.0) if len(ranked) > 0 else 0.0
    expected_recovery = _safe_float(ranked.iloc[0]["expected_recovery"], 0.0) if len(ranked) > 0 else 0.0

    coding_knowledge = build_cpt_icd_knowledge(raw["cpt_lines"], raw["icd"], raw["claims"])
    cpt_codes = [str(x).strip() for x in payload.get("cpt_codes", []) if str(x).strip()]
    coding_reco = build_coding_recommendation(
        runtime_claim,
        cpt_codes,
        coding_knowledge,
        _safe_float(mismatch_probs[sel_idx], 0.0),
        min_support=20,
    )

    nlp_model_bundle = train_notes_to_icd_model(raw["encounters"], raw["claims"], raw["icd"], min_samples=50)
    clinical_notes = str(payload.get("clinical_notes", "") or "")
    note_pred = predict_icd_from_notes_batch([clinical_notes], nlp_model_bundle, top_k=3)[0]
    nlp_coding_reco = build_nlp_coding_recommendation(
        runtime_claim,
        note_pred,
        _safe_float(mismatch_probs[sel_idx], 0.0),
    )

    predictions = {
        "denial_probability": _safe_float(denial_pred.get("denial_probability", 0.0), 0.0),
        "mismatch_probability": _safe_float(mismatch_probs[sel_idx], 0.0),
        "fraud_probability_improved": _safe_float(fraud_df.loc[sel_idx, "fraud_probability_improved"], 0.0),
        "reconciliation_risk_probability": _safe_float(recon_probs[sel_idx], 0.0),
        "coding_recommendation": coding_reco,
        "nlp_coding_recommendation": nlp_coding_reco,
        "p_appeal_success": p_appeal_success,
        "expected_recovery": expected_recovery,
    }

    agent = CoordinatorAgent()
    agent_out = agent.run(claim=runtime_claim, predictions=predictions)
    steps = [{"agent": s.agent, "step": s.step, "summary": s.summary} for s in agent_out.steps]
    return {
        "ok": True,
        "mode": "runtime_payload_inference",
        "claim_id": int(runtime_claim["claim_id"]),
        "recommendation": agent_out.recommendation,
        "action_items": agent_out.action_items,
        "appeal_letter": agent_out.appeal_letter,
        "metrics": agent_out.metrics,
        "predictions": predictions,
        "steps": steps,
    }


def _clear_data_caches() -> None:
    # Ensure new persisted claims appear in Streamlit/API reads.
    try:
        load_all.clear()
    except Exception:
        pass
    try:
        build_master.clear()
    except Exception:
        pass
    try:
        get_cpt_summary.clear()
    except Exception:
        pass


def build_agent_claim_payload(claim_id: int) -> Dict[str, Any]:
    master = build_master().reset_index(drop=True).copy()
    if claim_id not in set(master["claim_id"].astype(int).tolist()):
        return {"ok": False, "error": f"Claim ID {claim_id} not found"}

    raw = load_all()
    cpt_summary = get_cpt_summary()

    # Core models
    denial_model, denial_meta = ml_engine.load_model()
    if denial_model is None:
        denial_res = ml_engine.train_model(master, cpt_summary)
        denial_model = denial_res["model"]
        denial_meta = {"feature_cols": denial_res["feature_cols"], "all_probabilities": denial_res["all_probabilities"]}

    mismatch_model, mismatch_meta = ml_engine.load_mismatch_model()
    if mismatch_model is None:
        mismatch_res = ml_engine.train_mismatch_model(master, cpt_summary)
        mismatch_model, mismatch_meta = mismatch_res["model"], mismatch_res["meta"]

    appeals_success_model, appeals_success_meta = ml_engine.load_appeals_success_model()
    if appeals_success_model is None:
        tmp = ml_engine.train_appeals_success_model(master, cpt_summary)
        appeals_success_model, appeals_success_meta = tmp["model"], tmp["meta"]

    appeals_recovery_model, appeals_recovery_meta = ml_engine.load_appeals_recovery_model()
    if appeals_recovery_model is None:
        tmp = ml_engine.train_appeals_recovery_model(master, cpt_summary)
        appeals_recovery_model, appeals_recovery_meta = tmp["model"], tmp["meta"]

    fraud_prob_model, fraud_prob_meta = ml_engine.load_fraud_probability_model()
    if fraud_prob_model is None:
        tmp = ml_engine.train_fraud_probability_model(master, cpt_summary)
        fraud_prob_model, fraud_prob_meta = tmp["model"], tmp["meta"]

    fraud_anomaly_model, fraud_anomaly_meta = ml_engine.load_fraud_anomaly_model()
    if fraud_anomaly_model is None:
        tmp = ml_engine.train_fraud_anomaly_model(master)
        fraud_anomaly_model, fraud_anomaly_meta = tmp["model"], tmp["meta"]

    recon_model, recon_meta = ml_engine.load_reconciliation_risk_model()
    if recon_model is None:
        tmp = ml_engine.train_reconciliation_risk_model(master)
        recon_model, recon_meta = tmp["model"], tmp["meta"]

    # Batch scores (for aligned indexing)
    denial_probs = np.array(
        ml_engine.score_denial_portfolio(master, cpt_summary, denial_model, denial_meta["feature_cols"]),
        dtype=float,
    )
    mismatch_probs = ml_engine.score_all_mismatch(master, cpt_summary, mismatch_model, mismatch_meta["feature_cols"])
    fraud_df = ml_engine.score_fraud_enhanced(
        master, cpt_summary, fraud_prob_model, fraud_prob_meta["feature_cols"], fraud_anomaly_model, fraud_anomaly_meta, alpha=0.6
    )
    recon_probs = ml_engine.score_reconciliation_risk(master, recon_model, recon_meta["feature_cols"])

    # Appeals scores only for denied + not appealed set
    denied = master[master["is_denied"]].copy()
    not_appealed = denied[~denied["is_appealed"]].copy()
    appeal_map: Dict[int, Dict[str, float]] = {}
    if len(not_appealed) > 0:
        ranked = ml_engine.predict_appeals_ranked_candidates(
            not_appealed,
            cpt_summary,
            appeals_success_model,
            appeals_success_meta["feature_cols"],
            appeals_recovery_model,
            appeals_recovery_meta["feature_cols"],
        )
        for _, r in ranked.iterrows():
            cid = int(r["claim_id"])
            appeal_map[cid] = {
                "p_appeal_success": _safe_float(r["p_appeal_success"], 0.0),
                "expected_recovery": _safe_float(r["expected_recovery"], 0.0),
            }

    # Coding + NLP
    coding_knowledge = build_cpt_icd_knowledge(raw["cpt_lines"], raw["icd"], raw["claims"])
    nlp_model_bundle = train_notes_to_icd_model(raw["encounters"], raw["claims"], raw["icd"], min_samples=50)
    encounter_notes_map = raw["encounters"].set_index("encounter_id")["clinical_notes"].to_dict()
    notes_batch = [encounter_notes_map.get(eid, "") for eid in master["encounter_id"].tolist()]
    note_preds_batch = predict_icd_from_notes_batch(notes_batch, nlp_model_bundle, top_k=3)

    # Find selected row index
    sel_idx = int(master.index[master["claim_id"].astype(int) == int(claim_id)][0])
    row = master.iloc[sel_idx].to_dict()

    cpt_row = cpt_summary[cpt_summary["claim_id"] == int(claim_id)]
    cpt_codes = []
    if len(cpt_row) > 0:
        cpt_codes_str = str(cpt_row.iloc[0].get("cpt_codes_list", ""))
        cpt_codes = [x.strip() for x in cpt_codes_str.split(",") if x.strip()]

    coding_reco = build_coding_recommendation(
        row,
        cpt_codes,
        coding_knowledge,
        _safe_float(mismatch_probs[sel_idx], 0.0),
        min_support=20,
    )
    nlp_coding_reco = build_nlp_coding_recommendation(
        row,
        note_preds_batch[sel_idx] if sel_idx < len(note_preds_batch) else {},
        _safe_float(mismatch_probs[sel_idx], 0.0),
    )

    predictions = {
        "denial_probability": _safe_float(denial_probs[sel_idx], 0.0),
        "mismatch_probability": _safe_float(mismatch_probs[sel_idx], 0.0),
        "fraud_probability_improved": _safe_float(fraud_df.loc[sel_idx, "fraud_probability_improved"], 0.0),
        "reconciliation_risk_probability": _safe_float(recon_probs[sel_idx], 0.0),
        "coding_recommendation": coding_reco,
        "nlp_coding_recommendation": nlp_coding_reco,
        "p_appeal_success": _safe_float(appeal_map.get(int(claim_id), {}).get("p_appeal_success", 0.0), 0.0),
        "expected_recovery": _safe_float(appeal_map.get(int(claim_id), {}).get("expected_recovery", 0.0), 0.0),
    }

    agent = CoordinatorAgent()
    agent_out = agent.run(claim=row, predictions=predictions)

    steps = [{"agent": s.agent, "step": s.step, "summary": s.summary} for s in agent_out.steps]
    return {
        "ok": True,
        "claim_id": int(claim_id),
        "recommendation": agent_out.recommendation,
        "action_items": agent_out.action_items,
        "appeal_letter": agent_out.appeal_letter,
        "metrics": agent_out.metrics,
        "steps": steps,
    }


class Handler(BaseHTTPRequestHandler):
    def _set_headers(self, status: int = 200) -> None:
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_OPTIONS(self):
        self._set_headers(200)
        self.wfile.write(b"{}")

    def do_GET(self):
        if self.path == "/api/health":
            self._set_headers(200)
            self.wfile.write(json.dumps({"ok": True}).encode("utf-8"))
            return
        if self.path == "/api/summary":
            try:
                payload = build_summary_payload()
                self._set_headers(200)
                self.wfile.write(json.dumps(payload).encode("utf-8"))
            except Exception as e:
                self._set_headers(500)
                self.wfile.write(json.dumps({"ok": False, "error": str(e)}).encode("utf-8"))
            return
        if self.path.startswith("/api/agent/claim/"):
            try:
                claim_id_str = self.path.split("/api/agent/claim/")[1].strip("/")
                claim_id = int(claim_id_str)
                payload = build_agent_claim_payload(claim_id)
                status = 200 if payload.get("ok", False) else 404
                self._set_headers(status)
                self.wfile.write(json.dumps(payload).encode("utf-8"))
            except Exception as e:
                self._set_headers(500)
                self.wfile.write(json.dumps({"ok": False, "error": str(e)}).encode("utf-8"))
            return

        self._set_headers(404)
        self.wfile.write(json.dumps({"ok": False, "error": "Not found"}).encode("utf-8"))

    def do_POST(self):
        if self.path == "/api/score-claim":
            try:
                content_len = int(self.headers.get("Content-Length", "0"))
                raw_body = self.rfile.read(content_len) if content_len > 0 else b"{}"
                payload = json.loads(raw_body.decode("utf-8"))
                if not isinstance(payload, dict):
                    self._set_headers(400)
                    self.wfile.write(json.dumps({"ok": False, "error": "JSON payload must be an object"}).encode("utf-8"))
                    return

                persist = bool(payload.get("persist", False))
                payload_no_flag = {k: v for k, v in payload.items() if k != "persist"}
                result = build_realtime_agent_payload(payload)
                if persist and result.get("ok", False):
                    upsert_live_claim(payload_no_flag)
                    _clear_data_caches()
                    result["persisted"] = True
                else:
                    result["persisted"] = False
                status = 200 if result.get("ok", False) else 400
                self._set_headers(status)
                self.wfile.write(json.dumps(result).encode("utf-8"))
                return
            except Exception as e:
                self._set_headers(500)
                self.wfile.write(json.dumps({"ok": False, "error": str(e)}).encode("utf-8"))
                return

        self._set_headers(404)
        self.wfile.write(json.dumps({"ok": False, "error": "Not found"}).encode("utf-8"))


if __name__ == "__main__":
    server = ThreadingHTTPServer((HOST, PORT), Handler)
    print(f"RCM API running on http://{HOST}:{PORT}")
    server.serve_forever()
