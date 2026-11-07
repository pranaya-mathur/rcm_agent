"""
claim_scoring.py — Shared claim-level ML scoring for Streamlit pages and the API.

Centralizes model load/train, portfolio scoring, and runtime payload scoring so
pages do not hardcode predictions or duplicate ML logic.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.core import ml_engine
from src.core.data_loader import build_master, get_cpt_summary, load_all
from src.core.demo_db import db_signature
from src.core.value_coercion import coerce_bool, coerce_finite_number, try_coerce_bool, CoercionError
from src.agents.custom_coding_agent import build_cpt_icd_knowledge, build_coding_recommendation
from src.agents.clinical_nlp_agent import (
    train_notes_to_icd_model,
    predict_icd_from_notes_batch,
    build_nlp_coding_recommendation,
)
from src.agents.rcm_agent import CoordinatorAgent, AgentOutput


REQUIRED_PAYLOAD_FIELDS = ("claim_id", "insurance", "visit_type", "icd_code", "gender", "age", "claim_amount")
BOOL_PAYLOAD_FIELDS = (
    "is_denied",
    "is_appealed",
    "appeal_success",
    "cpt_icd_mismatch",
    "high_amount_flag",
    "strict_insurance_flag",
    "persist",
    "force_appeal",
)


def safe_float(v: Any, default: float = 0.0) -> float:
    try:
        if v is None:
            return default
        f = float(v)
        if np.isnan(f) or np.isinf(f):
            return default
        return f
    except Exception:
        return default


def validate_claim_payload(payload: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """Validate realtime claim JSON. Returns (ok, errors)."""
    errors: List[str] = []
    if not isinstance(payload, dict):
        return False, ["JSON payload must be an object"]

    for field in REQUIRED_PAYLOAD_FIELDS:
        if field not in payload or payload[field] in (None, ""):
            errors.append(f"Missing required field: {field}")

    if "claim_id" in payload:
        try:
            int(payload["claim_id"])
        except Exception:
            errors.append("claim_id must be an integer")

    # Strict numeric checks (finite; non-negative financials).
    for num_field, kwargs in (
        ("claim_amount", {"allow_negative": False, "minimum": 0.0}),
        ("paid_amount", {"allow_negative": False, "minimum": 0.0}),
        ("fraud_score", {"allow_negative": False, "minimum": 0.0, "maximum": 1.0}),
        ("total_cpt_amount", {"allow_negative": False, "minimum": 0.0}),
        ("num_cpt_codes", {"allow_negative": False, "minimum": 0.0}),
        ("age", {"allow_negative": False, "minimum": 0.0, "maximum": 120.0}),
    ):
        if num_field in payload and payload[num_field] is not None and payload[num_field] != "":
            try:
                coerce_finite_number(payload[num_field], field=num_field, **kwargs)
            except CoercionError as e:
                errors.append(str(e))

    for bool_field in BOOL_PAYLOAD_FIELDS:
        if bool_field in payload and payload[bool_field] is not None:
            _, err = try_coerce_bool(payload[bool_field], field=bool_field)
            if err:
                errors.append(err)

    if "cpt_codes" in payload and payload["cpt_codes"] is not None:
        if not isinstance(payload["cpt_codes"], list):
            errors.append("cpt_codes must be a list")

    if "visit_type" in payload and payload["visit_type"] not in (None, ""):
        vt = str(payload["visit_type"]).upper()
        if vt not in ("OP", "OPD", "ER", "IP", "IPD", "INPATIENT", "OUTPATIENT"):
            errors.append("visit_type should be one of: OPD, ER, IPD (aliases OP/IP accepted)")

    return len(errors) == 0, errors


def normalize_visit_type(visit_type: str) -> str:
    vt = str(visit_type or "OPD").upper().strip()
    aliases = {"OP": "OPD", "OUTPATIENT": "OPD", "IP": "IPD", "INPATIENT": "IPD"}
    return aliases.get(vt, vt)


def derive_claim_fields(claim: Dict[str, Any]) -> Dict[str, Any]:
    """Compute derived financial/scrub fields for a claim dict."""
    out = dict(claim)
    amount = safe_float(out.get("claim_amount"), 0.0)
    paid = safe_float(out.get("paid_amount"), 0.0)
    out["claim_amount"] = amount
    out["paid_amount"] = paid
    out["revenue_leakage"] = amount - paid
    out["collection_rate"] = round((paid / amount * 100), 2) if amount > 0 else 0.0
    if "high_amount_flag" not in out or out.get("high_amount_flag") is None:
        out["high_amount_flag"] = amount > 1500
    for flag in ("cpt_icd_mismatch", "strict_insurance_flag", "high_amount_flag"):
        if isinstance(out.get(flag), bool):
            continue
        if out.get(flag) is None:
            out[flag] = False
        else:
            out[flag] = coerce_bool(out.get(flag), field=flag, default=False)
    out["is_clean_claim"] = not any(
        out.get(f, False) for f in ("cpt_icd_mismatch", "high_amount_flag", "strict_insurance_flag")
    )
    age = int(out.get("age", 0) or 0)
    if age <= 18:
        out["age_group"] = "0-18"
    elif age <= 35:
        out["age_group"] = "19-35"
    elif age <= 50:
        out["age_group"] = "36-50"
    elif age <= 65:
        out["age_group"] = "51-65"
    else:
        out["age_group"] = "65+"
    out["visit_type"] = normalize_visit_type(str(out.get("visit_type", "OPD")))
    return out


def build_runtime_claim_row(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build a model-compatible claim row from payload alone.
    Does NOT inherit unspecified attributes from an unrelated historical claim.
    """
    claim_amount = coerce_finite_number(payload.get("claim_amount"), field="claim_amount", allow_negative=False)
    paid_amount = (
        coerce_finite_number(payload.get("paid_amount"), field="paid_amount", allow_negative=False)
        if payload.get("paid_amount") is not None
        else 0.0
    )
    fraud_score = (
        coerce_finite_number(payload.get("fraud_score"), field="fraud_score", allow_negative=False, minimum=0.0, maximum=1.0)
        if payload.get("fraud_score") is not None
        else 0.0
    )
    age = int(coerce_finite_number(payload.get("age"), field="age", allow_negative=False, minimum=0.0, maximum=120.0))

    high_amount_flag = None
    if "high_amount_flag" in payload and payload.get("high_amount_flag") is not None:
        high_amount_flag = coerce_bool(payload.get("high_amount_flag"), field="high_amount_flag")

    claim = {
        "claim_id": int(payload["claim_id"]),
        "encounter_id": int(payload.get("encounter_id") or (900_000_000 + int(payload["claim_id"]))),
        "patient_id": int(payload.get("patient_id") or (800_000_000 + int(payload["claim_id"]))),
        "insurance": str(payload.get("insurance", "Unknown")),
        "insurance_pat": str(payload.get("insurance", "Unknown")),
        "visit_type": normalize_visit_type(str(payload.get("visit_type", "OPD"))),
        "icd_code": str(payload.get("icd_code", "Unknown")),
        "gender": str(payload.get("gender", "U")),
        "age": age,
        "claim_amount": claim_amount,
        "paid_amount": paid_amount,
        "fraud_score": fraud_score,
        "denial_reason": str(payload.get("denial_reason", "None") or "None"),
        "is_denied": coerce_bool(payload.get("is_denied", False), field="is_denied", default=False),
        "is_appealed": coerce_bool(payload.get("is_appealed", False), field="is_appealed", default=False),
        "appeal_success": coerce_bool(payload.get("appeal_success", False), field="appeal_success", default=False),
        "cpt_icd_mismatch": coerce_bool(payload.get("cpt_icd_mismatch", False), field="cpt_icd_mismatch", default=False),
        "high_amount_flag": high_amount_flag,
        "strict_insurance_flag": coerce_bool(
            payload.get("strict_insurance_flag", False), field="strict_insurance_flag", default=False
        ),
        "clinical_notes": str(payload.get("clinical_notes", "") or ""),
    }
    return derive_claim_fields(claim)


def build_runtime_cpt_summary_row(
    payload: Dict[str, Any],
    claim_id: int,
    claim_amount: float,
    feature_columns: Optional[List[str]] = None,
) -> Dict[str, Any]:
    cpt_codes_raw = payload.get("cpt_codes", []) or []
    cpt_codes = [str(x).strip() for x in cpt_codes_raw if str(x).strip()]
    # Normalize codes for feature column matching (training uses int-like codes)
    normalized = []
    for code in cpt_codes:
        try:
            normalized.append(str(int(float(code))))
        except Exception:
            normalized.append(code)

    num_cpt_codes = int(payload.get("num_cpt_codes") or (len(normalized) if normalized else 1))
    total_cpt_amount = safe_float(payload.get("total_cpt_amount"), claim_amount)

    row: Dict[str, Any] = {
        "claim_id": int(claim_id),
        "num_cpt_codes": num_cpt_codes,
        "total_cpt_amount": total_cpt_amount,
        "cpt_codes_list": ",".join(normalized),
    }
    cols = feature_columns or []
    for col in cols:
        if col.startswith("cpt_") and col.endswith("_count"):
            row[col] = 0
    for code in normalized:
        col = f"cpt_{code}_count"
        row[col] = row.get(col, 0) + 1
    return row


class ModelBundle:
    """Loaded (or trained-once) model artifacts for demo scoring."""

    def __init__(self, artifacts: Dict[str, Any]):
        self.artifacts = artifacts

    @property
    def ready(self) -> bool:
        return bool(self.artifacts.get("denial_model"))


_MODEL_BUNDLE: Optional[ModelBundle] = None


def clear_model_bundle() -> None:
    global _MODEL_BUNDLE
    _MODEL_BUNDLE = None


def get_or_train_models(
    master: Optional[pd.DataFrame] = None,
    cpt_summary: Optional[pd.DataFrame] = None,
    train_if_missing: bool = True,
    db_sig: Optional[float] = None,
) -> ModelBundle:
    """Load models from src/core/models; optionally train missing ones once."""
    global _MODEL_BUNDLE
    sig = db_sig if db_sig is not None else db_signature()

    # Reuse loaded ML artifacts across calls, but refresh data-dependent coding/NLP
    # knowledge when the live DB signature changes.
    if _MODEL_BUNDLE is not None and _MODEL_BUNDLE.ready:
        if _MODEL_BUNDLE.artifacts.get("db_sig") == sig:
            return _MODEL_BUNDLE
        # Keep model weights; refresh knowledge tables against current live+historical data.
        raw = load_all(db_sig=sig)
        _MODEL_BUNDLE.artifacts["coding_knowledge"] = build_cpt_icd_knowledge(
            raw["cpt_lines"], raw["icd"], raw["claims"]
        )
        _MODEL_BUNDLE.artifacts["nlp_bundle"] = train_notes_to_icd_model(
            raw["encounters"], raw["claims"], raw["icd"], min_samples=50
        )
        _MODEL_BUNDLE.artifacts["db_sig"] = sig
        return _MODEL_BUNDLE

    if master is None:
        master = build_master(db_sig=sig)
    if cpt_summary is None:
        cpt_summary = get_cpt_summary(db_sig=sig)

    def _load_or_train(load_fn, train_fn, *train_args, meta_key: str = "meta"):
        model, meta = load_fn()
        if model is None and train_if_missing:
            res = train_fn(*train_args)
            model = res["model"]
            meta = res.get(meta_key, res)
            if "feature_cols" not in meta and "feature_cols" in res:
                meta = {"feature_cols": res["feature_cols"], **{k: v for k, v in meta.items() if k != "model"}}
        return model, meta

    denial_model, denial_meta = ml_engine.load_model()
    if denial_model is None and train_if_missing:
        res = ml_engine.train_model(master, cpt_summary)
        denial_model = res["model"]
        denial_meta = {"feature_cols": res["feature_cols"], "all_probabilities": res.get("all_probabilities")}

    mismatch_model, mismatch_meta = _load_or_train(
        ml_engine.load_mismatch_model, ml_engine.train_mismatch_model, master, cpt_summary
    )
    appeals_success_model, appeals_success_meta = _load_or_train(
        ml_engine.load_appeals_success_model, ml_engine.train_appeals_success_model, master, cpt_summary
    )
    appeals_recovery_model, appeals_recovery_meta = _load_or_train(
        ml_engine.load_appeals_recovery_model, ml_engine.train_appeals_recovery_model, master, cpt_summary
    )
    fraud_prob_model, fraud_prob_meta = _load_or_train(
        ml_engine.load_fraud_probability_model, ml_engine.train_fraud_probability_model, master, cpt_summary
    )
    fraud_anomaly_model, fraud_anomaly_meta = _load_or_train(
        ml_engine.load_fraud_anomaly_model, ml_engine.train_fraud_anomaly_model, master
    )
    # Ensure legacy anomaly artifacts get training reference stats when missing.
    if (
        fraud_anomaly_model is not None
        and isinstance(fraud_anomaly_meta, dict)
        and (
            "anomaly_strength_ref_min" not in fraud_anomaly_meta
            or "anomaly_strength_ref_max" not in fraud_anomaly_meta
        )
        and train_if_missing
    ):
        refreshed = ml_engine.train_fraud_anomaly_model(master)
        fraud_anomaly_model, fraud_anomaly_meta = refreshed["model"], refreshed["meta"]

    recon_model, recon_meta = _load_or_train(
        ml_engine.load_reconciliation_risk_model, ml_engine.train_reconciliation_risk_model, master
    )
    elig_model, elig_meta = _load_or_train(
        ml_engine.load_eligibility_risk_model, ml_engine.train_eligibility_risk_model, master
    )

    raw = load_all(db_sig=sig)
    coding_knowledge = build_cpt_icd_knowledge(raw["cpt_lines"], raw["icd"], raw["claims"])
    nlp_bundle = train_notes_to_icd_model(raw["encounters"], raw["claims"], raw["icd"], min_samples=50)

    artifacts = {
        "denial_model": denial_model,
        "denial_meta": denial_meta,
        "mismatch_model": mismatch_model,
        "mismatch_meta": mismatch_meta,
        "appeals_success_model": appeals_success_model,
        "appeals_success_meta": appeals_success_meta,
        "appeals_recovery_model": appeals_recovery_model,
        "appeals_recovery_meta": appeals_recovery_meta,
        "fraud_prob_model": fraud_prob_model,
        "fraud_prob_meta": fraud_prob_meta,
        "fraud_anomaly_model": fraud_anomaly_model,
        "fraud_anomaly_meta": fraud_anomaly_meta,
        "recon_model": recon_model,
        "recon_meta": recon_meta,
        "elig_model": elig_model,
        "elig_meta": elig_meta,
        "coding_knowledge": coding_knowledge,
        "nlp_bundle": nlp_bundle,
        "db_sig": sig,
    }
    _MODEL_BUNDLE = ModelBundle(artifacts)
    return _MODEL_BUNDLE


def _claim_feature_dict(claim: Dict[str, Any], cpt_row: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(claim)
    merged["num_cpt_codes"] = cpt_row.get("num_cpt_codes", claim.get("num_cpt_codes", 1))
    merged["total_cpt_amount"] = cpt_row.get("total_cpt_amount", claim.get("total_cpt_amount", claim.get("claim_amount", 0)))
    return merged


def score_claim_dict(
    claim: Dict[str, Any],
    cpt_row: Dict[str, Any],
    bundle: ModelBundle,
    *,
    clinical_notes: str = "",
    cpt_codes: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Score a single claim dict with the loaded model bundle."""
    a = bundle.artifacts
    feature_claim = _claim_feature_dict(claim, cpt_row)

    denial_pred = ml_engine.predict_single_claim(
        feature_claim, a["denial_model"], a["denial_meta"]["feature_cols"]
    )

    # Batch scorers merge CPT summary themselves — pass claim WITHOUT cpt aggregate cols
    # to avoid pandas merge suffixes (num_cpt_codes_x / _y).
    cpt_overlap = {"num_cpt_codes", "total_cpt_amount", "cpt_codes_list"} | {
        k for k in claim.keys() if str(k).startswith("cpt_") and str(k).endswith("_count")
    }
    base_claim = {k: v for k, v in claim.items() if k not in cpt_overlap}
    claim_df = pd.DataFrame([base_claim])
    cpt_df = pd.DataFrame([cpt_row])
    mismatch_probs = ml_engine.score_all_mismatch(
        claim_df, cpt_df, a["mismatch_model"], a["mismatch_meta"]["feature_cols"]
    )
    mismatch_p = float(mismatch_probs[0]) if len(mismatch_probs) else 0.0

    fraud_df = ml_engine.score_fraud_enhanced(
        claim_df,
        cpt_df,
        a["fraud_prob_model"],
        a["fraud_prob_meta"]["feature_cols"],
        a["fraud_anomaly_model"],
        a["fraud_anomaly_meta"],
        alpha=0.6,
    )
    fraud_p = float(fraud_df.iloc[0]["fraud_probability_improved"])
    fraud_anomaly_p = float(fraud_df.iloc[0].get("fraud_anomaly_probability", 0.0))

    recon_probs = ml_engine.score_reconciliation_risk(
        claim_df, a["recon_model"], a["recon_meta"]["feature_cols"]
    )
    recon_p = float(recon_probs[0]) if len(recon_probs) else 0.0
    recon_leakage_note = None
    if isinstance(a.get("recon_meta"), dict):
        recon_leakage_note = (a["recon_meta"].get("metrics") or {}).get("target_leakage_warning")

    # Appeals scores: compute always for metrics, but agents gate on denial status.
    ranked = ml_engine.predict_appeals_ranked_candidates(
        claim_df,
        cpt_df,
        a["appeals_success_model"],
        a["appeals_success_meta"]["feature_cols"],
        a["appeals_recovery_model"],
        a["appeals_recovery_meta"]["feature_cols"],
    )
    p_appeal_success = safe_float(ranked.iloc[0]["p_appeal_success"], 0.0) if len(ranked) else 0.0
    expected_recovery = safe_float(ranked.iloc[0]["expected_recovery"], 0.0) if len(ranked) else 0.0
    paid_if_success = (
        safe_float(ranked.iloc[0].get("predicted_paid_if_appeal_succeeds"), 0.0) if len(ranked) else 0.0
    )

    codes = cpt_codes or [x.strip() for x in str(cpt_row.get("cpt_codes_list", "")).split(",") if x.strip()]
    coding_reco = build_coding_recommendation(
        feature_claim, codes, a["coding_knowledge"], mismatch_p, min_support=20
    )
    note_pred = predict_icd_from_notes_batch(
        [clinical_notes or str(claim.get("clinical_notes", "") or "")],
        a["nlp_bundle"],
        top_k=3,
    )[0]
    nlp_coding_reco = build_nlp_coding_recommendation(feature_claim, note_pred, mismatch_p)

    return {
        "denial_probability": safe_float(denial_pred.get("denial_probability"), 0.0),
        "risk_level": denial_pred.get("risk_level", "LOW"),
        "shap_explanation": denial_pred.get("shap_explanation"),
        "top_drivers": (
            denial_pred["shap_explanation"].to_dict(orient="records")
            if isinstance(denial_pred.get("shap_explanation"), pd.DataFrame)
            else []
        ),
        "mismatch_probability": mismatch_p,
        "fraud_probability_improved": fraud_p,
        "fraud_anomaly_probability": fraud_anomaly_p,
        "reconciliation_risk_probability": recon_p,
        "reconciliation_leakage_note": recon_leakage_note,
        "p_appeal_success": p_appeal_success,
        "predicted_paid_if_appeal_succeeds": paid_if_success,
        "expected_recovery": expected_recovery,
        "expected_recovery_definition": (
            "P(appeal_success) * predicted_paid_amount_if_appeal_succeeds "
            "(experimental; not denied balance / AR owed)"
        ),
        "appeals_estimate_note": (
            "Experimental / synthetic-data estimate only — not an externally validated appeal success rate. "
            "expected_recovery is expected value under the appeal path, not the claim balance."
        ),
        "coding_recommendation": coding_reco,
        "nlp_coding_recommendation": nlp_coding_reco,
    }


def score_existing_claim(claim_id: int, train_if_missing: bool = True) -> Dict[str, Any]:
    """Score a claim already present in master (historical or live-merged)."""
    sig = db_signature()
    master = build_master(db_sig=sig).reset_index(drop=True)
    cpt_summary = get_cpt_summary(db_sig=sig).reset_index(drop=True)
    if claim_id not in set(master["claim_id"].astype(int).tolist()):
        return {"ok": False, "error": f"Claim ID {claim_id} not found", "db_sig": sig}

    bundle = get_or_train_models(master, cpt_summary, train_if_missing=train_if_missing, db_sig=sig)
    if not bundle.ready:
        return {"ok": False, "error": "Models are not available. Train models before scoring."}

    sel = master[master["claim_id"].astype(int) == int(claim_id)]
    claim = sel.iloc[0].to_dict()
    cpt_row_df = cpt_summary[cpt_summary["claim_id"].astype(int) == int(claim_id)]
    if len(cpt_row_df) == 0:
        cpt_row = {
            "claim_id": int(claim_id),
            "num_cpt_codes": 1,
            "total_cpt_amount": safe_float(claim.get("claim_amount"), 0.0),
            "cpt_codes_list": "",
        }
    else:
        cpt_row = cpt_row_df.iloc[0].to_dict()

    predictions = score_claim_dict(
        claim,
        cpt_row,
        bundle,
        clinical_notes=str(claim.get("clinical_notes", "") or ""),
    )
    return {"ok": True, "claim": claim, "cpt_row": cpt_row, "predictions": predictions}


def run_coordinator_for_claim(
    claim_id: int,
    *,
    force_appeal: bool = False,
    train_if_missing: bool = True,
) -> Dict[str, Any]:
    scored = score_existing_claim(claim_id, train_if_missing=train_if_missing)
    if not scored.get("ok"):
        return scored

    claim = scored["claim"]
    predictions = scored["predictions"]
    agent = CoordinatorAgent()
    agent_out: AgentOutput = agent.run(claim=claim, predictions=predictions, force_appeal=force_appeal)
    return {
        "ok": True,
        "claim_id": int(claim_id),
        "claim": claim,
        "predictions": predictions,
        "recommendation": agent_out.recommendation,
        "action_items": agent_out.action_items,
        "appeal_letter": agent_out.appeal_letter,
        "metrics": agent_out.metrics,
        "steps": [{"agent": s.agent, "step": s.step, "summary": s.summary} for s in agent_out.steps],
        "human_review_required": True,
        "persistence_note": "Orchestration is advisory only; no payer submission or approval was persisted.",
    }


def run_coordinator_for_payload(
    payload: Dict[str, Any],
    *,
    force_appeal: bool = False,
    train_if_missing: bool = True,
) -> Dict[str, Any]:
    ok, errors = validate_claim_payload(payload)
    if not ok:
        return {"ok": False, "error": "Invalid claim payload", "errors": errors}

    sig = db_signature()
    master = build_master(db_sig=sig)
    cpt_summary = get_cpt_summary(db_sig=sig)
    bundle = get_or_train_models(master, cpt_summary, train_if_missing=train_if_missing, db_sig=sig)
    if not bundle.ready:
        return {"ok": False, "error": "Models are not available. Train models before scoring."}

    runtime_claim = build_runtime_claim_row(payload)
    feature_cols = []
    if cpt_summary is not None and len(cpt_summary.columns):
        feature_cols = [c for c in cpt_summary.columns if c.startswith("cpt_") and c.endswith("_count")]
    runtime_cpt = build_runtime_cpt_summary_row(
        payload,
        claim_id=int(runtime_claim["claim_id"]),
        claim_amount=safe_float(runtime_claim["claim_amount"], 0.0),
        feature_columns=feature_cols,
    )
    cpt_codes = [str(x).strip() for x in (payload.get("cpt_codes") or []) if str(x).strip()]
    predictions = score_claim_dict(
        runtime_claim,
        runtime_cpt,
        bundle,
        clinical_notes=str(payload.get("clinical_notes", "") or ""),
        cpt_codes=cpt_codes,
    )
    agent = CoordinatorAgent()
    agent_out = agent.run(claim=runtime_claim, predictions=predictions, force_appeal=force_appeal)
    return {
        "ok": True,
        "mode": "runtime_payload_inference",
        "claim_id": int(runtime_claim["claim_id"]),
        "claim": runtime_claim,
        "predictions": {
            k: v
            for k, v in predictions.items()
            if k not in ("shap_explanation",)
        },
        "recommendation": agent_out.recommendation,
        "action_items": agent_out.action_items,
        "appeal_letter": agent_out.appeal_letter,
        "metrics": agent_out.metrics,
        "steps": [{"agent": s.agent, "step": s.step, "summary": s.summary} for s in agent_out.steps],
        "human_review_required": True,
        "persistence_note": "Orchestration is advisory only; no payer submission or approval was persisted.",
        "normalized_payload_for_persist": {
            **{k: runtime_claim.get(k) for k in (
                "claim_id", "encounter_id", "patient_id", "insurance", "visit_type", "icd_code",
                "gender", "age", "claim_amount", "paid_amount", "fraud_score", "denial_reason",
                "is_denied", "is_appealed", "appeal_success", "cpt_icd_mismatch",
                "high_amount_flag", "strict_insurance_flag",
            )},
            "cpt_codes": cpt_codes,
            "total_cpt_amount": runtime_cpt.get("total_cpt_amount"),
            "num_cpt_codes": runtime_cpt.get("num_cpt_codes"),
            "clinical_notes": runtime_claim.get("clinical_notes", ""),
        },
    }
