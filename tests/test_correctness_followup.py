"""Regression tests for correctness follow-up fixes."""

from __future__ import annotations

import os
import sys
import subprocess
import textwrap

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.core.value_coercion import coerce_bool, coerce_finite_number, CoercionError
from src.core.claim_scoring import validate_claim_payload, build_runtime_claim_row
from src.core import ml_engine
from src.agents.back_end_agent import BackEndAgent
from src.agents.rcm_agent import CoordinatorAgent
from src.core.claim_scoring import derive_claim_fields


def test_coerce_bool_false_string_is_false():
    assert coerce_bool("false", field="x") is False
    assert coerce_bool("FALSE", field="x") is False
    assert coerce_bool("0", field="x") is False
    assert coerce_bool(0, field="x") is False
    assert coerce_bool(False, field="x") is False


def test_coerce_bool_true_variants():
    assert coerce_bool("true", field="x") is True
    assert coerce_bool("1", field="x") is True
    assert coerce_bool(1, field="x") is True
    assert coerce_bool(True, field="x") is True


def test_coerce_bool_rejects_invalid_types():
    with pytest.raises(CoercionError):
        coerce_bool("maybe", field="x")
    with pytest.raises(CoercionError):
        coerce_bool({"a": 1}, field="x")
    with pytest.raises(CoercionError):
        coerce_bool(2, field="x")


def test_validate_rejects_string_false_confusion_and_negatives():
    # "false" must be accepted as False, not rejected / truthy
    ok, errors = validate_claim_payload(
        {
            "claim_id": 1,
            "insurance": "Aetna",
            "visit_type": "OPD",
            "icd_code": "E11",
            "gender": "F",
            "age": 40,
            "claim_amount": 100,
            "is_denied": "false",
            "persist": "false",
        }
    )
    assert ok, errors
    claim = build_runtime_claim_row(
        {
            "claim_id": 1,
            "insurance": "Aetna",
            "visit_type": "OPD",
            "icd_code": "E11",
            "gender": "F",
            "age": 40,
            "claim_amount": 100,
            "is_denied": "false",
        }
    )
    assert claim["is_denied"] is False

    ok2, errors2 = validate_claim_payload(
        {
            "claim_id": 1,
            "insurance": "Aetna",
            "visit_type": "OPD",
            "icd_code": "E11",
            "gender": "F",
            "age": 40,
            "claim_amount": -10,
        }
    )
    assert not ok2
    assert any("claim_amount" in e for e in errors2)

    ok3, errors3 = validate_claim_payload(
        {
            "claim_id": 1,
            "insurance": "Aetna",
            "visit_type": "OPD",
            "icd_code": "E11",
            "gender": "F",
            "age": 40,
            "claim_amount": float("nan"),
        }
    )
    assert not ok3


def test_coerce_finite_rejects_inf():
    with pytest.raises(CoercionError):
        coerce_finite_number(float("inf"), field="claim_amount")


def test_force_appeal_rejected_for_non_denied_and_resolved():
    preds = {
        "fraud_probability_improved": 0.1,
        "p_appeal_success": 0.9,
        "expected_recovery": 500.0,
        "reconciliation_risk_probability": 0.1,
    }
    clean = derive_claim_fields(
        {"claim_id": 1, "is_denied": False, "is_appealed": False, "appeal_success": False, "claim_amount": 100}
    )
    out = BackEndAgent().run(clean, preds, force_appeal=True)
    assert out["metrics"]["appealable"] is False
    assert out["metrics"]["override_rejected"] is True
    assert out["artifacts"]["appeal_letter"] is None

    resolved = derive_claim_fields(
        {
            "claim_id": 2,
            "is_denied": True,
            "is_appealed": True,
            "appeal_success": True,
            "claim_amount": 100,
            "denial_reason": "Auth required",
        }
    )
    out2 = BackEndAgent().run(resolved, preds, force_appeal=True)
    assert out2["metrics"]["appealable"] is False
    assert out2["metrics"]["override_rejected"] is True


def test_force_appeal_allowed_only_for_denied_open_when_gate_fails():
    preds = {
        "fraud_probability_improved": 0.1,
        "p_appeal_success": 0.1,
        "expected_recovery": 1.0,
        "reconciliation_risk_probability": 0.1,
    }
    denied_open = derive_claim_fields(
        {
            "claim_id": 3,
            "is_denied": True,
            "is_appealed": False,
            "appeal_success": False,
            "claim_amount": 200,
            "denial_reason": "Auth required",
        }
    )
    out = BackEndAgent().run(denied_open, preds, force_appeal=True)
    assert out["metrics"]["appealable"] is True
    assert "access-control" in " ".join(out["actions"]).lower() or "unauthenticated" in " ".join(out["actions"]).lower()
    assert "authorized operator" not in " ".join(out["actions"]).lower()


def test_coordinator_consolidated_recommendation_prioritizes_fraud_and_appeal():
    claim = derive_claim_fields(
        {
            "claim_id": 9,
            "is_denied": True,
            "is_appealed": False,
            "denial_reason": "Auth required",
            "claim_amount": 1000,
            "cpt_icd_mismatch": True,
        }
    )
    preds = {
        "denial_probability": 0.85,
        "mismatch_probability": 0.8,
        "fraud_probability_improved": 0.9,
        "p_appeal_success": 0.8,
        "expected_recovery": 400.0,
        "reconciliation_risk_probability": 0.7,
        "coding_recommendation": {"should_review": True, "suggestions": [("E11", 0.5)]},
        "nlp_coding_recommendation": {"should_review": False},
    }
    out = CoordinatorAgent().run(claim, preds)
    assert "Fraud" in out.recommendation or "fraud" in out.recommendation.lower()
    assert "Appeal" in out.recommendation or "appeal" in out.recommendation.lower()
    assert out.recommendation != "Add 'Auth required' to this claim and review documentation."
    assert any(s.step == "Consolidate" for s in out.steps)


def test_fraud_anomaly_single_claim_not_degenerate():
    """Single-row scoring must not collapse anomaly contribution to all zeros via batch min-max."""
    rng = np.random.default_rng(0)
    n = 200
    master = pd.DataFrame(
        {
            "claim_id": np.arange(1, n + 1),
            "claim_amount": rng.uniform(100, 5000, n),
            "age": rng.integers(18, 90, n),
            "revenue_leakage": rng.uniform(0, 2000, n),
            "collection_rate": rng.uniform(0, 100, n),
            "insurance": ["Aetna"] * n,
            "visit_type": ["OPD"] * n,
            "icd_code": ["E11"] * n,
            "gender": ["F"] * n,
            "fraud_score": rng.uniform(0, 1, n),
            "cpt_icd_mismatch": False,
            "high_amount_flag": False,
            "strict_insurance_flag": False,
            "denial_reason": ["None"] * n,
            "is_denied": False,
            "paid_amount": rng.uniform(0, 1000, n),
        }
    )
    # Extreme outlier claim
    master.loc[0, ["claim_amount", "revenue_leakage", "collection_rate"]] = [50000, 49999, 0.01]

    cpt = pd.DataFrame(
        {
            "claim_id": master["claim_id"],
            "num_cpt_codes": 1,
            "total_cpt_amount": master["claim_amount"],
            "cpt_codes_list": "99213",
        }
    )

    # Train fraud models into temp paths by monkeypatching MODEL_DIR via retrain functions
    # Use existing train APIs (writes to src/core/models — acceptable in demo workspace).
    anomaly = ml_engine.train_fraud_anomaly_model(master)
    assert "anomaly_strength_ref_min" in anomaly["meta"]
    assert "anomaly_strength_ref_max" in anomaly["meta"]

    # Supervised fraud model
    master["fraud_flag"] = (master["fraud_score"] > 0.8).astype(int)
    supervised = ml_engine.train_fraud_probability_model(master, cpt)

    single = master.iloc[[0]].copy()
    cpt_single = cpt.iloc[[0]].copy()
    scored_single = ml_engine.score_fraud_enhanced(
        single,
        cpt_single,
        supervised["model"],
        supervised["meta"]["feature_cols"],
        anomaly["model"],
        anomaly["meta"],
        alpha=0.6,
    )
    portfolio = ml_engine.score_fraud_enhanced(
        master,
        cpt,
        supervised["model"],
        supervised["meta"]["feature_cols"],
        anomaly["model"],
        anomaly["meta"],
        alpha=0.6,
    )
    a_single = float(scored_single.iloc[0]["fraud_anomaly_probability"])
    a_port = float(portfolio.iloc[0]["fraud_anomaly_probability"])
    # Meaningful contribution for outlier
    assert a_single > 0.05
    # Stable vs portfolio (same training reference)
    assert abs(a_single - a_port) < 1e-6


def test_score_existing_claim_sees_persisted_claim_across_process(tmp_path, monkeypatch):
    """Persisted claim must be retrievable/scorable in a fresh Python process (new cache)."""
    import src.core.demo_db as demo_db
    import src.core.data_loader as data_loader
    from src.core.claim_scoring import clear_model_bundle

    db_file = tmp_path / "demo_live.sqlite3"
    monkeypatch.setattr(demo_db, "DB_PATH", str(db_file))
    data_loader.clear_data_caches()
    clear_model_bundle()

    claim_id = 993311
    demo_db.upsert_live_claim(
        {
            "claim_id": claim_id,
            "insurance": "UHC",
            "visit_type": "OPD",
            "icd_code": "E11",
            "gender": "F",
            "age": 55,
            "claim_amount": 1200,
            "paid_amount": 0,
            "is_denied": True,
            "denial_reason": "Auth required",
            "cpt_codes": ["99213"],
            "clinical_notes": "Diabetes follow-up",
        }
    )

    # Simulate a separate process with its own interpreter / caches.
    script = textwrap.dedent(
        f"""
        import os, sys
        sys.path.insert(0, {ROOT!r})
        os.environ['STREAMLIT_BROWSER_GATHER_USAGE_STATS'] = 'false'
        import src.core.demo_db as demo_db
        demo_db.DB_PATH = {str(db_file)!r}
        from src.core.data_loader import clear_data_caches
        from src.core.claim_scoring import score_existing_claim, clear_model_bundle
        clear_data_caches()
        clear_model_bundle()
        out = score_existing_claim({claim_id}, train_if_missing=True)
        print('OK' if out.get('ok') else 'FAIL')
        print(out.get('error', ''))
        print(out.get('claim', {{}}).get('insurance', ''))
        print(round(float(out.get('predictions', {{}}).get('denial_probability', -1)), 6))
        """
    )
    proc = subprocess.run(
        [sys.executable, "-c", script],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert proc.returncode == 0, proc.stderr + proc.stdout
    lines = [ln.strip() for ln in proc.stdout.splitlines() if ln.strip()]
    lines = [ln for ln in lines if not ln.startswith("202") and "WARNING" not in ln and "urllib3" not in ln]
    assert "OK" in lines, proc.stdout + proc.stderr
    assert "UHC" in lines, proc.stdout
    # denial probability printed as a float line
    probs = []
    for ln in lines:
        try:
            probs.append(float(ln))
        except ValueError:
            pass
    assert probs and probs[-1] >= 0.0
