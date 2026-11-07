"""Unit tests: payload validation, appeal gating, draft safety."""

import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.core.claim_scoring import validate_claim_payload, build_runtime_claim_row, derive_claim_fields
from src.agents.back_end_agent import BackEndAgent
from src.agents.rcm_agent import CoordinatorAgent


def _base_payload(**overrides):
    p = {
        "claim_id": 990001,
        "insurance": "Aetna",
        "visit_type": "OPD",
        "icd_code": "E11",
        "gender": "F",
        "age": 54,
        "claim_amount": 1850,
        "paid_amount": 0,
        "fraud_score": 0.2,
        "is_denied": True,
        "denial_reason": "Auth required",
        "is_appealed": False,
        "cpt_codes": ["99213", "80053"],
        "clinical_notes": "Diabetes follow-up",
    }
    p.update(overrides)
    return p


def test_validate_requires_core_fields():
    ok, errors = validate_claim_payload({"claim_id": 1})
    assert not ok
    assert any("insurance" in e for e in errors)


def test_validate_accepts_good_payload():
    ok, errors = validate_claim_payload(_base_payload())
    assert ok, errors


def test_runtime_claim_does_not_invent_denied_from_defaults():
    claim = build_runtime_claim_row(_base_payload(is_denied=False, denial_reason="None"))
    assert claim["is_denied"] is False
    assert claim["insurance"] == "Aetna"
    assert claim["visit_type"] == "OPD"
    assert claim["revenue_leakage"] == 1850.0


def test_visit_type_alias_normalized():
    claim = build_runtime_claim_row(_base_payload(visit_type="OP"))
    assert claim["visit_type"] == "OPD"


def test_no_appeal_for_non_denied_claim():
    claim = derive_claim_fields(_base_payload(is_denied=False, denial_reason="None"))
    preds = {
        "fraud_probability_improved": 0.1,
        "p_appeal_success": 0.99,
        "expected_recovery": 1000.0,
        "reconciliation_risk_probability": 0.1,
    }
    out = BackEndAgent().run(claim, preds, force_appeal=False)
    assert out["metrics"]["appealable"] is False
    assert out["artifacts"]["appeal_letter"] is None


def test_appeal_draft_for_denied_open_claim():
    claim = derive_claim_fields(_base_payload(is_denied=True, is_appealed=False))
    preds = {
        "fraud_probability_improved": 0.1,
        "p_appeal_success": 0.8,
        "expected_recovery": 500.0,
        "reconciliation_risk_probability": 0.1,
    }
    out = BackEndAgent().run(claim, preds, force_appeal=False)
    assert out["metrics"]["appealable"] is True
    letter = out["artifacts"]["appeal_letter"]
    assert letter is not None
    assert "DRAFT" in letter
    assert "NOT verified" in letter
    assert "medically necessary" not in letter.lower() or "NOT verified" in letter


def test_no_appeal_for_already_resolved():
    claim = derive_claim_fields(_base_payload(is_denied=True, is_appealed=True, appeal_success=True))
    preds = {
        "fraud_probability_improved": 0.1,
        "p_appeal_success": 0.9,
        "expected_recovery": 800.0,
        "reconciliation_risk_probability": 0.1,
    }
    out = BackEndAgent().run(claim, preds, force_appeal=False)
    assert out["metrics"]["appealable"] is False


def test_force_appeal_override_is_labeled():
    claim = derive_claim_fields(_base_payload(is_denied=False))
    preds = {
        "fraud_probability_improved": 0.1,
        "p_appeal_success": 0.1,
        "expected_recovery": 10.0,
        "reconciliation_risk_probability": 0.1,
    }
    out = BackEndAgent().run(claim, preds, force_appeal=True)
    assert out["metrics"]["appealable"] is True
    assert any("override" in a.lower() for a in out["actions"])


def test_coordinator_review_gate_not_online_learning():
    claim = derive_claim_fields(_base_payload())
    preds = {
        "denial_probability": 0.8,
        "mismatch_probability": 0.2,
        "fraud_probability_improved": 0.1,
        "p_appeal_success": 0.7,
        "expected_recovery": 400.0,
        "reconciliation_risk_probability": 0.2,
        "coding_recommendation": {"should_review": False},
        "nlp_coding_recommendation": {"should_review": False},
    }
    out = CoordinatorAgent().run(claim, preds)
    steps = [s.step for s in out.steps]
    assert "Review Gate" in steps
    assert "Learn" not in steps
    assert any("HUMAN REVIEW REQUIRED" in a for a in out.action_items)
