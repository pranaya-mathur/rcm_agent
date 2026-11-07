"""Integration-ish tests for SQLite live claim persistence and merge behavior."""

import os
import sys
import tempfile
import importlib

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def test_live_roundtrip_preserves_demographics(monkeypatch, tmp_path):
    import src.core.demo_db as demo_db

    db_file = tmp_path / "demo_live.sqlite3"
    monkeypatch.setattr(demo_db, "DB_PATH", str(db_file))

    payload = {
        "claim_id": 991001,
        "encounter_id": 881001,
        "patient_id": 771001,
        "insurance": "Cigna",
        "visit_type": "OPD",
        "icd_code": "I10",
        "gender": "M",
        "age": 61,
        "claim_amount": 2200,
        "paid_amount": 0,
        "fraud_score": 0.15,
        "denial_reason": "Coverage",
        "is_denied": True,
        "is_appealed": False,
        "appeal_success": False,
        "cpt_icd_mismatch": True,
        "high_amount_flag": True,
        "strict_insurance_flag": False,
        "cpt_codes": ["99213", "80053"],
        "total_cpt_amount": 2100,
        "clinical_notes": "Hypertension follow-up",
    }
    demo_db.upsert_live_claim(payload)
    demo_db.upsert_live_claim(payload)  # idempotent

    frames = demo_db.load_live_frames()
    assert len(frames["claims"]) == 1
    assert frames["claims"].iloc[0]["insurance"] == "Cigna"
    assert frames["patients"].iloc[0]["age"] == 61
    assert frames["patients"].iloc[0]["gender"] == "M"
    assert frames["encounters"].iloc[0]["clinical_notes"] == "Hypertension follow-up"
    assert bool(frames["scrubbing"].iloc[0]["cpt_icd_mismatch"]) is True
    assert len(frames["cpt_lines"]) == 2

    stats = demo_db.get_live_stats()
    assert stats["count"] == 1
    assert stats["last_claim_id"] == 991001


def test_cpt_lines_merge_does_not_collapse_historical(monkeypatch, tmp_path):
    """Persisting one live claim must not drop_duplicates all historical CPT lines."""
    import src.core.demo_db as demo_db
    import src.core.data_loader as data_loader

    db_file = tmp_path / "demo_live.sqlite3"
    monkeypatch.setattr(demo_db, "DB_PATH", str(db_file))

    # Clear caches tied to previous DB signature
    data_loader.clear_data_caches()

    before = data_loader.load_all(db_sig=0.0)
    before_n = len(before["cpt_lines"])
    assert before_n > 50_000

    demo_db.upsert_live_claim(
        {
            "claim_id": 991002,
            "insurance": "Aetna",
            "visit_type": "ER",
            "icd_code": "J45",
            "gender": "F",
            "age": 40,
            "claim_amount": 900,
            "cpt_codes": ["99284"],
            "clinical_notes": "Asthma exacerbation",
            "is_denied": False,
        }
    )
    sig = demo_db.db_signature()
    data_loader.clear_data_caches()
    after = data_loader.load_all(db_sig=sig)
    # Should grow (or stay near before), never collapse to ~50k unique claim rows only.
    assert len(after["cpt_lines"]) >= before_n
    master = data_loader.build_master(db_sig=sig)
    row = master[master["claim_id"] == 991002]
    assert len(row) == 1
    assert row.iloc[0]["insurance"] == "Aetna"
    assert int(row.iloc[0]["age"]) == 40
    assert row.iloc[0]["gender"] == "F"
