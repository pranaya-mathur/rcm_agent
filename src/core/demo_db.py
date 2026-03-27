"""
demo_db.py — SQLite persistence for demo-time live claim intake.
"""

from __future__ import annotations

import json
import os
import sqlite3
from typing import Any, Dict

import pandas as pd

DB_PATH = os.path.join(os.path.dirname(__file__), "demo_live.sqlite3")


def _connect() -> sqlite3.Connection:
    return sqlite3.connect(DB_PATH)


def init_db() -> None:
    with _connect() as con:
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS live_claim_payloads (
                claim_id INTEGER PRIMARY KEY,
                encounter_id INTEGER,
                patient_id INTEGER,
                insurance TEXT,
                visit_type TEXT,
                icd_code TEXT,
                gender TEXT,
                age INTEGER,
                claim_amount REAL,
                paid_amount REAL,
                fraud_score REAL,
                denial_reason TEXT,
                is_denied INTEGER,
                is_appealed INTEGER,
                appeal_success INTEGER,
                cpt_codes_json TEXT,
                total_cpt_amount REAL,
                clinical_notes TEXT,
                payload_json TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
            """
        )


def db_signature() -> float:
    """
    File mtime marker for Streamlit cache invalidation.
    """
    if not os.path.exists(DB_PATH):
        return 0.0
    return os.path.getmtime(DB_PATH)


def upsert_live_claim(payload: Dict[str, Any]) -> None:
    init_db()
    claim_id = int(payload.get("claim_id"))
    cpt_codes = payload.get("cpt_codes", [])
    cpt_codes_json = json.dumps([str(x).strip() for x in cpt_codes if str(x).strip()])

    row = (
        claim_id,
        int(payload.get("encounter_id", 0) or 0),
        int(payload.get("patient_id", 0) or 0),
        str(payload.get("insurance", "Unknown")),
        str(payload.get("visit_type", "OP")),
        str(payload.get("icd_code", "Unknown")),
        str(payload.get("gender", "U")),
        int(payload.get("age", 0) or 0),
        float(payload.get("claim_amount", 0.0) or 0.0),
        float(payload.get("paid_amount", 0.0) or 0.0),
        float(payload.get("fraud_score", 0.0) or 0.0),
        str(payload.get("denial_reason", "None")),
        int(bool(payload.get("is_denied", False))),
        int(bool(payload.get("is_appealed", False))),
        int(bool(payload.get("appeal_success", False))),
        cpt_codes_json,
        float(payload.get("total_cpt_amount", payload.get("claim_amount", 0.0)) or 0.0),
        str(payload.get("clinical_notes", "") or ""),
        json.dumps(payload, ensure_ascii=False),
    )

    with _connect() as con:
        con.execute(
            """
            INSERT INTO live_claim_payloads (
                claim_id, encounter_id, patient_id, insurance, visit_type, icd_code, gender, age,
                claim_amount, paid_amount, fraud_score, denial_reason, is_denied, is_appealed,
                appeal_success, cpt_codes_json, total_cpt_amount, clinical_notes, payload_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(claim_id) DO UPDATE SET
                encounter_id=excluded.encounter_id,
                patient_id=excluded.patient_id,
                insurance=excluded.insurance,
                visit_type=excluded.visit_type,
                icd_code=excluded.icd_code,
                gender=excluded.gender,
                age=excluded.age,
                claim_amount=excluded.claim_amount,
                paid_amount=excluded.paid_amount,
                fraud_score=excluded.fraud_score,
                denial_reason=excluded.denial_reason,
                is_denied=excluded.is_denied,
                is_appealed=excluded.is_appealed,
                appeal_success=excluded.appeal_success,
                cpt_codes_json=excluded.cpt_codes_json,
                total_cpt_amount=excluded.total_cpt_amount,
                clinical_notes=excluded.clinical_notes,
                payload_json=excluded.payload_json,
                updated_at=CURRENT_TIMESTAMP
            """,
            row,
        )


def load_live_frames() -> Dict[str, pd.DataFrame]:
    """
    Return live-data DataFrames aligned to core pipeline table names.
    """
    init_db()
    with _connect() as con:
        live = pd.read_sql_query("SELECT * FROM live_claim_payloads", con)

    if live.empty:
        return {
            "claims": pd.DataFrame(),
            "encounters": pd.DataFrame(),
            "payments": pd.DataFrame(),
            "denials": pd.DataFrame(),
            "appeals": pd.DataFrame(),
            "fraud": pd.DataFrame(),
            "scrubbing": pd.DataFrame(),
            "icd": pd.DataFrame(),
            "cpt_lines": pd.DataFrame(),
        }

    def _bool_series(col: str) -> pd.Series:
        return live[col].fillna(0).astype(int).astype(bool)

    claims = live[["claim_id", "encounter_id", "claim_amount"]].copy()
    encounters = live[["encounter_id", "patient_id", "visit_type"]].copy()
    encounters["clinical_notes"] = live["clinical_notes"].fillna("")
    payments = live[["claim_id", "paid_amount"]].copy()
    denials = live.loc[_bool_series("is_denied"), ["claim_id", "denial_reason"]].copy()
    appeals = live.loc[_bool_series("is_appealed"), ["claim_id", "appeal_success"]].copy()
    appeals["success"] = appeals["appeal_success"].astype(bool)

    fraud = live[["claim_id", "fraud_score"]].copy()
    fraud["fraud_flag"] = (fraud["fraud_score"] >= 0.8).astype(int)

    scrubbing = live[["claim_id"]].copy()
    scrubbing["cpt_icd_mismatch"] = False
    scrubbing["high_amount_flag"] = (live["claim_amount"].fillna(0) > 1500).astype(bool).values
    scrubbing["strict_insurance_flag"] = False

    icd = live[["claim_id", "icd_code"]].copy()

    cpt_rows = []
    for r in live.itertuples(index=False):
        try:
            codes = json.loads(getattr(r, "cpt_codes_json", "[]") or "[]")
        except Exception:
            codes = []
        if not codes:
            continue
        per_amt = float(getattr(r, "total_cpt_amount", 0.0) or 0.0) / max(len(codes), 1)
        for code in codes:
            cpt_rows.append({"claim_id": int(r.claim_id), "cpt_code": str(code), "amount": per_amt})
    cpt_lines = pd.DataFrame(cpt_rows, columns=["claim_id", "cpt_code", "amount"])

    return {
        "claims": claims,
        "encounters": encounters,
        "payments": payments,
        "denials": denials,
        "appeals": appeals,
        "fraud": fraud,
        "scrubbing": scrubbing,
        "icd": icd,
        "cpt_lines": cpt_lines,
    }

