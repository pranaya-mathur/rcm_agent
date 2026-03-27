"""
data_loader.py — Centralized data loading & transformation for the RCM Dashboard.
Uses Streamlit caching to avoid re-reading CSVs on every rerun.
"""

import os
import streamlit as st
import pandas as pd
import numpy as np
from typing import Optional
from src.core.demo_db import load_live_frames, db_signature

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


@st.cache_data(show_spinner=False)
def load_all(_db_sig: Optional[float] = None):
    """Load every CSV and return a dict of DataFrames."""
    files = {
        "claims":     "v3_claims.csv",
        "patients":   "v3_patients.csv",
        "encounters": "v3_encounters.csv",
        "denials":    "v3_denials.csv",
        "appeals":    "v3_appeals.csv",
        "payments":   "v3_payments.csv",
        "fraud":      "v3_fraud.csv",
        "scrubbing":  "v3_scrubbing.csv",
        "icd":        "v3_icd.csv",
        "cpt_lines":  "v3_cpt_lines.csv",
        "events":     "v3_events.csv",
    }
    dfs = {}
    for key, fname in files.items():
        dfs[key] = pd.read_csv(os.path.join(DATA_DIR, fname))

    # Merge live claims captured in SQLite demo DB (if any).
    live = load_live_frames()
    for key, live_df in live.items():
        if key not in dfs or live_df is None or len(live_df) == 0:
            continue
        base_df = dfs[key]
        # Keep latest version by key while preferring live records on collisions.
        if "claim_id" in base_df.columns and "claim_id" in live_df.columns:
            combined = pd.concat([base_df, live_df], ignore_index=True)
            dfs[key] = combined.drop_duplicates(subset=["claim_id"], keep="last")
        elif "encounter_id" in base_df.columns and "encounter_id" in live_df.columns:
            combined = pd.concat([base_df, live_df], ignore_index=True)
            dfs[key] = combined.drop_duplicates(subset=["encounter_id"], keep="last")
        else:
            dfs[key] = pd.concat([base_df, live_df], ignore_index=True)
    return dfs


@st.cache_data(show_spinner=False)
def build_master():
    """
    Build a master claims-level DataFrame by merging:
    claims + payments + denials + fraud + scrubbing + encounters + patients + icd
    """
    dfs = load_all(_db_sig=db_signature())

    master = dfs["claims"].copy()

    # payments
    master = master.merge(dfs["payments"], on="claim_id", how="left")

    # denials — flag
    denials = dfs["denials"].copy()
    denials["is_denied"] = True
    master = master.merge(denials, on="claim_id", how="left")
    master["is_denied"] = master["is_denied"].astype("boolean").fillna(False).astype(bool)
    master["denial_reason"] = master["denial_reason"].fillna("None")

    # appeals — flag
    appeals = dfs["appeals"].copy()
    appeals["is_appealed"] = True
    appeals["appeal_success"] = appeals["success"].map({"True": True, "False": False, True: True, False: False})
    master = master.merge(
        appeals[["claim_id", "is_appealed", "appeal_success"]],
        on="claim_id", how="left"
    )
    master["is_appealed"] = master["is_appealed"].astype("boolean").fillna(False).astype(bool)
    master["appeal_success"] = master["appeal_success"].astype("boolean").fillna(False).astype(bool)

    # fraud
    master = master.merge(dfs["fraud"], on="claim_id", how="left")

    # scrubbing
    master = master.merge(dfs["scrubbing"], on="claim_id", how="left")
    # Ensure scrubbing booleans are always present for downstream feature engineering.
    for col in ["cpt_icd_mismatch", "high_amount_flag", "strict_insurance_flag"]:
        if col in master.columns:
            master[col] = master[col].astype("boolean").fillna(False).astype(bool)

    # encounters (to get visit_type, patient_id)
    master = master.merge(dfs["encounters"], on="encounter_id", how="left")

    # patients (age, gender)
    master = master.merge(dfs["patients"], on="patient_id", how="left", suffixes=("", "_pat"))

    # ICD codes
    master = master.merge(dfs["icd"], on="claim_id", how="left")

    # derived columns
    master["revenue_leakage"] = master["claim_amount"] - master["paid_amount"]
    master["collection_rate"] = (master["paid_amount"] / master["claim_amount"] * 100).round(2)

    # clean claim flag (no scrubbing issues)
    scrub_cols = ["cpt_icd_mismatch", "high_amount_flag", "strict_insurance_flag"]
    master["is_clean_claim"] = ~master[scrub_cols].any(axis=1)

    # age buckets
    bins = [0, 18, 35, 50, 65, 120]
    labels = ["0-18", "19-35", "36-50", "51-65", "65+"]
    master["age_group"] = pd.cut(master["age"], bins=bins, labels=labels, right=True)

    return master


@st.cache_data(show_spinner=False)
def build_events_timeline():
    """Pivot events to get per-claim timestamps for CREATED, SUBMITTED, PROCESSED."""
    dfs = load_all(_db_sig=db_signature())
    events = dfs["events"].copy()
    events["timestamp"] = pd.to_datetime(events["timestamp"])

    pivot = events.pivot_table(index="claim_id", columns="event", values="timestamp", aggfunc="first")
    pivot = pivot.reset_index()

    if "CREATED" in pivot.columns and "SUBMITTED" in pivot.columns:
        pivot["days_to_submit"] = (pivot["SUBMITTED"] - pivot["CREATED"]).dt.days
    if "SUBMITTED" in pivot.columns and "PROCESSED" in pivot.columns:
        pivot["days_to_process"] = (pivot["PROCESSED"] - pivot["SUBMITTED"]).dt.days
    if "CREATED" in pivot.columns and "PROCESSED" in pivot.columns:
        pivot["total_cycle_days"] = (pivot["PROCESSED"] - pivot["CREATED"]).dt.days

    return pivot


@st.cache_data(show_spinner=False)
def get_cpt_summary(top_k: int = 25):
    """CPT line-item summary per claim.

    Includes aggregate features (num_cpt_codes, total_cpt_amount) plus counts for the
    most frequent CPT codes across the dataset to make CPT-aware ML feasible.
    """
    dfs = load_all(_db_sig=db_signature())
    cpt = dfs["cpt_lines"].copy()

    # Base aggregates
    summary = cpt.groupby("claim_id").agg(
        num_cpt_codes=("cpt_code", "count"),
        total_cpt_amount=("amount", "sum"),
        cpt_codes_list=("cpt_code", lambda x: ", ".join(x.astype(str)))
    ).reset_index()

    # CPT-aware features: counts for top CPT codes
    top_codes = cpt["cpt_code"].value_counts().head(top_k).index.tolist()
    cpt_top = cpt[cpt["cpt_code"].isin(top_codes)]
    pivot = cpt_top.pivot_table(
        index="claim_id",
        columns="cpt_code",
        values="cpt_code",
        aggfunc="count",
        fill_value=0,
    )
    pivot.columns = [f"cpt_{int(code)}_count" for code in pivot.columns]
    pivot = pivot.reset_index()

    summary = summary.merge(pivot, on="claim_id", how="left")
    cpt_code_cols = [c for c in summary.columns if c.startswith("cpt_") and c.endswith("_count")]
    if cpt_code_cols:
        summary[cpt_code_cols] = summary[cpt_code_cols].fillna(0)
    return summary
