"""
data_loader.py — Centralized data loading & transformation for the RCM Dashboard.
Uses Streamlit caching to avoid re-reading CSVs on every rerun.
"""

import os
import streamlit as st
import pandas as pd
from typing import Optional
from src.core.demo_db import load_live_frames, db_signature

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

# Tables that are one-row-per-claim_id
_CLAIM_KEYED = {
    "claims", "denials", "appeals", "payments", "fraud", "scrubbing", "icd", "events"
}
# Tables that are one-row-per-encounter_id
_ENCOUNTER_KEYED = {"encounters"}
# Tables that are one-row-per-patient_id
_PATIENT_KEYED = {"patients"}
# Multi-row tables keyed by claim_id (replace all rows for colliding claim_ids)
_CLAIM_MULTI = {"cpt_lines"}


def _merge_live(base_df: pd.DataFrame, live_df: pd.DataFrame, key: str, table: str) -> pd.DataFrame:
    if live_df is None or len(live_df) == 0:
        return base_df

    if table in _CLAIM_MULTI and "claim_id" in live_df.columns:
        live_ids = set(live_df["claim_id"].astype(int).tolist())
        kept = base_df[~base_df["claim_id"].astype(int).isin(live_ids)].copy() if "claim_id" in base_df.columns else base_df.copy()
        return pd.concat([kept, live_df], ignore_index=True)

    if table in _CLAIM_KEYED and "claim_id" in base_df.columns and "claim_id" in live_df.columns:
        combined = pd.concat([base_df, live_df], ignore_index=True)
        return combined.drop_duplicates(subset=["claim_id"], keep="last")

    if table in _ENCOUNTER_KEYED and "encounter_id" in base_df.columns and "encounter_id" in live_df.columns:
        combined = pd.concat([base_df, live_df], ignore_index=True)
        return combined.drop_duplicates(subset=["encounter_id"], keep="last")

    if table in _PATIENT_KEYED and "patient_id" in base_df.columns and "patient_id" in live_df.columns:
        combined = pd.concat([base_df, live_df], ignore_index=True)
        return combined.drop_duplicates(subset=["patient_id"], keep="last")

    return pd.concat([base_df, live_df], ignore_index=True)


@st.cache_data(show_spinner=False)
def load_all(db_sig: Optional[float] = None):
    """Load every CSV and merge live SQLite frames. db_sig must be hashed for cache invalidation."""
    files = {
        "claims": "v3_claims.csv",
        "patients": "v3_patients.csv",
        "encounters": "v3_encounters.csv",
        "denials": "v3_denials.csv",
        "appeals": "v3_appeals.csv",
        "payments": "v3_payments.csv",
        "fraud": "v3_fraud.csv",
        "scrubbing": "v3_scrubbing.csv",
        "icd": "v3_icd.csv",
        "cpt_lines": "v3_cpt_lines.csv",
        "events": "v3_events.csv",
    }
    dfs = {}
    for key, fname in files.items():
        dfs[key] = pd.read_csv(os.path.join(DATA_DIR, fname))

    live = load_live_frames()
    for key, live_df in live.items():
        if key not in dfs:
            if live_df is not None and len(live_df) > 0:
                dfs[key] = live_df.copy()
            continue
        dfs[key] = _merge_live(dfs[key], live_df, key="claim_id", table=key)
    return dfs


@st.cache_data(show_spinner=False)
def build_master(db_sig: Optional[float] = None):
    """
    Build a master claims-level DataFrame by merging:
    claims + payments + denials + fraud + scrubbing + encounters + patients + icd
    """
    sig = db_sig if db_sig is not None else db_signature()
    dfs = load_all(db_sig=sig)

    master = dfs["claims"].copy()

    master = master.merge(dfs["payments"], on="claim_id", how="left")

    denials = dfs["denials"].copy()
    denials["is_denied"] = True
    master = master.merge(denials, on="claim_id", how="left")
    master["is_denied"] = master["is_denied"].astype("boolean").fillna(False).astype(bool)
    master["denial_reason"] = master["denial_reason"].fillna("None")

    appeals = dfs["appeals"].copy()
    appeals["is_appealed"] = True
    if "success" in appeals.columns:
        appeals["appeal_success"] = appeals["success"].map(
            {"True": True, "False": False, True: True, False: False}
        )
    else:
        appeals["appeal_success"] = False
    master = master.merge(
        appeals[["claim_id", "is_appealed", "appeal_success"]],
        on="claim_id", how="left"
    )
    master["is_appealed"] = master["is_appealed"].astype("boolean").fillna(False).astype(bool)
    master["appeal_success"] = master["appeal_success"].astype("boolean").fillna(False).astype(bool)

    master = master.merge(dfs["fraud"], on="claim_id", how="left")

    master = master.merge(dfs["scrubbing"], on="claim_id", how="left")
    for col in ["cpt_icd_mismatch", "high_amount_flag", "strict_insurance_flag"]:
        if col in master.columns:
            master[col] = master[col].astype("boolean").fillna(False).astype(bool)

    master = master.merge(dfs["encounters"], on="encounter_id", how="left")
    master = master.merge(dfs["patients"], on="patient_id", how="left", suffixes=("", "_pat"))
    master = master.merge(dfs["icd"], on="claim_id", how="left")

    master["revenue_leakage"] = master["claim_amount"] - master["paid_amount"]
    master["collection_rate"] = (master["paid_amount"] / master["claim_amount"] * 100).round(2)

    scrub_cols = ["cpt_icd_mismatch", "high_amount_flag", "strict_insurance_flag"]
    master["is_clean_claim"] = ~master[scrub_cols].any(axis=1)

    bins = [0, 18, 35, 50, 65, 120]
    labels = ["0-18", "19-35", "36-50", "51-65", "65+"]
    master["age_group"] = pd.cut(master["age"], bins=bins, labels=labels, right=True)

    return master


@st.cache_data(show_spinner=False)
def build_events_timeline(db_sig: Optional[float] = None):
    """Pivot events to get per-claim timestamps for CREATED, SUBMITTED, PROCESSED."""
    sig = db_sig if db_sig is not None else db_signature()
    dfs = load_all(db_sig=sig)
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
def get_cpt_summary(top_k: int = 25, db_sig: Optional[float] = None):
    """CPT line-item summary per claim."""
    sig = db_sig if db_sig is not None else db_signature()
    dfs = load_all(db_sig=sig)
    cpt = dfs["cpt_lines"].copy()
    # Normalize CPT codes to comparable dtype for feature columns
    cpt["cpt_code"] = cpt["cpt_code"].apply(
        lambda x: int(float(x)) if str(x).replace(".", "", 1).isdigit() else x
    )

    summary = cpt.groupby("claim_id").agg(
        num_cpt_codes=("cpt_code", "count"),
        total_cpt_amount=("amount", "sum"),
        cpt_codes_list=("cpt_code", lambda x: ", ".join(x.astype(str)))
    ).reset_index()

    top_codes = cpt["cpt_code"].value_counts().head(top_k).index.tolist()
    cpt_top = cpt[cpt["cpt_code"].isin(top_codes)]
    pivot = cpt_top.pivot_table(
        index="claim_id",
        columns="cpt_code",
        values="cpt_code",
        aggfunc="count",
        fill_value=0,
    )
    pivot.columns = [f"cpt_{int(code)}_count" if str(code).replace('-', '').isdigit() else f"cpt_{code}_count"
                     for code in pivot.columns]
    pivot = pivot.reset_index()

    summary = summary.merge(pivot, on="claim_id", how="left")
    cpt_code_cols = [c for c in summary.columns if c.startswith("cpt_") and c.endswith("_count")]
    if cpt_code_cols:
        summary[cpt_code_cols] = summary[cpt_code_cols].fillna(0)
    return summary


def clear_data_caches() -> None:
    """Clear Streamlit data caches in the current process."""
    for fn in (load_all, build_master, build_events_timeline, get_cpt_summary):
        try:
            fn.clear()
        except Exception:
            pass
