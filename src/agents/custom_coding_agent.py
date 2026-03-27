"""
custom_coding_agent.py — Custom coding assistance for CPT-ICD validation.

This is a lightweight, data-driven coding assistant for prototype use:
- Learns CPT -> ICD likelihoods from historical claims data
- Suggests likely ICD codes for a claim based on its CPT lines
- Produces a confidence score and recommendation text
"""

from __future__ import annotations

from typing import Dict, List, Tuple, Any, Optional

import pandas as pd


def _build_prob_table(
    merged_df: pd.DataFrame,
    group_cols: List[str],
) -> Tuple[Dict[str, Dict[str, float]], Dict[str, int]]:
    """
    Build nested probability table from grouped counts.

    Example:
    - group_cols=["cpt_code"] => {cpt: {icd: prob}}
    - group_cols=["insurance", "cpt_code"] => {"Aetna||99213": {icd: prob}}
    """
    if merged_df.empty:
        return {}, {}

    counts = (
        merged_df.groupby(group_cols + ["icd_code"])
        .size()
        .reset_index(name="count")
    )
    totals = counts.groupby(group_cols)["count"].sum().reset_index(name="total")
    counts = counts.merge(totals, on=group_cols, how="left")
    counts["prob"] = counts["count"] / counts["total"]

    table: Dict[str, Dict[str, float]] = {}
    support_map: Dict[str, int] = {}
    for key_vals, grp in counts.groupby(group_cols):
        if not isinstance(key_vals, tuple):
            key_vals = (key_vals,)
        key = "||".join([str(v) for v in key_vals])
        table[key] = {
            str(row["icd_code"]): float(row["prob"])
            for _, row in grp.sort_values("prob", ascending=False).iterrows()
        }
        support_map[key] = int(grp["count"].sum())
    return table, support_map


def build_cpt_icd_knowledge(
    cpt_lines_df: pd.DataFrame,
    icd_df: pd.DataFrame,
    claims_df: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:
    """
    Build CPT -> ICD probability table from historical claim-level joins.
    """
    merged = cpt_lines_df.merge(icd_df, on="claim_id", how="inner")
    if merged.empty:
        return {"global": {}, "payer": {}}

    global_table, global_support = _build_prob_table(merged, ["cpt_code"])

    payer_table: Dict[str, Dict[str, float]] = {}
    payer_support: Dict[str, int] = {}
    if claims_df is not None and "claim_id" in claims_df.columns and "insurance" in claims_df.columns:
        merged_payer = merged.merge(claims_df[["claim_id", "insurance"]], on="claim_id", how="left")
        merged_payer["insurance"] = merged_payer["insurance"].fillna("UNKNOWN")
        payer_table, payer_support = _build_prob_table(merged_payer, ["insurance", "cpt_code"])

    payer_visit_table: Dict[str, Dict[str, float]] = {}
    payer_visit_support: Dict[str, int] = {}
    if claims_df is not None and all(col in claims_df.columns for col in ["claim_id", "insurance", "visit_type"]):
        merged_payer_visit = merged.merge(
            claims_df[["claim_id", "insurance", "visit_type"]],
            on="claim_id",
            how="left",
        )
        merged_payer_visit["insurance"] = merged_payer_visit["insurance"].fillna("UNKNOWN")
        merged_payer_visit["visit_type"] = merged_payer_visit["visit_type"].fillna("UNKNOWN")
        payer_visit_table, payer_visit_support = _build_prob_table(
            merged_payer_visit,
            ["insurance", "visit_type", "cpt_code"],
        )

    return {
        "global": global_table,
        "payer": payer_table,
        "payer_visit": payer_visit_table,
        "support": {
            "global": global_support,
            "payer": payer_support,
            "payer_visit": payer_visit_support,
        },
    }


def suggest_icd_for_claim(
    cpt_codes: List[str],
    knowledge_bundle: Dict[str, Any],
    payer: Optional[str] = None,
    visit_type: Optional[str] = None,
    min_support: int = 20,
    top_k: int = 3,
) -> Tuple[List[Tuple[str, float]], float, str, int]:
    """
    Aggregate CPT->ICD probabilities over a claim's CPT list.
    Returns top ICD suggestions and a confidence score.
    """
    scores: Dict[str, float] = {}
    votes = 0
    source_used = "global"

    global_table = knowledge_bundle.get("global", {}) if isinstance(knowledge_bundle, dict) else knowledge_bundle
    payer_table = knowledge_bundle.get("payer", {}) if isinstance(knowledge_bundle, dict) else {}
    payer_visit_table = knowledge_bundle.get("payer_visit", {}) if isinstance(knowledge_bundle, dict) else {}
    support = knowledge_bundle.get("support", {}) if isinstance(knowledge_bundle, dict) else {}
    global_support = support.get("global", {})
    payer_support = support.get("payer", {})
    payer_visit_support = support.get("payer_visit", {})
    payer = (payer or "UNKNOWN").strip()
    visit_type = (visit_type or "UNKNOWN").strip()
    support_used = 0

    for cpt in cpt_codes:
        cpt_key = str(cpt).strip()
        payer_visit_key = f"{payer}||{visit_type}||{cpt_key}"
        payer_key = f"{payer}||{cpt_key}"
        if payer_visit_key in payer_visit_table and int(payer_visit_support.get(payer_visit_key, 0)) >= min_support:
            mapping = payer_visit_table[payer_visit_key]
            source_used = "payer+visit_type"
            support_used = max(support_used, int(payer_visit_support.get(payer_visit_key, 0)))
        elif payer_key in payer_table and int(payer_support.get(payer_key, 0)) >= min_support:
            mapping = payer_table[payer_key]
            source_used = "payer"
            support_used = max(support_used, int(payer_support.get(payer_key, 0)))
        elif cpt_key in global_table:
            mapping = global_table[cpt_key]
            support_used = max(support_used, int(global_support.get(cpt_key, 0)))
        else:
            continue
        votes += 1
        for icd, prob in mapping.items():
            scores[icd] = scores.get(icd, 0.0) + prob

    if votes == 0 or not scores:
        return [], 0.0, source_used, support_used

    # Normalize by vote count to keep score in a predictable range.
    for icd in list(scores.keys()):
        scores[icd] = scores[icd] / votes

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
    confidence = float(ranked[0][1]) if ranked else 0.0
    return ranked, confidence, source_used, support_used


def build_coding_recommendation(
    claim_row: Dict[str, Any],
    cpt_codes: List[str],
    knowledge_bundle: Dict[str, Any],
    mismatch_probability: float,
    min_support: int = 20,
) -> Dict[str, Any]:
    """
    Generate recommendation payload for coding validation agent.
    """
    current_icd = str(claim_row.get("icd_code", "")).strip()
    payer = str(claim_row.get("insurance", "UNKNOWN"))
    visit_type = str(claim_row.get("visit_type", "UNKNOWN"))
    suggestions, confidence, source_used, support_used = suggest_icd_for_claim(
        cpt_codes,
        knowledge_bundle,
        payer=payer,
        visit_type=visit_type,
        min_support=min_support,
        top_k=3,
    )

    suggested_icd = suggestions[0][0] if suggestions else None
    should_review = mismatch_probability >= 0.7 or (suggested_icd is not None and suggested_icd != current_icd)

    if should_review and suggested_icd:
        recommendation = (
            f"Coding review suggested: current ICD `{current_icd}` vs likely `{suggested_icd}` "
            f"(confidence {confidence:.2f}, source={source_used}, support={support_used})."
        )
    elif should_review:
        recommendation = "Coding review suggested due to high mismatch probability."
    else:
        recommendation = "Current ICD appears consistent with historical CPT patterns."

    return {
        "current_icd": current_icd,
        "suggestions": suggestions,
        "confidence": confidence,
        "source_used": source_used,
        "support_used": support_used,
        "min_support": min_support,
        "payer": payer,
        "visit_type": visit_type,
        "should_review": should_review,
        "recommendation": recommendation,
    }

