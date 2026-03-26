"""
clinical_nlp_agent.py — NLP support for note-to-ICD coding suggestions.

Prototype approach:
- Train lightweight TF-IDF + LogisticRegression from historical clinical_notes -> icd_code
- Provide per-note top ICD candidates with confidence
- Include robust fallbacks for sparse/single-class datasets
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


class ConstantICDModel:
    """Fallback model for single-class NLP training data."""

    def __init__(self, class_label: str):
        self.class_label = class_label
        self.classes_ = np.array([class_label], dtype=object)

    def predict_proba(self, X):
        # Always 100% confidence in the single available class
        return np.ones((X.shape[0], 1), dtype=float)


def train_notes_to_icd_model(
    encounters_df: pd.DataFrame,
    claims_df: pd.DataFrame,
    icd_df: pd.DataFrame,
    min_samples: int = 50,
) -> Dict[str, Any]:
    """
    Train note -> ICD classifier using:
      claims(encounter_id) + encounters(clinical_notes) + icd(icd_code)
    """
    merged = (
        claims_df[["claim_id", "encounter_id"]]
        .merge(encounters_df[["encounter_id", "clinical_notes"]], on="encounter_id", how="left")
        .merge(icd_df[["claim_id", "icd_code"]], on="claim_id", how="left")
    )
    merged["clinical_notes"] = merged["clinical_notes"].fillna("").astype(str)
    merged["icd_code"] = merged["icd_code"].fillna("").astype(str)

    train_df = merged[(merged["clinical_notes"].str.strip() != "") & (merged["icd_code"].str.strip() != "")]

    # Insufficient data fallback
    if len(train_df) < min_samples:
        return {
            "status": "fallback",
            "reason": f"insufficient_samples<{min_samples}",
            "model": None,
            "vectorizer": None,
            "classes": [],
        }

    y = train_df["icd_code"].values
    classes = np.unique(y)

    vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=5000, min_df=2)
    X = vectorizer.fit_transform(train_df["clinical_notes"].values)

    if len(classes) < 2:
        model = ConstantICDModel(classes[0])
        return {
            "status": "single_class",
            "reason": "single_icd_class",
            "model": model,
            "vectorizer": vectorizer,
            "classes": list(model.classes_),
        }

    model = LogisticRegression(max_iter=200, n_jobs=1, class_weight="balanced")
    model.fit(X, y)

    return {
        "status": "ok",
        "reason": None,
        "model": model,
        "vectorizer": vectorizer,
        "classes": list(model.classes_),
    }


def predict_icd_from_notes_batch(
    notes: List[str],
    model_bundle: Dict[str, Any],
    top_k: int = 3,
) -> List[Dict[str, Any]]:
    """
    Batch prediction for clinical notes.
    Returns one dict per input note:
      {
        "suggestions": [(icd, score), ...],
        "confidence": float,
        "source_used": "nlp_note_model" | "fallback_none",
        "recommendation": str,
      }
    """
    if model_bundle.get("status") == "fallback":
        out = []
        for _ in notes:
            out.append({
                "suggestions": [],
                "confidence": 0.0,
                "source_used": "fallback_none",
                "recommendation": "NLP coding support unavailable (insufficient training samples).",
            })
        return out

    model = model_bundle.get("model")
    vectorizer = model_bundle.get("vectorizer")
    if model is None or vectorizer is None:
        return [{
            "suggestions": [],
            "confidence": 0.0,
            "source_used": "fallback_none",
            "recommendation": "NLP coding support unavailable.",
        } for _ in notes]

    notes_clean = [str(n or "") for n in notes]
    X = vectorizer.transform(notes_clean)
    proba = model.predict_proba(X)
    classes = model.classes_

    outputs: List[Dict[str, Any]] = []
    for i in range(proba.shape[0]):
        row = proba[i]
        idx = np.argsort(row)[::-1][:top_k]
        suggestions = [(str(classes[j]), float(row[j])) for j in idx]
        conf = float(suggestions[0][1]) if suggestions else 0.0
        outputs.append({
            "suggestions": suggestions,
            "confidence": conf,
            "source_used": "nlp_note_model",
            "recommendation": (
                f"NLP note-based coding suggestion ready (confidence {conf:.2f})."
                if suggestions else
                "NLP note-based coding suggestion unavailable."
            ),
        })
    return outputs


def build_nlp_coding_recommendation(
    claim_row: Dict[str, Any],
    note_prediction: Dict[str, Any],
    mismatch_probability: float,
) -> Dict[str, Any]:
    """
    Build claim-level NLP coding recommendation payload.
    """
    current_icd = str(claim_row.get("icd_code", "")).strip()
    suggestions = note_prediction.get("suggestions", [])
    confidence = float(note_prediction.get("confidence", 0.0))
    suggested_icd = suggestions[0][0] if suggestions else None

    should_review = mismatch_probability >= 0.7 or (
        suggested_icd is not None and suggested_icd != current_icd and confidence >= 0.35
    )

    if should_review and suggested_icd:
        recommendation = (
            f"NLP coding review suggested: note implies `{suggested_icd}` vs current `{current_icd}` "
            f"(confidence {confidence:.2f})."
        )
    elif should_review:
        recommendation = "NLP coding review suggested due to mismatch risk."
    else:
        recommendation = "NLP notes appear consistent with current coding."

    return {
        "current_icd": current_icd,
        "suggestions": suggestions,
        "confidence": confidence,
        "source_used": note_prediction.get("source_used", "nlp_note_model"),
        "should_review": should_review,
        "recommendation": recommendation,
    }

