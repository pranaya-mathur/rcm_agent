from __future__ import annotations

from typing import Any, Dict, List


class MidCycleAgent:
    """
    Stage 2: Documentation and coding integrity.
    Focus: coding recommendations and note-to-code validation.
    """

    name = "Mid-Cycle Agent"

    def run(self, claim: Dict[str, Any], predictions: Dict[str, Any]) -> Dict[str, Any]:
        mismatch_probability = float(predictions.get("mismatch_probability", 0.0) or 0.0)
        coding_reco = predictions.get("coding_recommendation")
        nlp_coding_reco = predictions.get("nlp_coding_recommendation")

        needs_review = (
            (isinstance(coding_reco, dict) and bool(coding_reco.get("should_review", False)))
            or (isinstance(nlp_coding_reco, dict) and bool(nlp_coding_reco.get("should_review", False)))
            or mismatch_probability >= 0.7
        )

        actions: List[str] = []
        if needs_review:
            actions.append("Run coder QA review for ICD/CPT consistency")
            if isinstance(coding_reco, dict) and coding_reco.get("suggestions"):
                actions.append(f"Consider ICD update review toward likely code: {coding_reco['suggestions'][0][0]}")
            if isinstance(nlp_coding_reco, dict) and nlp_coding_reco.get("suggestions"):
                actions.append(f"Cross-check with NLP note-derived ICD suggestion: {nlp_coding_reco['suggestions'][0][0]}")

        summary = (
            "Coding/documentation review triggered."
            if needs_review
            else "Coding/documentation appears consistent with current rules."
        )
        handoff = "Mid-cycle validation complete; pass to Back-End for denial/appeals/fraud/reconciliation actions."

        return {
            "stage": "mid_cycle",
            "summary": summary,
            "actions": actions,
            "artifacts": {
                "coding_recommendation": coding_reco,
                "nlp_coding_recommendation": nlp_coding_reco,
            },
            "metrics": {
                "coding_review_required": needs_review,
                "coding_recommendation": coding_reco.get("recommendation") if isinstance(coding_reco, dict) else None,
                "nlp_coding_recommendation": (
                    nlp_coding_reco.get("recommendation") if isinstance(nlp_coding_reco, dict) else None
                ),
            },
            "handoff": handoff,
        }

