from __future__ import annotations

from typing import Any, Dict, List


class FrontEndAgent:
    """
    Stage 1: Patient access and pre-submission readiness.
    Focus: eligibility/auth/docs/scrubbing readiness.
    """

    name = "Front-End Agent"

    def run(self, claim: Dict[str, Any], predictions: Dict[str, Any]) -> Dict[str, Any]:
        denial_probability = float(predictions.get("denial_probability", 0.0) or 0.0)
        mismatch_probability = float(predictions.get("mismatch_probability", 0.0) or 0.0)
        reconciliation_risk_probability = float(predictions.get("reconciliation_risk_probability", 0.0) or 0.0)

        denial_risk = "HIGH" if denial_probability >= 0.7 else ("MEDIUM" if denial_probability >= 0.4 else "LOW")
        high_scrub_risk = denial_probability >= 0.7 or mismatch_probability >= 0.7
        recommendation = (
            "Add 'Auth required' to this claim and review documentation."
            if high_scrub_risk
            else "Standard scrubbing + coding review"
        )

        actions: List[str] = []
        if bool(claim.get("strict_insurance_flag", False)):
            actions.append("Verify insurance eligibility/benefit rules")
        if bool(claim.get("high_amount_flag", False)):
            actions.append("Escalate high-amount claim for pre-auth review")
        if bool(claim.get("cpt_icd_mismatch", False)) or mismatch_probability >= 0.7:
            actions.append("Route to coding QA for CPT-ICD alignment")
        if reconciliation_risk_probability >= 0.65:
            actions.append("Pre-flag claim for payment posting reconciliation review")

        handoff = (
            "Front-end checks complete; pass to Mid-Cycle for coding/documentation validation "
            "with eligibility/auth context."
        )

        return {
            "stage": "front_end",
            "summary": (
                f"Denial risk={denial_risk}, mismatch={mismatch_probability:.1%}, "
                f"front-end recommendation='{recommendation}'."
            ),
            "recommendation": recommendation,
            "actions": actions,
            "artifacts": {},
            "metrics": {
                "denial_probability": denial_probability,
                "denial_risk_level": denial_risk,
                "mismatch_probability": mismatch_probability,
                "reconciliation_risk_probability": reconciliation_risk_probability,
            },
            "handoff": handoff,
        }
