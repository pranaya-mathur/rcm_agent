from __future__ import annotations

from typing import Any, Dict, List, Optional


class BackEndAgent:
    """
    Stage 3: Revenue protection and financial outcomes.
    Focus: appeals, fraud, and payment reconciliation.
    """

    name = "Back-End Agent"

    def _build_appeal_letter(
        self,
        claim: Dict[str, Any],
        p_appeal_success: float,
        expected_recovery: float,
    ) -> str:
        denial_reason = claim.get("denial_reason", "Unknown")
        patient_id = claim.get("patient_id", "N/A")
        icd = claim.get("icd_code", "N/A")
        visit_type = claim.get("visit_type", "N/A")
        claim_id = claim.get("claim_id", "N/A")
        payer = claim.get("insurance", "Payer")
        return (
            f"Date: [Current Date]\n"
            f"To: Claims Appeals Department - {payer}\n"
            f"Re: FORMAL APPEAL FOR CLAIM #{claim_id}\n\n"
            f"Patient ID: {patient_id}\n"
            f"Visit Type: {visit_type}\n"
            f"ICD-10 Code: {icd}\n"
            f"Denial Reason: {denial_reason}\n\n"
            f"Dear Appeals Supervisor,\n\n"
            f"We are formally appealing the denial of the above-referenced claim. Our internal "
            f"clinical review indicates that the services provided were medically necessary and "
            f"conform to standard billing protocols for {visit_type} encounters.\n\n"
            f"Based on predictive RCM metrics (Success Probability: {p_appeal_success:.1%}), "
            f"this claim appears appropriate for reconsideration with expected recovery "
            f"of ${expected_recovery:,.2f}.\n\n"
            f"Sincerely,\nRevenue Cycle Management Team"
        )

    def run(self, claim: Dict[str, Any], predictions: Dict[str, Any], force_appeal: bool = False) -> Dict[str, Any]:
        fraud_probability_improved = float(predictions.get("fraud_probability_improved", 0.0) or 0.0)
        p_appeal_success = float(predictions.get("p_appeal_success", 0.0) or 0.0)
        expected_recovery = float(predictions.get("expected_recovery", 0.0) or 0.0)
        reconciliation_risk_probability = float(predictions.get("reconciliation_risk_probability", 0.0) or 0.0)

        appealable = (p_appeal_success >= 0.55 and expected_recovery > 0) or force_appeal
        high_fraud = fraud_probability_improved >= 0.7
        high_recon_risk = reconciliation_risk_probability >= 0.65

        actions: List[str] = []
        appeal_letter: Optional[str] = None

        if appealable:
            actions.append("Draft professional appeal letter and attach supporting documents")
            actions.append("Submit appeal for reprocessing (human approval recommended)")
            appeal_letter = self._build_appeal_letter(claim, p_appeal_success, expected_recovery)
            if force_appeal and not (p_appeal_success >= 0.55 and expected_recovery > 0):
                actions.append("Note: Appeal drafted via manual override (low predicted success)")

        if high_fraud:
            actions.append("Flag for enhanced manual review (possible fraud risk)")
            actions.append("Require supervisor sign-off before submission/appeal")

        if high_recon_risk:
            actions.append("Route remittance mapping check before final closure")

        summary = (
            f"Appealable={appealable}, fraud_high={high_fraud}, "
            f"reconciliation_review={high_recon_risk}."
        )
        handoff = "Back-end actions generated; route consolidated plan for supervisor approval."

        return {
            "stage": "back_end",
            "summary": summary,
            "actions": actions,
            "artifacts": {"appeal_letter": appeal_letter},
            "metrics": {
                "p_appeal_success": p_appeal_success,
                "expected_recovery": expected_recovery,
                "appealable": appealable,
                "high_fraud": high_fraud,
                "fraud_probability_improved": fraud_probability_improved,
                "reconciliation_review_required": high_recon_risk,
                "reconciliation_risk_probability": reconciliation_risk_probability,
            },
            "handoff": handoff,
        }

