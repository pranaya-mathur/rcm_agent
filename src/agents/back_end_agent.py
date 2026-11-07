from __future__ import annotations

from datetime import date
from typing import Any, Dict, List, Optional


class BackEndAgent:
    """
    Stage 3: Revenue protection and financial outcomes.
    Focus: appeals (draft only), fraud flags, and payment reconciliation routing.
    """

    name = "Back-End Agent"

    def _build_appeal_draft(
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
        amount = float(claim.get("claim_amount", 0.0) or 0.0)
        return (
            f"*** DRAFT — FOR HUMAN REVIEW ONLY — DO NOT SUBMIT ***\n"
            f"Date: {date.today().isoformat()}\n"
            f"To: Claims Appeals Department - {payer}\n"
            f"Re: DRAFT APPEAL FOR CLAIM #{claim_id}\n\n"
            f"Patient ID: {patient_id}\n"
            f"Visit Type: {visit_type}\n"
            f"ICD-10 Code: {icd}\n"
            f"Claim Amount: ${amount:,.2f}\n"
            f"Denial Reason: {denial_reason}\n\n"
            f"Dear Appeals Supervisor,\n\n"
            f"We are preparing a formal appeal of the denial for the above-referenced claim.\n\n"
            f"Evidence checklist (placeholders — complete before any external submission):\n"
            f"1. [PENDING HUMAN REVIEW] Clinical documentation supporting medical necessity — NOT verified by this system.\n"
            f"2. [PENDING HUMAN REVIEW] Coding validation (CPT/ICD alignment) — recommendation only; not an approved coding determination.\n"
            f"3. [PENDING HUMAN REVIEW] Authorization / eligibility verification — not confirmed against payer systems.\n"
            f"4. [PENDING] Supporting attachments — none generated or attached by this prototype.\n\n"
            f"Experimental model estimates (synthetic-data prototype only; not externally validated):\n"
            f"- Predicted appeal success probability: {p_appeal_success:.1%}\n"
            f"- Estimated recovery if successful: ${expected_recovery:,.2f}\n\n"
            f"This draft must be reviewed and approved by an authorized human reviewer before any payer submission.\n"
            f"This system does not submit appeals or fabricate supporting documents.\n\n"
            f"Sincerely,\nRevenue Cycle Management Team (Draft Generator)"
        )

    def run(self, claim: Dict[str, Any], predictions: Dict[str, Any], force_appeal: bool = False) -> Dict[str, Any]:
        fraud_probability_improved = float(predictions.get("fraud_probability_improved", 0.0) or 0.0)
        p_appeal_success = float(predictions.get("p_appeal_success", 0.0) or 0.0)
        expected_recovery = float(predictions.get("expected_recovery", 0.0) or 0.0)
        reconciliation_risk_probability = float(predictions.get("reconciliation_risk_probability", 0.0) or 0.0)

        is_denied = bool(claim.get("is_denied", False))
        is_appealed = bool(claim.get("is_appealed", False))
        appeal_success = bool(claim.get("appeal_success", False))
        already_resolved = is_appealed and appeal_success

        model_gate = p_appeal_success >= 0.55 and expected_recovery > 0
        # Never draft an appeal for a non-denied claim unless explicitly forced in a labeled demo override.
        appealable = False
        override_note = None
        if already_resolved and not force_appeal:
            appealable = False
        elif not is_denied and not force_appeal:
            appealable = False
        elif is_denied and is_appealed and not force_appeal:
            appealable = False
        elif is_denied and not is_appealed and model_gate:
            appealable = True
        elif force_appeal:
            appealable = True
            override_note = (
                "Manual demo override: appeal draft forced by authorized operator. "
                "Still requires documented human review before any external submission."
            )

        high_fraud = fraud_probability_improved >= 0.7
        high_recon_risk = reconciliation_risk_probability >= 0.65

        actions: List[str] = []
        appeal_letter: Optional[str] = None

        if appealable:
            actions.append("Prepare DRAFT appeal letter for human review (no auto-submission)")
            actions.append("HUMAN APPROVAL REQUIRED before any payer submission — approval is not persisted by this demo")
            appeal_letter = self._build_appeal_draft(claim, p_appeal_success, expected_recovery)
            if override_note:
                actions.append(override_note)
            elif not model_gate:
                actions.append("Note: Draft created despite low experimental success estimate (override path)")
            actions.append(
                "Label: appeal success / expected recovery figures are experimental synthetic-data estimates only"
            )
        elif not is_denied:
            actions.append("No appeal draft: claim is not marked denied")
        elif already_resolved:
            actions.append("No appeal draft: claim already marked appealed/successful")
        elif is_appealed:
            actions.append("No new appeal draft: claim already in appealed state")

        if high_fraud:
            actions.append("Flag for enhanced manual review (possible fraud risk — model estimate)")
            actions.append("Require supervisor sign-off before submission/appeal")

        if high_recon_risk:
            actions.append("Route remittance mapping check before final closure (risk score — not an auto-match)")

        summary = (
            f"Appeal draft eligible={appealable} (denied={is_denied}, appealed={is_appealed}), "
            f"fraud_high={high_fraud}, reconciliation_review={high_recon_risk}. "
            f"Estimates are experimental/synthetic-data only."
        )
        handoff = (
            "Back-end recommendations generated. Human supervisor approval is required; "
            "this prototype does not persist approvals or submit to payers."
        )

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
                "force_appeal": force_appeal,
                "appeals_estimate_note": predictions.get(
                    "appeals_estimate_note",
                    "Experimental / synthetic-data estimate only.",
                ),
            },
            "handoff": handoff,
        }
