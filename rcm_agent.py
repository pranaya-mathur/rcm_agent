"""
rcm_agent.py — Simple (starter) agentic workflow for RCM.

This is a demo-friendly implementation that orchestrates multiple "agents"
over the existing ML predictions in ml_engine / data_loader.

No external agent framework is required for the starter demo; the structure
matches a typical LangGraph/CrewAI style pipeline (observe -> think -> plan
-> act -> learn), with human-in-the-loop style final actions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class AgentStep:
    agent: str
    step: str
    summary: str


@dataclass
class AgentOutput:
    steps: List[AgentStep]
    recommendation: str
    appeal_letter: Optional[str]
    action_items: List[str]
    metrics: Dict[str, Any]


class DenialPredictionAgent:
    name = "Denial Prediction Agent"

    def run(self, claim: Dict[str, Any], denial_probability: float) -> Dict[str, Any]:
        risk_level = "HIGH" if denial_probability >= 0.7 else ("MEDIUM" if denial_probability >= 0.4 else "LOW")
        if risk_level == "HIGH":
            rec = "Pre-submission review required (Auth docs + coding checks)."
        elif risk_level == "MEDIUM":
            rec = "Verify payer rules and supporting documents."
        else:
            rec = "Proceed with standard submission."

        return {
            "risk_level": risk_level,
            "recommendation": rec,
        }


class ScrubbingAgent:
    name = "Scrubbing Agent"

    def run(self, claim: Dict[str, Any], mismatch_probability: float, denial_probability: float,
            mismatch_threshold: float = 0.7, denial_threshold: float = 0.7) -> Dict[str, Any]:
        high = (denial_probability >= denial_threshold) or (mismatch_probability >= mismatch_threshold)
        if high:
            # Exact recommendation requested for demo.
            rec = "Is claim mein Auth required add kar do"
        else:
            rec = "Standard scrubbing + coding review"

        actions: List[str] = []
        if claim.get("cpt_icd_mismatch", False) and mismatch_probability >= mismatch_threshold:
            actions.append("Route to coding QA for CPT-ICD alignment")
        if claim.get("strict_insurance_flag", False):
            actions.append("Verify insurance eligibility/benefit rules")
        if claim.get("high_amount_flag", False):
            actions.append("Escalate high-amount claim for pre-auth review")

        return {
            "recommendation": rec,
            "actions": actions,
            "high": high,
        }


class AppealsAgent:
    name = "Appeals Agent"

    def run(self, claim: Dict[str, Any], p_appeal_success: float, expected_recovery: float,
            success_threshold: float = 0.55) -> Dict[str, Any]:
        appealable = p_appeal_success >= success_threshold and expected_recovery > 0

        denial_reason = claim.get("denial_reason", "Unknown")
        appeal_letter = None
        if appealable:
            patient_id = claim.get("patient_id", "N/A")
            icd = claim.get("icd_code", "N/A")
            visit_type = claim.get("visit_type", "N/A")

            appeal_letter = (
                f"Subject: Appeal Request for Claim {claim.get('claim_id')}\n"
                f"Denial Reason: {denial_reason}\n"
                f"Patient ID: {patient_id}\n"
                f"Visit Type: {visit_type}\n"
                f"ICD Code: {icd}\n\n"
                "Dear Supervisor,\n\n"
                "We respectfully request reconsideration of the denied claim.\n"
                "Action plan (per ML prioritization):\n"
                "- Attach missing documentation and/or authorization evidence as required.\n"
                "- Ensure CPT-ICD coding alignment where mismatch risk is detected.\n\n"
                f"Predicted appeal success probability: {p_appeal_success:.2%}\n"
                f"Expected recovery (estimate): ${expected_recovery:,.2f}\n\n"
                "Sincerely,\nRCM Agent Team"
            )

        actions = []
        if appealable:
            actions.append("Draft appeal letter and attach supporting documents")
            actions.append("Submit appeal for reprocessing (human approval recommended)")

        return {
            "appealable": appealable,
            "actions": actions,
            "appeal_letter": appeal_letter,
        }


class FraudAgent:
    name = "Fraud Agent"

    def run(self, claim: Dict[str, Any], fraud_probability_improved: float,
            fraud_threshold: float = 0.7) -> Dict[str, Any]:
        high_fraud = fraud_probability_improved >= fraud_threshold
        actions: List[str] = []
        if high_fraud:
            actions.append("Flag for enhanced manual review (possible fraud risk)")
            actions.append("Require supervisor sign-off before submission/appeal")

        return {
            "high_fraud": high_fraud,
            "actions": actions,
        }


class ReconciliationAgent:
    name = "Reconciliation Agent"

    def run(self, reconciliation_risk_probability: float, threshold: float = 0.65) -> Dict[str, Any]:
        high_recon_risk = reconciliation_risk_probability >= threshold
        actions: List[str] = []
        if high_recon_risk:
            actions.append("Pre-flag claim for payment posting reconciliation review")
            actions.append("Route remittance mapping check before final closure")
        return {"high_recon_risk": high_recon_risk, "actions": actions}


class CodingValidationAgent:
    name = "Coding Validation Agent"

    def run(
        self,
        coding_reco: Optional[Dict[str, Any]],
        mismatch_probability: float,
        nlp_coding_reco: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        if not coding_reco and not nlp_coding_reco:
            return {"needs_review": mismatch_probability >= 0.7, "actions": []}

        needs_review = (
            bool(coding_reco.get("should_review", False)) if isinstance(coding_reco, dict) else False
        ) or (
            bool(nlp_coding_reco.get("should_review", False)) if isinstance(nlp_coding_reco, dict) else False
        ) or mismatch_probability >= 0.7
        actions: List[str] = []
        if needs_review:
            actions.append("Run coder QA review for ICD/CPT consistency")
            if isinstance(coding_reco, dict) and coding_reco.get("suggestions"):
                top_icd = coding_reco["suggestions"][0][0]
                actions.append(f"Consider ICD update review toward likely code: {top_icd}")
            if isinstance(nlp_coding_reco, dict) and nlp_coding_reco.get("suggestions"):
                top_nlp_icd = nlp_coding_reco["suggestions"][0][0]
                actions.append(f"Cross-check with NLP note-derived ICD suggestion: {top_nlp_icd}")
        return {"needs_review": needs_review, "actions": actions}


class CoordinatorAgent:
    """
    Orchestrates multiple sub-agents.

    Inputs are precomputed predictions for a single claim, so the agent stays fast.
    """

    name = "Coordinator Agent"

    def run(
        self,
        claim: Dict[str, Any],
        predictions: Dict[str, Any],
    ) -> AgentOutput:
        steps: List[AgentStep] = []
        action_items: List[str] = []

        denial_probability = float(predictions.get("denial_probability", 0.0))
        mismatch_probability = float(predictions.get("mismatch_probability", 0.0))
        fraud_probability_improved = float(predictions.get("fraud_probability_improved", 0.0))
        p_appeal_success = float(predictions.get("p_appeal_success", 0.0))
        expected_recovery = float(predictions.get("expected_recovery", 0.0))
        coding_reco = predictions.get("coding_recommendation")
        nlp_coding_reco = predictions.get("nlp_coding_recommendation")
        reconciliation_risk_probability = float(predictions.get("reconciliation_risk_probability", 0.0))

        # Observe
        steps.append(AgentStep(self.name, "Observe",
                                f"Claim {claim.get('claim_id')} observed with denial_reason={claim.get('denial_reason')}."))

        # Think + Plan
        denial_agent = DenialPredictionAgent()
        denial_out = denial_agent.run(claim, denial_probability)
        steps.append(AgentStep(denial_agent.name, "Think",
                                f"Denial risk={denial_out['risk_level']} (p={denial_probability:.1%})."))

        scrub_agent = ScrubbingAgent()
        scrub_out = scrub_agent.run(claim, mismatch_probability, denial_probability)
        steps.append(AgentStep(scrub_agent.name, "Plan",
                                f"Scrubbing recommendation prepared (mismatch_p={mismatch_probability:.1%})."))

        appeals_agent = AppealsAgent()
        appeals_out = appeals_agent.run(claim, p_appeal_success, expected_recovery)
        if appeals_out["appealable"]:
            steps.append(AgentStep(appeals_agent.name, "Plan",
                                    f"Appeal recommended (p_success={p_appeal_success:.1%})."))
        else:
            steps.append(AgentStep(appeals_agent.name, "Plan",
                                    "Appeal not recommended based on predicted success/recovery."))

        fraud_agent = FraudAgent()
        fraud_out = fraud_agent.run(claim, fraud_probability_improved)
        if fraud_out["high_fraud"]:
            steps.append(AgentStep(fraud_agent.name, "Plan",
                                    f"Fraud risk flagged (p={fraud_probability_improved:.1%})."))
        else:
            steps.append(AgentStep(fraud_agent.name, "Plan", "Fraud risk within acceptable range."))

        recon_agent = ReconciliationAgent()
        recon_out = recon_agent.run(reconciliation_risk_probability)
        if recon_out["high_recon_risk"]:
            steps.append(AgentStep(recon_agent.name, "Plan",
                                   f"Reconciliation risk flagged (p={reconciliation_risk_probability:.1%})."))
        else:
            steps.append(AgentStep(recon_agent.name, "Plan", "Reconciliation risk within acceptable range."))

        coding_agent = CodingValidationAgent()
        coding_out = coding_agent.run(coding_reco, mismatch_probability, nlp_coding_reco)
        if coding_out["needs_review"]:
            steps.append(AgentStep(coding_agent.name, "Plan", "Coding validation review triggered."))
        else:
            steps.append(AgentStep(coding_agent.name, "Plan", "Coding appears consistent with historical patterns."))

        # Act
        action_items.extend(scrub_out["actions"])
        action_items.extend(appeals_out["actions"])
        action_items.extend(fraud_out["actions"])
        action_items.extend(recon_out["actions"])
        action_items.extend(coding_out["actions"])

        # Human-in-the-loop guardrail:
        action_items.append("Supervisor approval required before any submission/appeal action")

        # Learn (demo placeholder)
        steps.append(AgentStep(self.name, "Learn",
                                "Outcome feedback will be stored for future calibration (demo placeholder)."))

        recommendation = scrub_out["recommendation"]

        return AgentOutput(
            steps=steps,
            recommendation=recommendation,
            appeal_letter=appeals_out.get("appeal_letter"),
            action_items=action_items,
            metrics={
                "denial_probability": denial_probability,
                "mismatch_probability": mismatch_probability,
                "fraud_probability_improved": fraud_probability_improved,
                "p_appeal_success": p_appeal_success,
                "expected_recovery": expected_recovery,
                "denial_risk_level": denial_out["risk_level"],
                "appealable": appeals_out["appealable"],
                "high_fraud": fraud_out["high_fraud"],
                "reconciliation_risk_probability": reconciliation_risk_probability,
                "reconciliation_review_required": recon_out["high_recon_risk"],
                "coding_review_required": coding_out["needs_review"],
                "coding_recommendation": coding_reco.get("recommendation") if isinstance(coding_reco, dict) else None,
                "nlp_coding_recommendation": nlp_coding_reco.get("recommendation") if isinstance(nlp_coding_reco, dict) else None,
            },
        )

