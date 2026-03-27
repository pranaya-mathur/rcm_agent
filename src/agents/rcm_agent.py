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
from src.agents.front_end_agent import FrontEndAgent
from src.agents.mid_cycle_agent import MidCycleAgent
from src.agents.back_end_agent import BackEndAgent


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
            success_threshold: float = 0.55, force_appeal: bool = False) -> Dict[str, Any]:
        
        # Decision logic: appealable if above threshold OR if user forced it.
        is_high_yield = p_appeal_success >= success_threshold and expected_recovery > 0
        appealable = is_high_yield or force_appeal

        denial_reason = claim.get("denial_reason", "Unknown")
        appeal_letter = None
        
        if appealable:
            patient_id = claim.get("patient_id", "N/A")
            icd = claim.get("icd_code", "N/A")
            visit_type = claim.get("visit_type", "N/A")
            claim_id = claim.get("claim_id", "N/A")
            payer = claim.get("insurance", "Payer")

            # Improved, more professional medical appeal template
            appeal_letter = (
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
                f"Action taken to address the denial reason ({denial_reason}):\n"
                f"1. Clinical documentation has been reviewed and is attached for reconsideration.\n"
                f"2. Coding validation has been performed to ensure alignment between CPT and ICD-10 codes.\n"
                f"3. Any required authorizations have been cross-referenced with your payer database.\n\n"
                f"Based on our predictive revenue cycle metrics (Success Probability: {p_appeal_success:.1%}), "
                f"we believe this claim meets all requirements for full reimbursement totaling ${expected_recovery:,.2f}.\n\n"
                f"Please re-process this claim at your earliest convenience. If you require further "
                f"information, please contact the RCM department immediately.\n\n"
                f"Sincerely,\n"
                f"Revenue Cycle Management Team\n"
                f"[Internal ML Agent ID: RCM-APPEAL-BETA]"
            )

        actions = []
        if appealable:
            actions.append("Draft professional appeal letter and attach supporting documents")
            actions.append("Submit appeal for reprocessing (human approval recommended)")
        if force_appeal and not is_high_yield:
            actions.append("Note: Appeal drafted via manual override (low predicted success)")

        return {
            "appealable": appealable,
            "is_high_yield": is_high_yield,
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
        force_appeal: bool = False,
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
        steps.append(
            AgentStep(
                self.name,
                "Observe",
                f"Claim {claim.get('claim_id')} observed with denial_reason={claim.get('denial_reason')}.",
            )
        )

        # Stage 1: Front-End
        front_stage = FrontEndAgent().run(claim, predictions)
        steps.append(AgentStep("Front-End Agent", "Think", front_stage["summary"]))
        steps.append(AgentStep("Front-End Agent", "Handoff", front_stage["handoff"]))
        action_items.extend(front_stage.get("actions", []))

        # Stage 2: Mid-Cycle
        mid_stage = MidCycleAgent().run(claim, predictions)
        steps.append(AgentStep("Mid-Cycle Agent", "Plan", mid_stage["summary"]))
        steps.append(AgentStep("Mid-Cycle Agent", "Handoff", mid_stage["handoff"]))
        action_items.extend(mid_stage.get("actions", []))

        # Stage 3: Back-End
        back_stage = BackEndAgent().run(claim, predictions, force_appeal=force_appeal)
        steps.append(AgentStep("Back-End Agent", "Act", back_stage["summary"]))
        steps.append(AgentStep("Back-End Agent", "Handoff", back_stage["handoff"]))
        action_items.extend(back_stage.get("actions", []))

        # Human-in-the-loop guardrail.
        action_items.append("Supervisor approval required before any submission/appeal action")

        # Learn (demo placeholder)
        steps.append(
            AgentStep(
                self.name,
                "Learn",
                "Stage outcomes captured for future calibration and workflow optimization (demo placeholder).",
            )
        )

        recommendation = front_stage.get("recommendation", "Standard scrubbing + coding review")
        back_metrics = back_stage.get("metrics", {})
        mid_metrics = mid_stage.get("metrics", {})
        front_metrics = front_stage.get("metrics", {})
        appeal_letter = back_stage.get("artifacts", {}).get("appeal_letter")

        return AgentOutput(
            steps=steps,
            recommendation=recommendation,
            appeal_letter=appeal_letter,
            action_items=action_items,
            metrics={
                "denial_probability": front_metrics.get("denial_probability", denial_probability),
                "mismatch_probability": front_metrics.get("mismatch_probability", mismatch_probability),
                "fraud_probability_improved": back_metrics.get("fraud_probability_improved", fraud_probability_improved),
                "p_appeal_success": back_metrics.get("p_appeal_success", p_appeal_success),
                "expected_recovery": back_metrics.get("expected_recovery", expected_recovery),
                "denial_risk_level": front_metrics.get("denial_risk_level", "LOW"),
                "appealable": back_metrics.get("appealable", False),
                "high_fraud": back_metrics.get("high_fraud", False),
                "reconciliation_risk_probability": back_metrics.get(
                    "reconciliation_risk_probability", reconciliation_risk_probability
                ),
                "reconciliation_review_required": back_metrics.get("reconciliation_review_required", False),
                "coding_review_required": mid_metrics.get("coding_review_required", False),
                "coding_recommendation": mid_metrics.get("coding_recommendation"),
                "nlp_coding_recommendation": mid_metrics.get("nlp_coding_recommendation"),
                "stage_front_end": front_stage,
                "stage_mid_cycle": mid_stage,
                "stage_back_end": back_stage,
            },
        )

