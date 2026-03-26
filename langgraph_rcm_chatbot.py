"""
langgraph_rcm_chatbot.py — LangGraph-based conversational RCM assistant (demo).

This does not replace your ML models. It orchestrates multiple "reasoning steps"
using LangGraph:
1) Decide which topics to answer (denial/scrubbing/appeals/fraud)
2) Generate a final response string from provided model outputs

All heavy scoring happens outside (in `app.py`) and is passed in as `insights`.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, TypedDict

from langgraph.graph import END, START, StateGraph


class ChatState(TypedDict, total=False):
    claim_id: int
    user_message: str
    # Provided by app.py (pre-scored)
    claim: Dict[str, Any]
    insights: Dict[str, Any]
    # Optional explanation (computed in app.py only when asked)
    denial_shap_top_drivers: Optional[List[Dict[str, Any]]]
    # Working fields
    topics: List[str]
    response: str


def _decide_topics_node(state: ChatState) -> Dict[str, Any]:
    msg = (state.get("user_message") or "").lower()

    topics: List[str] = []
    if any(k in msg for k in ["deny", "denial", "probability", "risk", "shap", "driver", "reason", "why"]):
        topics.append("denial")
    if any(k in msg for k in ["scrub", "mismatch", "cpt", "icd", "auth", "coding"]):
        topics.append("scrubbing")
    if any(k in msg for k in ["coding", "code", "icd", "cpt", "suggest"]):
        topics.append("coding")
    if any(k in msg for k in ["appeal", "recovery", "success", "expected", "rank"]):
        topics.append("appeals")
    if any(k in msg for k in ["fraud", "suspicious", "anomaly"]):
        topics.append("fraud")

    # Default: give the full picture.
    if not topics:
        topics = ["denial", "scrubbing", "coding", "appeals", "fraud"]

    return {"topics": topics}


def _format_risk_level(p: float) -> str:
    if p >= 0.7:
        return "HIGH"
    if p >= 0.4:
        return "MEDIUM"
    return "LOW"


def _generate_response_node(state: ChatState) -> Dict[str, Any]:
    ins = state.get("insights", {})
    claim = state.get("claim", {})
    topics = state.get("topics", [])

    denial_p = float(ins.get("denial_probability", 0.0) or 0.0)
    mismatch_p = float(ins.get("mismatch_probability", 0.0) or 0.0)
    fraud_p = float(ins.get("fraud_probability_improved", 0.0) or 0.0)
    p_appeal_success = ins.get("p_appeal_success", None)
    expected_recovery = ins.get("expected_recovery", None)
    coding_reco = ins.get("coding_recommendation", None)
    nlp_coding_reco = ins.get("nlp_coding_recommendation", None)

    denial_risk = _format_risk_level(denial_p)
    mismatch_high = mismatch_p >= 0.7
    denial_high = denial_p >= 0.7

    # Exact recommendation text (as requested).
    if denial_high or mismatch_high:
        scrub_recommendation = "Is claim mein Auth required add kar do"
    else:
        scrub_recommendation = "Standard scrubbing + coding review"

    parts: List[str] = []

    if "denial" in topics:
        parts.append(
            f"Denial Prediction: {denial_p*100:.1f}% (Risk: {denial_risk})"
        )
        if state.get("denial_shap_top_drivers") and len(state["denial_shap_top_drivers"]) > 0:
            drivers = state["denial_shap_top_drivers"][:6]
            driver_lines = [f"- {d.get('feature')} ({d.get('shap_value'):.4f})" for d in drivers]
            parts.append("Top Risk Drivers (SHAP):\n" + "\n".join(driver_lines))

    if "scrubbing" in topics:
        parts.append(
            f"CPT-ICD Mismatch Probability: {mismatch_p*100:.1f}%"
        )
        parts.append(f"Smart Scrubbing Recommendation: {scrub_recommendation}")

    if "coding" in topics:
        if isinstance(coding_reco, dict):
            parts.append(f"Coding Validation: {coding_reco.get('recommendation', 'No recommendation')}")
            parts.append(
                "Coding Source Context: "
                f"{coding_reco.get('source_used', 'unknown')} "
                f"(payer={coding_reco.get('payer', 'UNKNOWN')}, "
                f"visit_type={coding_reco.get('visit_type', 'UNKNOWN')}, "
                f"support={coding_reco.get('support_used', 0)}/"
                f"{coding_reco.get('min_support', 20)})"
            )
            suggestions = coding_reco.get("suggestions", [])
            if suggestions:
                suggestion_text = ", ".join([f"{icd} ({score:.2f})" for icd, score in suggestions[:3]])
                parts.append(f"Suggested ICD candidates: {suggestion_text}")
        if isinstance(nlp_coding_reco, dict):
            parts.append(f"NLP Note Coding: {nlp_coding_reco.get('recommendation', 'No recommendation')}")
            nlp_suggestions = nlp_coding_reco.get("suggestions", [])
            if nlp_suggestions:
                nlp_text = ", ".join([f"{icd} ({score:.2f})" for icd, score in nlp_suggestions[:3]])
                parts.append(f"NLP ICD candidates: {nlp_text}")
        else:
            parts.append("Coding Validation: No coding recommendation available for this claim.")

    if "appeals" in topics:
        if p_appeal_success is None or expected_recovery is None:
            parts.append("Appeals: Not available for this claim (it may already be appealed or not in the denial set).")
        else:
            parts.append(
                f"Appeal Success Probability: {float(p_appeal_success)*100:.1f}%"
            )
            parts.append(
                f"Expected Recovery (estimate): ${float(expected_recovery):,.2f}"
            )
            if float(p_appeal_success) >= 0.55:
                parts.append("Appeal Suggestion: Worth considering for this claim (based on predicted success).")
            else:
                parts.append("Appeal Suggestion: Lower expected success—consider reviewing documents/coding first.")

    if "fraud" in topics:
        parts.append(
            f"Fraud Probability (Improved): {fraud_p*100:.1f}%"
        )
        if fraud_p >= 0.7:
            parts.append("Fraud Action: Flag for enhanced manual review and supervisor sign-off.")
        else:
            parts.append("Fraud Action: No immediate fraud red flags from the model.")

    # Always include human-in-the-loop.
    parts.append("Next Step: Supervisor approval required before any submission/appeal action.")

    response = "\n\n".join(parts).strip()
    return {"response": response}


def build_rcm_langgraph() -> Any:
    """
    Build and compile the LangGraph workflow.
    """
    graph = StateGraph(ChatState)
    graph.add_node("decide_topics", _decide_topics_node)
    graph.add_node("generate_response", _generate_response_node)

    graph.add_edge(START, "decide_topics")
    graph.add_edge("decide_topics", "generate_response")
    graph.add_edge("generate_response", END)
    return graph.compile()


# Ready-to-use compiled graph (import-time cheap).
rcm_langgraph_app = build_rcm_langgraph()

