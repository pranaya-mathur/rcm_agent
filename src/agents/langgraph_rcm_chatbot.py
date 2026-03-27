"""
langgraph_rcm_chatbot.py — LangGraph + Groq conversational RCM copilot.
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional, TypedDict
from langgraph.graph import END, START, StateGraph
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from config import SCRUB_RECOMMENDATION_HIGH_RISK, SCRUB_RECOMMENDATION_STANDARD, GROQ_API_KEY, GROQ_MODEL, OLLAMA_MODEL, DEFAULT_LLM

class ChatState(TypedDict, total=False):
    claim_id: int
    user_message: str
    claim: Dict[str, Any]
    insights: Dict[str, Any]
    denial_shap_top_drivers: Optional[List[Dict[str, Any]]]
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

    if not topics:
        topics = ["denial", "scrubbing", "coding", "appeals", "fraud"]
    return {"topics": topics}

def _generate_response_node(state: ChatState) -> Dict[str, Any]:
    ins = state.get("insights", {})
    claim = state.get("claim", {})
    topics = state.get("topics", [])

    denial_p = float(ins.get("denial_probability", 0.0) or 0.0)
    mismatch_p = float(ins.get("mismatch_probability", 0.0) or 0.0)
    fraud_p = float(ins.get("fraud_probability_improved", 0.0) or 0.0)
    
    if denial_p >= 0.7 or mismatch_p >= 0.7:
        scrub_recommendation = SCRUB_RECOMMENDATION_HIGH_RISK
    else:
        scrub_recommendation = SCRUB_RECOMMENDATION_STANDARD

    drivers = ins.get("top_drivers", [])
    drivers_str = "\n".join([f"- {d['feature']}: {d['shap_value']:.4f}" for d in drivers[:5]]) if drivers else "None"

    context = f"""
    Claim ID: {state.get('claim_id')}
    Insurance: {claim.get('insurance')}
    Visit Type: {claim.get('visit_type')}
    Denial Probability: {denial_p*100:.1f}%
    Mismatch Probability: {mismatch_p*100:.1f}%
    Fraud Probability: {fraud_p*100:.1f}%
    Scrubbing Recommendation: {scrub_recommendation}
    
    Top Risk Drivers (SHAP):
    {drivers_str}
    
    Full Insights: {ins}
    """
    
    system_prompt = f"""
    You are an expert RCM (Revenue Cycle Management) Intelligence Assistant. 
    Analyze the provided claim context and explain the reasoning behind the predictions (denial, mismatch, fraud).
    Use the SHAP risk drivers to explain WHY a claim is at risk.
    Provide professional, actionable advice for an RCM manager.
    Focus on: {', '.join(topics)}.
    Keep your response concise (3-5 sentences), authoritative, and professional.
    """

    try:
        # Groq-first for demo consistency. Ollama is only used when explicitly requested.
        if DEFAULT_LLM == "ollama":
            from langchain_ollama import ChatOllama
            llm = ChatOllama(model=OLLAMA_MODEL)
        else:
            if not GROQ_API_KEY:
                raise RuntimeError("GROQ_API_KEY missing. Set it in .env to enable chatbot responses.")
            llm = ChatGroq(api_key=GROQ_API_KEY, model=GROQ_MODEL)

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"Context: {context}\nUser: {state.get('user_message')}")
        ]
        response = llm.invoke(messages).content
    except Exception as e:
        response = f"LLM Error: {str(e)}\n\n(Fallback) Denial Risk: {denial_p*100:.1f}%. Action: {scrub_recommendation}."

    return {"response": response}

def build_rcm_langgraph() -> Any:
    graph = StateGraph(ChatState)
    graph.add_node("decide_topics", _decide_topics_node)
    graph.add_node("generate_response", _generate_response_node)
    graph.add_edge(START, "decide_topics")
    graph.add_edge("decide_topics", "generate_response")
    graph.add_edge("generate_response", END)
    return graph.compile()

rcm_langgraph_app = build_rcm_langgraph()
