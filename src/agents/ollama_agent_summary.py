"""
ollama_agent_summary.py — Generate a 2-page agent summary using Ollama (local).

Assumes Ollama is running locally and accessible at:
  http://localhost:11434

No API keys are required.
"""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from typing import Any, Dict, Tuple


def _extract_first_json_object(text: str) -> Dict[str, Any]:
    text_strip = text.strip()
    if text_strip.startswith("{") and text_strip.endswith("}"):
        return json.loads(text_strip)
    match = re.search(r"\{.*\}", text_strip, flags=re.DOTALL)
    if not match:
        raise ValueError("No JSON object found in Ollama response.")
    return json.loads(match.group(0))


def generate_two_page_agent_summary_ollama(
    agent_out: Any,
    *,
    model: str = "llama3:70b",
    temperature: float = 0.2,
    base_url: str = "http://localhost:11434",
) -> Tuple[str, str]:
    """
    Returns (page1_markdown, page2_markdown).

    agent_out is duck-typed from src.agents.rcm_agent.AgentOutput.
    """
    steps = getattr(agent_out, "steps", [])
    steps_summaries = [
        {
            "agent": getattr(s, "agent", ""),
            "step": getattr(s, "step", ""),
            "summary": getattr(s, "summary", ""),
        }
        for s in steps[:10]
    ]

    recommendation = getattr(agent_out, "recommendation", "")
    action_items = getattr(agent_out, "action_items", [])
    appeal_letter = getattr(agent_out, "appeal_letter", None)
    if appeal_letter:
        appeal_letter = str(appeal_letter)[:1200] + ("..." if len(str(appeal_letter)) > 1200 else "")

    agent_output_payload: Dict[str, Any] = {
        "recommendation": recommendation,
        "action_items": action_items,
        "steps": steps_summaries,
        "appeal_letter": appeal_letter,
        "metrics": getattr(agent_out, "metrics", {}),
    }

    prompt = (
        "You are an RCM analytics specialist. Create a 2-page executive summary in Hindi/English mix "
        "for a dashboard demo. Use clear headings and keep it concise but action-oriented.\n\n"
        "Requirements:\n"
        "1) Output MUST be valid JSON with keys: page1_markdown, page2_markdown.\n"
        "2) Each page should be clearly headed (Page 1 / Page 2) inside the markdown.\n"
        "3) Include a small 'PDF Alignment' table on Page 1 mapping stages to: "
        "Title/Section. Stages: Predictive Denial, Smart Scrubbing, Appeals Prioritization, Fraud Detection, Agentic Workflow.\n"
        "4) On Page 2 include: 'Demo Walkthrough' (what to click on the dashboard) and 'Expected Outputs'.\n"
        "5) Must explicitly include the scrubbing recommendation text when present: "
        "\"Is claim mein Auth required add kar do\".\n\n"
        "AgentOutput JSON:\n"
        f"{json.dumps(agent_output_payload, ensure_ascii=False)}"
    )

    url = f"{base_url}/api/generate"
    body_bytes = json.dumps(
        {"model": model, "prompt": prompt, "stream": False, "options": {"temperature": temperature}},
        ensure_ascii=False,
    ).encode("utf-8")

    req = urllib.request.Request(
        url,
        data=body_bytes,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            raw = resp.read().decode("utf-8")
    except urllib.error.URLError as e:
        raise RuntimeError(
            f"Ollama request failed. Is Ollama running? {e}"
        ) from e
    except Exception as e:
        raise RuntimeError(f"Ollama API request failed: {e}") from e

    parsed = json.loads(raw)
    text = parsed.get("response", "") or ""
    obj = _extract_first_json_object(text)
    return obj["page1_markdown"], obj["page2_markdown"]

