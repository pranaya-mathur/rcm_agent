"""
groq_agent_summary.py — Generate a 2-page agent summary using Groq.

This module uses Groq's OpenAI-compatible API endpoint. It reads API key from
environment variable `GROQ_API_KEY` (never hardcode keys).

Output is returned as two markdown strings (page 1 + page 2) and can be
rendered/downloaded by the Streamlit UI.
"""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from typing import Any, Dict, Tuple


def _extract_first_json_object(text: str) -> Dict[str, Any]:
    """
    Extract the first {...} JSON object from the LLM response.
    Handles cases where model wraps output in code fences or extra text.
    """
    # Try strict JSON first
    text_strip = text.strip()
    if text_strip.startswith("{") and text_strip.endswith("}"):
        return json.loads(text_strip)

    # Fallback: locate first JSON object with a brace-based regex.
    match = re.search(r"\{.*\}", text_strip, flags=re.DOTALL)
    if not match:
        raise ValueError("No JSON object found in Groq response.")
    return json.loads(match.group(0))


def generate_two_page_agent_summary(
    agent_out: Any,
    *,
    model: str | None = None,
    temperature: float = 0.2,
) -> Tuple[str, str]:
    """
    Returns (page1_markdown, page2_markdown).

    agent_out is expected to be the AgentOutput dataclass from `rcm_agent.py`,
    but we only use duck-typed attributes:
    - recommendation
    - action_items
    - steps (list of AgentStep)
    - appeal_letter (optional)
    - metrics (dict)
    """
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Missing GROQ_API_KEY environment variable. Set it locally (do not paste secrets)."
        )

    model = model or os.getenv("GROQ_MODEL", "llama3-70b-8192")

    # Keep prompt compact: include only essential agent output.
    steps = getattr(agent_out, "steps", [])
    steps_summaries = [
        {
            "agent": getattr(s, "agent", ""),
            "step": getattr(s, "step", ""),
            "summary": getattr(s, "summary", ""),
        }
        for s in steps[:10]
    ]
    appeal_letter = getattr(agent_out, "appeal_letter", None)
    if appeal_letter:
        appeal_letter = str(appeal_letter)[:1200] + ("..." if len(str(appeal_letter)) > 1200 else "")

    agent_output_payload = {
        "recommendation": getattr(agent_out, "recommendation", ""),
        "action_items": getattr(agent_out, "action_items", []),
        "steps": steps_summaries,
        "appeal_letter": appeal_letter,
        "metrics": getattr(agent_out, "metrics", {}),
    }
    agent_output_json = json.dumps(agent_output_payload, ensure_ascii=False)

    payload: Dict[str, Any] = {
        "model": model,
        "temperature": temperature,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are an RCM analytics specialist. Create a 2-page executive summary in Hindi/English mix "
                    "for a dashboard demo. Use clear headings and keep it concise but action-oriented."
                ),
            },
            {
                "role": "user",
                "content": (
                    "Given the following agent output, generate a 2-page summary in Markdown.\n\n"
                    "Requirements:\n"
                    "1) Output MUST be valid JSON with keys: page1_markdown, page2_markdown.\n"
                    "2) Each page should be clearly headed (Page 1 / Page 2) inside the markdown.\n"
                    "3) Include a small 'PDF Alignment' table on Page 1 mapping stages to: "
                    "Title/Section. Stages: Predictive Denial, Smart Scrubbing, Appeals Prioritization, Fraud Detection, Agentic Workflow.\n"
                    "4) On Page 2 include: 'Demo Walkthrough' (what to click on the dashboard) and 'Expected Outputs'.\n"
                    "5) Must explicitly include the scrubbing recommendation text when present: "
                    "\"Add 'Auth required' to this claim and review documentation.\".\n\n"
                    "AgentOutput JSON:\n"
                    f"{agent_output_json}"
                ),
            },
        ],
    }

    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    data_bytes = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data_bytes, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            raw = resp.read().decode("utf-8")
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="ignore") if hasattr(e, "read") else ""
        raise RuntimeError(f"Groq API HTTPError: {e.code}. Body: {body[:500]}") from e
    except Exception as e:
        raise RuntimeError(f"Groq API request failed: {e}") from e

    parsed = json.loads(raw)
    content = parsed["choices"][0]["message"]["content"]
    obj = _extract_first_json_object(content)

    page1 = obj["page1_markdown"]
    page2 = obj["page2_markdown"]
    return page1, page2


def markdown_to_html_page(
    markdown_text: str,
    *,
    title: str,
) -> str:
    """
    Convert markdown to a simple printable HTML page without extra dependencies.

    NOTE: This does not do full markdown rendering; it uses <pre> for faithful layout.
    Users can Print -> Save as PDF.
    """
    safe_title = title.replace("<", "&lt;").replace(">", "&gt;")
    return f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>{safe_title}</title>
  <style>
    @page {{ size: A4; margin: 18mm; }}
    body {{ font-family: Arial, Helvetica, sans-serif; color: #0b0f19; }}
    pre {{ white-space: pre-wrap; word-wrap: break-word; font-size: 12.5px; line-height: 1.35; }}
    .subtitle {{ margin-top: -6px; color: #475569; font-size: 12px; }}
  </style>
</head>
<body>
  <h2 style="margin:0 0 6px 0;">{safe_title}</h2>
  <div class="subtitle">Generated from Agentic RCM Demo</div>
  <pre>{markdown_text}</pre>
</body>
</html>"""
