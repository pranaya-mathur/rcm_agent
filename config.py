"""
config.py — Configuration and constants for the RCM App.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# LLM Configurations
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama3-70b-8192")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3:70b")
# Supported: 'openai' | 'groq' | 'ollama'
DEFAULT_LLM = os.getenv("DEFAULT_LLM", "openai").strip().lower()


def llm_provider_label() -> str:
    return {"openai": "OpenAI", "groq": "Groq", "ollama": "Ollama"}.get(DEFAULT_LLM, DEFAULT_LLM)


def active_llm_model() -> str:
    if DEFAULT_LLM == "ollama":
        return OLLAMA_MODEL
    if DEFAULT_LLM == "groq":
        return GROQ_MODEL
    return OPENAI_MODEL

# Recommendation Strings
SCRUB_RECOMMENDATION_HIGH_RISK = "Add 'Auth required' to this claim and review documentation."
SCRUB_RECOMMENDATION_STANDARD = "Standard scrubbing + coding review"

# UI Configuration
PAGE_TITLE = "AI-Powered Smart RCM Dashboard"
PAGE_ICON = "🏥"
