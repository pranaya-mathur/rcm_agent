"""
config.py — Configuration and constants for the RCM App.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# LLM Configurations
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama3-70b-8192")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3:70b")
DEFAULT_LLM = os.getenv("DEFAULT_LLM", "groq")  # 'groq' or 'ollama'

# Recommendation Strings
SCRUB_RECOMMENDATION_HIGH_RISK = "Add 'Auth required' to this claim and review documentation."
SCRUB_RECOMMENDATION_STANDARD = "Standard scrubbing + coding review"

# UI Configuration
PAGE_TITLE = "AI-Powered Smart RCM Dashboard"
PAGE_ICON = "🏥"
