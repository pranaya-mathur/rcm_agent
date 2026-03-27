import pytest
import os
import sys

# Add the current directory to sys.path to allow importing local modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

def test_core_imports():
    """Test that core modules can be imported without errors."""
    import app
    import utils_ui
    import config
    from src.core import ml_engine
    from src.core import data_loader
    from src.agents import langgraph_rcm_chatbot

def test_page_imports():
    """Test that all Streamlit pages can be imported."""
    pages_dir = "pages"
    if not os.path.exists(pages_dir):
        pytest.skip("Pages directory not found")
        
    for filename in os.listdir(pages_dir):
        if filename.endswith(".py") and not filename.startswith("__"):
            module_name = f"pages.{filename[:-3]}"
            __import__(module_name)

def test_agent_imports():
    """Test that all agents can be imported."""
    from src.agents import rcm_agent
    from src.agents import clinical_nlp_agent
    from src.agents import custom_coding_agent

def test_api_imports():
    """Test that API modules can be imported."""
    from src.api import backend_api
