"""Smoke import tests for core modules and pages."""

import importlib
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def test_core_imports():
    importlib.import_module("config")
    importlib.import_module("utils_ui")
    importlib.import_module("src.core.ml_engine")
    importlib.import_module("src.core.data_loader")
    importlib.import_module("src.core.claim_scoring")
    importlib.import_module("src.agents.langgraph_rcm_chatbot")


def test_page_imports():
    pages_dir = os.path.join(ROOT, "pages")
    assert os.path.isdir(pages_dir)
    for fname in sorted(os.listdir(pages_dir)):
        if not fname.endswith(".py"):
            continue
        path = os.path.join(pages_dir, fname)
        # Execute module load via importlib from file path would be heavy; compile check is enough here.
        with open(path, "r", encoding="utf-8") as f:
            compile(f.read(), path, "exec")


def test_agent_imports():
    importlib.import_module("src.agents.rcm_agent")
    importlib.import_module("src.agents.front_end_agent")
    importlib.import_module("src.agents.mid_cycle_agent")
    importlib.import_module("src.agents.back_end_agent")
    importlib.import_module("src.agents.clinical_nlp_agent")
    importlib.import_module("src.agents.custom_coding_agent")


def test_api_imports():
    importlib.import_module("src.api.backend_api")
