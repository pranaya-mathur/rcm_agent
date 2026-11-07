#!/usr/bin/env python3
"""Pre-train / verify models under src/core/models for demo readiness."""

from __future__ import annotations

import sys
import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from src.core.data_loader import build_master, get_cpt_summary, clear_data_caches
from src.core.claim_scoring import get_or_train_models, clear_model_bundle
from src.core.demo_db import db_signature


def main() -> int:
    clear_data_caches()
    clear_model_bundle()
    sig = db_signature()
    print(f"db_sig={sig}")
    master = build_master(db_sig=sig)
    cpt = get_cpt_summary(db_sig=sig)
    print(f"master_rows={len(master)} cpt_rows={len(cpt)}")
    bundle = get_or_train_models(master, cpt, train_if_missing=True)
    print("models_ready=", bundle.ready)
    print("artifact_keys=", sorted(k for k in bundle.artifacts if "model" in k or k.endswith("_meta")))
    return 0 if bundle.ready else 1


if __name__ == "__main__":
    raise SystemExit(main())
