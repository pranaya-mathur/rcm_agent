"""
backend_api.py — Lightweight JSON API for React demo frontend.

Run from repo root:
  python -m src.api.backend_api

Endpoints:
  GET /api/health
  GET /api/summary
  GET /api/agent/claim/<claim_id>
  POST /api/score-claim
"""

from __future__ import annotations

import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, List

import numpy as np

from src.core.data_loader import build_master, clear_data_caches
from src.core.demo_db import upsert_live_claim, db_signature
from src.core import ml_engine
from src.core.claim_scoring import (
    run_coordinator_for_claim,
    run_coordinator_for_payload,
    validate_claim_payload,
)
from src.core.value_coercion import coerce_bool, CoercionError


# Localhost-only by default for demo safety. Override with RCM_API_HOST=0.0.0.0 if needed.
HOST = os.getenv("RCM_API_HOST", "127.0.0.1")
PORT = int(os.getenv("RCM_API_PORT", "8001"))


def _fmt_money(v: float) -> str:
    if v >= 1_000_000:
        return f"${v/1_000_000:.2f}M"
    if v >= 1_000:
        return f"${v/1_000:.1f}K"
    return f"${v:,.0f}"


def _risk_label(p: float) -> str:
    if p >= 0.7:
        return "HIGH"
    if p >= 0.4:
        return "MEDIUM"
    return "LOW"


def build_summary_payload() -> Dict[str, Any]:
    master = build_master(db_sig=db_signature()).reset_index(drop=True).copy()

    denial_rate = float(master["is_denied"].mean() * 100)
    clean_claim_rate = float(master["is_clean_claim"].mean() * 100)
    recovered = float(master[master["appeal_success"] == True]["paid_amount"].sum())  # noqa: E712
    fraud_high = float((master["fraud_score"] > 0.8).mean() * 100)

    recon_model, recon_meta = ml_engine.load_reconciliation_risk_model()
    if recon_model is not None and recon_meta is not None:
        recon_probs = ml_engine.score_reconciliation_risk(master, recon_model, recon_meta["feature_cols"])
        recon_high = float((recon_probs >= 0.65).mean() * 100)
    else:
        posting_gap = (master["claim_amount"] - master["paid_amount"]).clip(lower=0)
        recon_high = float((posting_gap > 50).mean() * 100)

    scrub_bars = [
        {"label": "CPT-ICD Mismatch", "v": float(master["cpt_icd_mismatch"].mean() * 100)},
        {"label": "Auth Required Missing", "v": float((master["denial_reason"] == "Auth required").mean() * 100)},
        {"label": "High Amount Flags", "v": float(master["high_amount_flag"].mean() * 100)},
    ]

    # Denied claims only for denial table; still uses payer historical rate as a proxy label.
    payer_den = master.groupby("insurance")["is_denied"].mean().to_dict()
    denial_rows: List[Dict[str, Any]] = []
    sample_denied = master[master["is_denied"] == True].sort_values("claim_amount", ascending=False)  # noqa: E712
    for _, r in sample_denied.head(3).iterrows():
        p = float(payer_den.get(r["insurance"], master["is_denied"].mean()))
        risk = _risk_label(p)
        denial_rows.append(
            {
                "claim": f"CLM-{int(r['claim_id'])}",
                "payer": str(r["insurance"]),
                "risk": risk,
                "prob": f"{p*100:.1f}%",
                "action": "Add Auth + Coding QA" if risk == "HIGH" else (
                    "Check docs before submit" if risk == "MEDIUM" else "Proceed standard"
                ),
                "note": "Payer historical denial rate proxy — not live model score",
            }
        )

    denied = master[master["is_denied"] == True].copy()  # noqa: E712
    not_appealed = denied[denied["is_appealed"] == False].copy()  # noqa: E712
    appeal_rows: List[Dict[str, Any]] = []
    if len(not_appealed) > 0:
        tmp = not_appealed.copy()
        base_success = (
            master[master["is_appealed"] == True]["appeal_success"].mean()  # noqa: E712
            if (master["is_appealed"] == True).any()  # noqa: E712
            else 0.5
        )
        tmp["p_success"] = np.clip(
            base_success + (tmp["claim_amount"] / max(tmp["claim_amount"].max(), 1)) * 0.15, 0.05, 0.95
        )
        tmp["expected_recovery"] = tmp["p_success"] * tmp["claim_amount"] * 0.5
        tmp = tmp.sort_values("expected_recovery", ascending=False).head(3)
        for _, r in tmp.iterrows():
            p = float(r["p_success"])
            appeal_rows.append(
                {
                    "claim": f"CLM-{int(r['claim_id'])}",
                    "success": f"{p*100:.0f}%",
                    "recovery": _fmt_money(float(r["expected_recovery"])),
                    "priority": "P1" if p >= 0.7 else "P2",
                    "note": "Experimental / synthetic-data estimate only",
                }
            )

    return {
        "ok": True,
        "source": "live_api",
        "data_notice": "Synthetic demo dataset. Metrics are prototype analytics, not production KPIs.",
        "kpis": [
            {"label": "Denial Rate (dataset)", "value": f"{denial_rate:.1f}%", "delta": "synthetic rate"},
            {"label": "Clean Claim Rate", "value": f"{clean_claim_rate:.1f}%", "delta": "target 95%+ illustrative"},
            {"label": "Appeal Recovery (paid)", "value": _fmt_money(recovered), "delta": "from successful appeals rows"},
            {"label": "Reconciliation High-Risk", "value": f"{recon_high:.1f}%", "delta": "model/proxy posting risk"},
        ],
        "scrubBars": scrub_bars,
        "denialRows": denial_rows,
        "appealRows": appeal_rows,
        "meta": {
            "claims_count": int(len(master)),
            "fraud_high_risk_pct": round(fraud_high, 2),
            "synthetic": True,
        },
    }


def build_agent_claim_payload(claim_id: int) -> Dict[str, Any]:
    return run_coordinator_for_claim(int(claim_id), force_appeal=False, train_if_missing=True)


def build_realtime_agent_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    try:
        force = coerce_bool(payload.get("force_appeal", False), field="force_appeal", default=False)
    except CoercionError as e:
        return {"ok": False, "error": "Invalid claim payload", "errors": [str(e)]}
    return run_coordinator_for_payload(payload, force_appeal=force, train_if_missing=True)


class Handler(BaseHTTPRequestHandler):
    def _set_headers(self, status: int = 200) -> None:
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        # Local demo only — do not treat as a public API.
        origin = self.headers.get("Origin", "http://localhost:8080")
        if origin.startswith("http://localhost") or origin.startswith("http://127.0.0.1"):
            self.send_header("Access-Control-Allow-Origin", origin)
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def log_message(self, fmt: str, *args) -> None:
        # Avoid logging request bodies / secrets.
        sys_stderr_write = getattr(self, "_quiet", False)
        if not sys_stderr_write:
            super().log_message(fmt, *args)

    def do_OPTIONS(self):
        self._set_headers(200)
        self.wfile.write(b"{}")

    def do_GET(self):
        if self.path == "/api/health":
            self._set_headers(200)
            self.wfile.write(json.dumps({"ok": True, "host": HOST, "port": PORT}).encode("utf-8"))
            return
        if self.path == "/api/summary":
            try:
                payload = build_summary_payload()
                self._set_headers(200)
                self.wfile.write(json.dumps(payload).encode("utf-8"))
            except Exception as e:
                self._set_headers(500)
                self.wfile.write(json.dumps({"ok": False, "error": type(e).__name__}).encode("utf-8"))
            return
        if self.path.startswith("/api/agent/claim/"):
            try:
                claim_id_str = self.path.split("/api/agent/claim/")[1].strip("/")
                claim_id = int(claim_id_str)
                payload = build_agent_claim_payload(claim_id)
                # Strip heavy / non-JSON objects
                if "predictions" in payload and isinstance(payload["predictions"], dict):
                    payload["predictions"] = {
                        k: v for k, v in payload["predictions"].items() if k not in ("shap_explanation",)
                    }
                if "claim" in payload and isinstance(payload["claim"], dict):
                    # Keep a compact claim view
                    keys = (
                        "claim_id", "insurance", "visit_type", "icd_code", "age", "gender",
                        "claim_amount", "is_denied", "denial_reason", "is_appealed",
                    )
                    payload["claim"] = {k: payload["claim"].get(k) for k in keys}
                status = 200 if payload.get("ok", False) else 404
                self._set_headers(status)
                self.wfile.write(json.dumps(payload, default=str).encode("utf-8"))
            except Exception as e:
                self._set_headers(500)
                self.wfile.write(json.dumps({"ok": False, "error": type(e).__name__}).encode("utf-8"))
            return

        self._set_headers(404)
        self.wfile.write(json.dumps({"ok": False, "error": "Not found"}).encode("utf-8"))

    def do_POST(self):
        if self.path == "/api/score-claim":
            try:
                content_len = int(self.headers.get("Content-Length", "0"))
                raw_body = self.rfile.read(content_len) if content_len > 0 else b"{}"
                payload = json.loads(raw_body.decode("utf-8"))
                if not isinstance(payload, dict):
                    self._set_headers(400)
                    self.wfile.write(json.dumps({"ok": False, "error": "JSON payload must be an object"}).encode("utf-8"))
                    return

                try:
                    persist = coerce_bool(payload.get("persist", False), field="persist", default=False)
                except CoercionError as e:
                    self._set_headers(400)
                    self.wfile.write(
                        json.dumps({"ok": False, "error": "Invalid claim payload", "errors": [str(e)]}).encode("utf-8")
                    )
                    return
                ok, errors = validate_claim_payload(payload)
                if not ok:
                    self._set_headers(400)
                    self.wfile.write(json.dumps({"ok": False, "error": "Invalid claim payload", "errors": errors}).encode("utf-8"))
                    return

                result = build_realtime_agent_payload(payload)
                if not result.get("ok", False):
                    self._set_headers(400)
                    self.wfile.write(json.dumps(result, default=str).encode("utf-8"))
                    return

                if persist:
                    to_store = result.get("normalized_payload_for_persist") or {
                        k: v for k, v in payload.items() if k not in ("persist", "force_appeal")
                    }
                    upsert_live_claim(to_store)
                    clear_data_caches()
                    result["persisted"] = True
                else:
                    result["persisted"] = False

                # Keep response JSON-safe
                if "predictions" in result:
                    result["predictions"] = {
                        k: v for k, v in result["predictions"].items() if k not in ("shap_explanation",)
                    }
                result.pop("normalized_payload_for_persist", None)
                self._set_headers(200)
                self.wfile.write(json.dumps(result, default=str).encode("utf-8"))
                return
            except Exception as e:
                self._set_headers(500)
                self.wfile.write(json.dumps({"ok": False, "error": type(e).__name__}).encode("utf-8"))
                return

        self._set_headers(404)
        self.wfile.write(json.dumps({"ok": False, "error": "Not found"}).encode("utf-8"))


if __name__ == "__main__":
    if HOST not in ("127.0.0.1", "localhost") and os.getenv("RCM_API_ALLOW_REMOTE") != "1":
        print(
            f"Refusing to bind {HOST} without RCM_API_ALLOW_REMOTE=1 "
            "(demo API is unauthenticated)."
        )
        HOST_SAFE = "127.0.0.1"
    else:
        HOST_SAFE = HOST
    server = ThreadingHTTPServer((HOST_SAFE, PORT), Handler)
    print(f"RCM API running on http://{HOST_SAFE}:{PORT} (local demo; unauthenticated)")
    server.serve_forever()
