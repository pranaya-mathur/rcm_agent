"""
backend_api.py — Lightweight JSON API for React demo frontend.

Run:
  python3 backend_api.py

Endpoints:
  GET /api/health
  GET /api/summary
"""

from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from data_loader import build_master


HOST = "0.0.0.0"
PORT = 8001


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
    master = build_master().reset_index(drop=True).copy()

    # KPI-like fields from available data
    denial_rate = float(master["is_denied"].mean() * 100)
    clean_claim_rate = float(master["is_clean_claim"].mean() * 100)
    recovered = float(master[master["appeal_success"] == True]["paid_amount"].sum())  # noqa: E712

    if "fraud_probability_improved" in master.columns:
        fraud_high = float((master["fraud_probability_improved"] > 0.7).mean() * 100)
    else:
        fraud_high = float((master["fraud_score"] > 0.8).mean() * 100)

    # Scrub bars
    scrub_bars = [
        {"label": "CPT-ICD Mismatch", "v": float(master["cpt_icd_mismatch"].mean() * 100)},
        {"label": "Auth Required Missing", "v": float((master["denial_reason"] == "Auth required").mean() * 100)},
        {"label": "High Amount Flags", "v": float(master["high_amount_flag"].mean() * 100)},
    ]

    # Denial table using historical denial rate by payer as proxy probability
    payer_den = master.groupby("insurance")["is_denied"].mean().to_dict()
    denial_rows: List[Dict[str, Any]] = []
    sample_denied = master.sort_values("claim_amount", ascending=False).head(20)
    for _, r in sample_denied.head(3).iterrows():
        p = float(payer_den.get(r["insurance"], master["is_denied"].mean()))
        risk = _risk_label(p)
        denial_rows.append(
            {
                "claim": f"CLM-{int(r['claim_id'])}",
                "payer": str(r["insurance"]),
                "risk": risk,
                "prob": f"{p*100:.1f}%",
                "action": "Add Auth + Coding QA" if risk == "HIGH" else ("Check docs before submit" if risk == "MEDIUM" else "Proceed standard"),
            }
        )

    # Appeals table from denied and not appealed population (proxy scoring)
    denied = master[master["is_denied"] == True].copy()  # noqa: E712
    not_appealed = denied[denied["is_appealed"] == False].copy()  # noqa: E712
    appeal_rows: List[Dict[str, Any]] = []
    if len(not_appealed) > 0:
        # Simple proxy expected recovery for front-end demo endpoint
        tmp = not_appealed.copy()
        base_success = master[master["is_appealed"] == True]["appeal_success"].mean() if (master["is_appealed"] == True).any() else 0.5  # noqa: E712
        tmp["p_success"] = np.clip(base_success + (tmp["claim_amount"] / tmp["claim_amount"].max()) * 0.15, 0.05, 0.95)
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
                }
            )

    payload = {
        "kpis": [
            {"label": "Denial Risk (Avg)", "value": f"{denial_rate:.1f}%", "delta": "-20% target path"},
            {"label": "Clean Claim Rate", "value": f"{clean_claim_rate:.1f}%", "delta": "towards 95%+"},
            {"label": "Appeal Recovery", "value": _fmt_money(recovered), "delta": "ML-prioritized opportunity"},
            {"label": "Fraud High-Risk", "value": f"{fraud_high:.1f}%", "delta": "improved blended score"},
        ],
        "scrubBars": scrub_bars,
        "denialRows": denial_rows,
        "appealRows": appeal_rows,
        "meta": {"claims_count": int(len(master))},
    }
    return payload


class Handler(BaseHTTPRequestHandler):
    def _set_headers(self, status: int = 200) -> None:
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_OPTIONS(self):
        self._set_headers(200)
        self.wfile.write(b"{}")

    def do_GET(self):
        if self.path == "/api/health":
            self._set_headers(200)
            self.wfile.write(json.dumps({"ok": True}).encode("utf-8"))
            return
        if self.path == "/api/summary":
            try:
                payload = build_summary_payload()
                self._set_headers(200)
                self.wfile.write(json.dumps(payload).encode("utf-8"))
            except Exception as e:
                self._set_headers(500)
                self.wfile.write(json.dumps({"ok": False, "error": str(e)}).encode("utf-8"))
            return

        self._set_headers(404)
        self.wfile.write(json.dumps({"ok": False, "error": "Not found"}).encode("utf-8"))


if __name__ == "__main__":
    server = ThreadingHTTPServer((HOST, PORT), Handler)
    print(f"RCM API running on http://{HOST}:{PORT}")
    server.serve_forever()

