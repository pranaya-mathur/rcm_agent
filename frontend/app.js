const e = React.createElement;

const defaultKpis = [
  { label: "Denial Risk (Avg)", value: "18.7%", delta: "-20% target path" },
  { label: "Clean Claim Rate", value: "93.4%", delta: "towards 95%+" },
  { label: "Appeal Recovery", value: "$1.42M", delta: "+14.8% QoQ" },
  { label: "Fraud High-Risk", value: "2.9%", delta: "improved blended score" },
];

const defaultScrubBars = [
  { label: "CPT-ICD Mismatch", v: 28 },
  { label: "Auth Required Missing", v: 46 },
  { label: "High Amount Flags", v: 21 },
];

const defaultDenialRows = [
  { claim: "CLM-10291", payer: "Aetna", risk: "HIGH", prob: "87.1%", action: "Add Auth + Coding QA" },
  { claim: "CLM-77342", payer: "UHC", risk: "MEDIUM", prob: "63.4%", action: "Check docs before submit" },
  { claim: "CLM-22018", payer: "Cigna", risk: "LOW", prob: "22.2%", action: "Proceed standard" },
];

const defaultAppealRows = [
  { claim: "CLM-55612", success: "78%", recovery: "$18,220", priority: "P1" },
  { claim: "CLM-10291", success: "74%", recovery: "$15,040", priority: "P1" },
  { claim: "CLM-87643", success: "66%", recovery: "$11,880", priority: "P2" },
];

function riskPill(risk) {
  const cls = risk === "HIGH" ? "high" : risk === "MEDIUM" ? "med" : "low";
  return e("span", { className: `pill ${cls}` }, risk);
}

function Card({ title, children, extra }) {
  return e("div", { className: "card" }, [
    e("div", { style: { display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 8 } }, [
      e("h3", null, title),
      extra || null,
    ]),
    children,
  ]);
}

function App() {
  const [data, setData] = React.useState({
    kpis: defaultKpis,
    scrubBars: defaultScrubBars,
    denialRows: defaultDenialRows,
    appealRows: defaultAppealRows,
    meta: {},
  });
  const [status, setStatus] = React.useState("loading");

  React.useEffect(() => {
    fetch("http://localhost:8001/api/summary")
      .then((r) => r.json())
      .then((json) => {
        setData({
          kpis: json.kpis || defaultKpis,
          scrubBars: json.scrubBars || defaultScrubBars,
          denialRows: json.denialRows || defaultDenialRows,
          appealRows: json.appealRows || defaultAppealRows,
          meta: json.meta || {},
        });
        setStatus("live");
      })
      .catch(() => setStatus("fallback"));
  }, []);

  return e("div", { className: "container" }, [
    e("div", { className: "header" }, [
      e("div", null, [
        e("h1", { className: "title" }, "AI-Powered RCM Command Center"),
        e("p", { className: "subtitle" }, "React prototype aligned to PDF: denial prevention, coding, appeals, fraud, agentic actions."),
      ]),
      e("span", { className: `badge ${status === "live" ? "success" : "warn"}` }, status === "live" ? "Live API" : (status === "loading" ? "Loading..." : "Fallback Data")),
    ]),

    e("div", { className: "grid kpis" },
      data.kpis.map((k) =>
        e("div", { className: "card", key: k.label }, [
          e("div", { className: "kpi-label" }, k.label),
          e("div", { className: "kpi-value" }, k.value),
          e("div", { className: "kpi-delta" }, k.delta),
        ])
      )
    ),

    e("div", { className: "grid two" }, [
      e(Card, {
        title: "Smart Claim Scrubbing",
        extra: e("span", { className: "badge warn" }, "Phase 3"),
        children: e("div", { className: "bars" },
          data.scrubBars.map((b) =>
            e("div", { className: "bar-row", key: b.label }, [
              e("div", { className: "bar-label" }, b.label),
              e("div", { className: "bar-track" },
                e("div", { className: "bar-fill", style: { width: `${b.v}%` } })
              ),
              e("div", { className: "bar-val" }, `${b.v}%`),
            ])
          )
        ),
      }),
      e(Card, {
        title: "Agentic Workflow Snapshot",
        extra: e("span", { className: "badge success" }, "Coordinator"),
        children: e("div", null, [
          e("p", { style: { color: "var(--muted)", marginTop: 0 } }, "Observe → Think → Plan → Act with human-in-loop."),
          e("ul", { style: { margin: 0, paddingLeft: 18, lineHeight: 1.8 } }, [
            e("li", null, "Denial Agent flags high-risk claims pre-submission"),
            e("li", null, "Coding Agent suggests ICD candidates from CPT + notes"),
            e("li", null, "Appeals Agent ranks by success probability and recovery"),
            e("li", null, "Fraud Agent highlights suspicious patterns"),
          ]),
        ]),
      }),
    ]),

    e("div", { className: "grid three" }, [
      e(Card, {
        title: "Predictive Denial",
        extra: e("span", { className: "badge danger" }, "XGBoost + SHAP"),
        children: e("table", null, [
          e("thead", null, e("tr", null, [e("th", null, "Claim"), e("th", null, "Payer"), e("th", null, "Risk"), e("th", null, "Prob"), e("th", null, "Action")])),
          e("tbody", null, data.denialRows.map((r) =>
            e("tr", { key: r.claim }, [
              e("td", null, r.claim),
              e("td", null, r.payer),
              e("td", null, riskPill(r.risk)),
              e("td", null, r.prob),
              e("td", null, r.action),
            ])
          )),
        ]),
      }),
      e(Card, {
        title: "Appeals Prioritization",
        extra: e("span", { className: "badge warn" }, "Phase 4"),
        children: e("table", null, [
          e("thead", null, e("tr", null, [e("th", null, "Claim"), e("th", null, "Success"), e("th", null, "Recovery"), e("th", null, "Priority")])),
          e("tbody", null, data.appealRows.map((r) =>
            e("tr", { key: r.claim }, [
              e("td", null, r.claim),
              e("td", null, r.success),
              e("td", null, r.recovery),
              e("td", null, e("span", { className: "pill med" }, r.priority)),
            ])
          )),
        ]),
      }),
      e(Card, {
        title: "Fraud + Forecast",
        extra: e("span", { className: "badge success" }, "Phase 5 + Bonus"),
        children: e("div", null, [
          e("p", { style: { marginTop: 0, color: "var(--muted)" } }, "Improved fraud probability (supervised + anomaly blend)"),
          e("div", { className: "bars" }, [
            e("div", { className: "bar-row" }, [
              e("div", { className: "bar-label" }, "Fraud High-Risk"),
              e("div", { className: "bar-track" }, e("div", { className: "bar-fill", style: { width: "29%" } })),
              e("div", { className: "bar-val" }, "2.9%"),
            ]),
            e("div", { className: "bar-row" }, [
              e("div", { className: "bar-label" }, "Forecast Confidence"),
              e("div", { className: "bar-track" }, e("div", { className: "bar-fill", style: { width: "85%" } })),
              e("div", { className: "bar-val" }, "85%"),
            ]),
          ]),
          e("p", { className: "footer-note" }, "What-if example: reducing 'Missing docs' denials by 50% projects notable revenue lift."),
        ]),
      }),
    ]),

    e("p", { className: "footer-note" },
      `Data source: ${status === "live" ? "http://localhost:8001/api/summary" : "built-in mock data"}`
    ),
  ]);
}

ReactDOM.createRoot(document.getElementById("root")).render(e(App));

