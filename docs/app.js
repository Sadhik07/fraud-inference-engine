/* QuantStrike — client-side simulation of the inference engine.
   Mirrors the Python stack conceptually:
     - benchmark.py            -> per-precision latency / throughput sweep
     - models/tabular_net.py   -> a fraud score per transaction
     - features/graph_features -> ring_size, velocity, amount_zscore drivers
     - explain.py              -> SHAP-style driver bars
   No build step, no dependencies. */

(() => {
  "use strict";
  const $ = (s) => document.querySelector(s);
  const reduce = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  // ---- benchmark profile (target numbers; reproduce via src/benchmark.py) ----
  const MODES = [
    { key: "fp32", name: "TensorRT FP32", note: "baseline", lat: 78.0, thr: 290, prauc: 0.943 },
    { key: "fp16", name: "TensorRT FP16", note: "half precision", lat: 11.4, thr: 2100, prauc: 0.941 },
    { key: "int8", name: "TensorRT INT8", note: "quantized", lat: 5.3, thr: 4800, prauc: 0.936 },
  ];
  const maxLat = MODES[0].lat;

  function renderModes(run) {
    $("#modes").innerHTML = MODES.map((m) => {
      const width = run ? Math.max(4, (m.lat / maxLat) * 100) : 0;
      const speedup = (MODES[0].lat / m.lat).toFixed(1);
      return `<div class="mode ${m.key}">
        <div class="name">${m.name}<small>${m.note}</small></div>
        <div class="track"><div class="barfill" style="width:${width}%"></div>
          <span class="lbl">${run ? m.lat.toFixed(1) + " ms" : ""}</span></div>
        <div class="metric">${run ? `<b>${m.thr.toLocaleString()}</b>/s<br>PR-AUC ${m.prauc}<br>${speedup}×` : "—"}</div>
      </div>`;
    }).join("");
  }
  renderModes(false);
  $("#run").addEventListener("click", () => {
    renderModes(false);
    requestAnimationFrame(() => setTimeout(() => renderModes(true), 60));
  });

  // ---- transaction stream + scoring -------------------------------------
  const DRIVERS = ["ring_size", "amount_zscore", "txn_velocity_1h", "device_share",
                   "device_degree", "time_since_prev"];
  let ringPool = [];
  for (let i = 0; i < 4; i++) ringPool.push({ id: "ring-" + i, size: 18 + ((Math.random() * 90) | 0) });

  function makeTxn() {
    const inRing = Math.random() < 0.18;
    const amount = inRing ? 200 + Math.random() * 900 : Math.exp(2.6 + Math.random() * 1.4);
    const hour = inRing ? (Math.random() * 5) | 0 : (Math.random() * 24) | 0;
    const ring = inRing ? ringPool[(Math.random() * ringPool.length) | 0] : null;

    // pseudo-model: weighted logit over driver signals
    const f = {
      ring_size: ring ? ring.size : 1,
      amount_zscore: inRing ? 2 + Math.random() * 2 : (Math.random() * 2 - 0.5),
      txn_velocity_1h: inRing ? 4 + Math.random() * 8 : Math.random() * 2,
      device_share: inRing ? 3 + Math.random() * 5 : Math.random(),
      device_degree: ring ? ring.size + 5 : 1 + ((Math.random() * 3) | 0),
      time_since_prev: inRing ? Math.random() * 60 : 200 + Math.random() * 3000,
    };
    const z = 0.06 * f.ring_size + 0.5 * f.amount_zscore + 0.18 * f.txn_velocity_1h
      + 0.22 * f.device_share + 0.04 * f.device_degree - 0.0008 * f.time_since_prev - 2.4;
    const p = 1 / (1 + Math.exp(-z));
    return { amount, hour, ring, p, f };
  }

  function tier(p) { return p > 0.7 ? "high" : p > 0.35 ? "med" : "low"; }

  const ticker = $("#ticker");
  function addRow() {
    const txn = makeTxn();
    const t = tier(txn.p);
    const el = document.createElement("div");
    el.className = "row" + (t === "high" ? " flag" : "");
    el.innerHTML = `<span class="mono">#${(Math.random() * 1e6 | 0).toString().padStart(6, "0")}</span>
      <span class="amt">$${txn.amount.toFixed(2)}</span>
      <span class="mono">${String(txn.hour).padStart(2, "0")}:00</span>
      <span class="tier ${t}">${t.toUpperCase()}</span>`;
    ticker.prepend(el);
    while (ticker.children.length > 9) ticker.lastChild.remove();
    if (t === "high") showCase(txn);
  }

  function showCase(txn) {
    $("#caseId").textContent = `flagged · risk tier HIGH · scored on INT8 engine`;
    $("#score").innerHTML = `${(txn.p * 100).toFixed(1)}<small>% fraud probability</small>`;
    const imp = DRIVERS.map((d) => {
      const raw = { ring_size: 0.06 * txn.f.ring_size, amount_zscore: 0.5 * txn.f.amount_zscore,
        txn_velocity_1h: 0.18 * txn.f.txn_velocity_1h, device_share: 0.22 * txn.f.device_share,
        device_degree: 0.04 * txn.f.device_degree, time_since_prev: 0.0008 * txn.f.time_since_prev };
      return { d, v: Math.abs(raw[d]) };
    }).sort((a, b) => b.v - a.v);
    const max = imp[0].v || 1;
    $("#shap").innerHTML = imp.map((x) =>
      `<div class="srow"><span class="f">${x.d}</span>
        <span class="sbar" style="width:${Math.max(4, (x.v / max) * 100)}%"></span></div>`).join("");
    $("#ring").innerHTML = txn.ring
      ? `linked component → <b>${txn.ring.id}</b> · ${txn.ring.size} cards share ${(2 + Math.random() * 3) | 0} devices → auto-routed to analyst queue`
      : `no ring linkage · isolated high-amount anomaly`;
  }

  setInterval(addRow, reduce ? 1400 : 850);
  for (let i = 0; i < 5; i++) addRow();
})();
