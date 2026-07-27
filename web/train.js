/* DGTrain — live training over WebSocket, streamed onto a telemetry chart.
 * DGTrain.init(onDone) wires the Start/Stop button; each rollout pushes a point. */
const DGTrain = (function () {
  let cv, ctx, DPR; const CW = 1280, CH = 360;
  let ws = null, points = [], running = false, total = 1, onDone = () => {};
  const $ = id => document.getElementById(id);
  const fmtk = n => n >= 1000 ? (n / 1000).toFixed(n >= 100000 ? 0 : 1) + "k" : String(n);

  function drawChart() {
    ctx.setTransform(DPR, 0, 0, DPR, 0, 0);
    ctx.fillStyle = "#0d1016"; ctx.fillRect(0, 0, CW, CH);
    const L = 58, R = CW - 58, Tp = 30, B = CH - 34, W = R - L, H = B - Tp;
    ctx.font = '12px ui-monospace, "Cascadia Code", monospace'; ctx.textBaseline = "alphabetic";
    ctx.textAlign = "left"; ctx.fillStyle = "#ff8c32"; ctx.fillText("EPISODE REWARD", L, 18);
    ctx.textAlign = "right"; ctx.fillStyle = "#00c8ff"; ctx.fillText("SUCCESS RATE", R, 18);
    if (!points.length) { ctx.textAlign = "left"; ctx.fillStyle = "#4a5160"; ctx.fillText("awaiting telemetry…", L, Tp + H / 2); return; }

    const rew = points.map(p => p.reward);
    let rmin = Math.min(0, ...rew), rmax = Math.max(1, ...rew);
    const pad = (rmax - rmin) * 0.12 || 1; rmin -= pad; rmax += pad;
    const n = points.length;
    const X = i => n <= 1 ? L + W / 2 : L + W * i / (n - 1);
    const Yr = v => B - H * (v - rmin) / (rmax - rmin);
    const Ys = v => B - H * Math.max(0, Math.min(1, v));

    ctx.lineWidth = 1;
    for (let g = 0; g <= 4; g++) {
      const y = Tp + H * g / 4;
      ctx.strokeStyle = "rgba(120,130,150,0.13)"; ctx.beginPath(); ctx.moveTo(L, y); ctx.lineTo(R, y); ctx.stroke();
      ctx.fillStyle = "#7a8091"; ctx.textAlign = "right"; ctx.fillText((rmax - (rmax - rmin) * g / 4).toFixed(0), L - 8, y + 4);
      ctx.fillStyle = "#3f6d78"; ctx.textAlign = "left"; ctx.fillText((1 - g / 4).toFixed(1), R + 8, y + 4);
    }
    if (rmin < 0 && rmax > 0) { const y = Yr(0); ctx.strokeStyle = "rgba(120,130,150,0.32)"; ctx.beginPath(); ctx.moveTo(L, y); ctx.lineTo(R, y); ctx.stroke(); }

    ctx.textAlign = "center";
    for (let i = 1; i < n; i++) if (points[i].level > points[i - 1].level) {
      const x = X(i); ctx.strokeStyle = "rgba(255,140,50,0.4)"; ctx.setLineDash([4, 4]);
      ctx.beginPath(); ctx.moveTo(x, Tp); ctx.lineTo(x, B); ctx.stroke(); ctx.setLineDash([]);
      ctx.fillStyle = "#ff8c32"; ctx.fillText("L" + points[i].level, x, Tp - 3);
    }
    ctx.strokeStyle = "rgba(0,200,255,0.55)"; ctx.lineWidth = 1.5; ctx.beginPath();
    points.forEach((p, i) => { const x = X(i), y = Ys(p.success); i ? ctx.lineTo(x, y) : ctx.moveTo(x, y); }); ctx.stroke();
    ctx.strokeStyle = "#ff8c32"; ctx.lineWidth = 2; ctx.beginPath();
    points.forEach((p, i) => { const x = X(i), y = Yr(p.reward); i ? ctx.lineTo(x, y) : ctx.moveTo(x, y); }); ctx.stroke();
    const lx = X(n - 1), ly = Yr(points[n - 1].reward);
    ctx.fillStyle = "#ff8c32"; ctx.beginPath(); ctx.arc(lx, ly, 3.5, 0, 7); ctx.fill();
    ctx.fillStyle = "#7a8091"; ctx.textAlign = "center"; ctx.fillText("rollout iteration →", (L + R) / 2, B + 24);
  }

  function setBtn(run) {
    running = run; const b = $("trainbtn");
    b.textContent = run ? "■ Stop" : "▶ Start training";
    b.classList.toggle("stop", run);
  }
  function reset(tot) {
    points = []; total = tot || 1;
    ["m-iter", "m-steps", "m-fps", "m-reward", "m-level", "m-success"].forEach(id => $(id).textContent = "—");
    $("trainbar").style.width = "0%"; drawChart();
  }
  function endState() { setBtn(false); $("led-train").className = "led"; if (ws) { try { ws.close(); } catch (e) {} ws = null; } }

  function onMsg(m) {
    if (m.type === "start") { reset(m.total); $("led-train").className = "led busy"; $("trainstatus").style.color = ""; $("trainstatus").textContent = "initialising policy…"; }
    else if (m.type === "metric") {
      points.push(m);
      $("m-iter").textContent = m.iter; $("m-steps").textContent = fmtk(m.steps);
      $("m-fps").textContent = m.fps; $("m-reward").textContent = m.reward.toFixed(1);
      $("m-level").textContent = "L" + m.level; $("m-success").textContent = (m.success * 100).toFixed(0) + "%";
      $("trainbar").style.width = Math.min(100, 100 * m.steps / total) + "%";
      $("trainstatus").textContent = `training… ${fmtk(m.steps)} / ${fmtk(total)} · ${m.fps} steps/s`;
      drawChart();
    }
    else if (m.type === "done") { endState(); $("trainstatus").textContent = m.aborted ? "stopped." : `done — saved ${m.model}`; onDone(m.model, m.aborted); }
    else if (m.type === "error") { endState(); $("trainstatus").style.color = "var(--color-red)"; $("trainstatus").textContent = "error: " + m.message; }
  }

  function start() {
    if (running) { $("trainstatus").textContent = "stopping…"; endState(); return; }
    setBtn(true); $("trainstatus").style.color = ""; $("trainstatus").textContent = "connecting…";
    const steps = parseInt($("steps").value, 10);
    const proto = location.protocol === "https:" ? "wss" : "ws";
    ws = new WebSocket(`${proto}://${location.host}/ws/train`);
    ws.onopen = () => ws.send(JSON.stringify({ steps, scenario: "static" }));
    ws.onmessage = e => onMsg(JSON.parse(e.data));
    ws.onerror = () => { $("trainstatus").textContent = "connection error"; };
    ws.onclose = () => { if (running) endState(); };
  }

  function init(onDoneCb) {
    cv = $("train-chart"); ctx = cv.getContext("2d");
    DPR = Math.max(1, Math.min(2, window.devicePixelRatio || 1));
    cv.width = CW * DPR; cv.height = CH * DPR;
    onDone = onDoneCb || (() => {});
    $("trainbtn").onclick = start;
    drawChart();
  }
  return { init };
})();
