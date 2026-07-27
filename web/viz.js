/* DGViz — faithful canvas port of gui.py's three instrument panels.
 *
 * gui.py draws the right-hand side with 2D vector primitives (draw_line,
 * draw_circle, draw_rectangle, draw_text, draw_polyline, draw_arrow). Those map
 * 1:1 onto a canvas 2D context, so this re-implements draw_iso / draw_fpv /
 * draw_att — same isometric projection, colours, stick bars and attitude ring —
 * and drives them with a play/scrub bar. DGViz.load(run, mode) shows a run.
 */
const DGViz = (function () {
  let cv, ctx, DPR;
  const CW = 1280, CH = 576;
  let MAXRATE = 180 / Math.PI * 14.0;               // degrees(14 rad/s)

  // per-run state
  let RUN, POS, QUAT, BBOX, ACT, T, N, DT, t0, tvel, TRAD, TSPD, MOVING, HIT, TGT, MODE, FIT;
  // playback
  let idx = 0, playing = false, last = 0;
  let playBtn, scrub, tlab, spdSel;
  const SPEEDS = { "0.25x": 0.25, "0.5x": 0.5, "1x": 1, "2x": 2, "4x": 4 };

  // --- math ------------------------------------------------------------
  const C = Math.cos(Math.PI / 6), S = Math.sin(Math.PI / 6);
  function qrot(q) {
    let n = Math.hypot(q[0], q[1], q[2], q[3]) || 1,
        w = q[0] / n, x = q[1] / n, y = q[2] / n, z = q[3] / n;
    return [[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]];
  }
  function mv(R, v) {
    return [R[0][0] * v[0] + R[0][1] * v[1] + R[0][2] * v[2],
            R[1][0] * v[0] + R[1][1] * v[1] + R[1][2] * v[2],
            R[2][0] * v[0] + R[2][1] * v[1] + R[2][2] * v[2]];
  }
  const add = (a, b) => [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
  const isoP = p => [(p[0] - p[1]) * C, (p[0] + p[1]) * S - p[2]];
  function eulerDeg(q) {
    const R = qrot(q), d = 180 / Math.PI;
    return [Math.atan2(R[2][1], R[2][2]) * d,
            Math.asin(Math.max(-1, Math.min(1, R[2][0]))) * d,
            Math.atan2(R[1][0], R[0][0]) * d];
  }

  // --- isometric fit (gui.fit_view / IsoView) --------------------------
  function makeView(pts, w, h, margin) {
    const uv = pts.map(isoP);
    const u0 = Math.min(...uv.map(a => a[0])), u1 = Math.max(...uv.map(a => a[0]));
    const v0 = Math.min(...uv.map(a => a[1])), v1 = Math.max(...uv.map(a => a[1]));
    const du = Math.max(u1 - u0, 1e-6), dv = Math.max(v1 - v0, 1e-6);
    const k = Math.min((w - 2 * margin) / du, (h - 2 * margin) / dv);
    const ox = margin - k * u0 + ((w - 2 * margin) - k * du) / 2;
    const oy = margin - k * v0 + ((h - 2 * margin) - k * dv) / 2;
    return { k, px: p => { const uu = isoP(p); return [ox + k * uu[0], oy + k * uu[1]]; } };
  }
  function fitView() {
    const t = t0;
    const pts = [[-2, -3, 0], [-2, 3, 0], [Math.max(4, t[0] + 2), -3, 0], [0, 0, 4],
                 [t[0], t[1] - 2, 0], [t[0], t[1] + 2, 0], [t[0], t[1], t[2] + 1.5]];
    const mn = [1e9, 1e9, 1e9], mx = [-1e9, -1e9, -1e9];
    for (const p of POS) for (let j = 0; j < 3; j++) { mn[j] = Math.min(mn[j], p[j]); mx[j] = Math.max(mx[j], p[j]); }
    pts.push(mn.slice(), mx.slice());
    if (MOVING) { const L = T[T.length - 1]; pts.push([t0[0] + tvel[0] * L, t0[1] + tvel[1] * L, t0[2] + tvel[2] * L]); }
    let x0 = 1e9, x1 = -1e9, y0 = 1e9, y1 = -1e9;
    for (const p of pts) { x0 = Math.min(x0, p[0]); x1 = Math.max(x1, p[0]); y0 = Math.min(y0, p[1]); y1 = Math.max(y1, p[1]); }
    const corners = [[x0, y0, 0], [x0, y1, 0], [x1, y0, 0], [x1, y1, 0]];
    return { view: makeView(pts.concat(corners), 580, 540, 55), grid: [x0, x1, y0, y1] };
  }

  // --- canvas helpers --------------------------------------------------
  function font(px, wt) { ctx.font = (wt || '') + ' ' + px + 'px "Segoe UI",system-ui,sans-serif'; }
  function line(a, b, col, w) { ctx.strokeStyle = col; ctx.lineWidth = w || 1; ctx.beginPath(); ctx.moveTo(a[0], a[1]); ctx.lineTo(b[0], b[1]); ctx.stroke(); }
  function circle(c, r, stroke, fill, w) { ctx.beginPath(); ctx.arc(c[0], c[1], Math.max(0.1, r), 0, 7); if (fill) { ctx.fillStyle = fill; ctx.fill(); } if (stroke) { ctx.strokeStyle = stroke; ctx.lineWidth = w || 1; ctx.stroke(); } }
  function rect(a, b, stroke, fill, w) { const x = Math.min(a[0], b[0]), y = Math.min(a[1], b[1]), ww = Math.abs(b[0] - a[0]), hh = Math.abs(b[1] - a[1]); if (fill) { ctx.fillStyle = fill; ctx.fillRect(x, y, ww, hh); } if (stroke) { ctx.strokeStyle = stroke; ctx.lineWidth = w || 1; ctx.strokeRect(x, y, ww, hh); } }
  function text(s, x, y, px, col, wt) { font(px, wt); ctx.fillStyle = col; ctx.textBaseline = 'top'; ctx.fillText(s, x, y); }
  function polyline(pts, col, w) { if (pts.length < 2) return; ctx.strokeStyle = col; ctx.lineWidth = w || 1; ctx.beginPath(); ctx.moveTo(pts[0][0], pts[0][1]); for (let i = 1; i < pts.length; i++) ctx.lineTo(pts[i][0], pts[i][1]); ctx.stroke(); }
  function arrow(tip, tail, col, w, size) { line(tail, tip, col, w); const a = Math.atan2(tip[1] - tail[1], tip[0] - tail[0]), s = size || 8; line(tip, [tip[0] - s * Math.cos(a - 0.4), tip[1] - s * Math.sin(a - 0.4)], col, w); line(tip, [tip[0] - s * Math.cos(a + 0.4), tip[1] - s * Math.sin(a + 0.4)], col, w); }

  // --- WORLD (draw_iso) ------------------------------------------------
  function drawIso(ox, oy, i) {
    ctx.save(); ctx.translate(ox, oy);
    const V = FIT.view, g = FIT.grid, px = V.px;
    text("WORLD", 8, -2, 13, "#ff8c32", "600");
    const [x0, x1, y0, y1] = g;
    for (let gx = Math.floor(x0); gx <= Math.ceil(x1) + 0.01; gx += 2) line(px([gx, y0, 0]), px([gx, y1, 0]), "rgb(46,50,62)", 1);
    for (let gy = Math.floor(y0); gy <= Math.ceil(y1) + 0.01; gy += 2) line(px([x0, gy, 0]), px([x1, gy, 0]), "rgb(46,50,62)", 1);
    if (MOVING) { const L = T[T.length - 1]; line(px(t0), px([t0[0] + tvel[0] * L, t0[1] + tvel[1] * L, t0[2] + tvel[2] * L]), "rgba(255,140,50,0.27)", 1); }
    const tgt = TGT[i], sh = px([tgt[0], tgt[1], 0]), tp = px(tgt);
    circle(sh, Math.max(3, V.k * TRAD * 0.6), null, "rgba(0,0,0,0.35)");
    line(sh, tp, "rgba(120,120,135,0.43)", 1);
    circle(tp, Math.max(4, V.k * TRAD), "rgb(255,190,80)", "rgba(255,140,50,0.82)", 2);
    if (MOVING) { const nv = Math.hypot(tvel[0], tvel[1], tvel[2]); arrow(px([tgt[0] + tvel[0] / nv * 1.4, tgt[1] + tvel[1] / nv * 1.4, tgt[2] + tvel[2] / nv * 1.4]), tp, "rgba(255,140,50,0.78)", 2, 7); }
    const atEnd = i >= N - 1;
    if (atEnd && HIT) { circle(tp, Math.max(7, V.k * TRAD * 2.2), "rgba(255,240,120,0.86)", null, 3); text("HIT!", tp[0] + 12, tp[1] - 26, 22, "rgb(255,240,120)", "700"); }
    const step = Math.max(1, Math.floor((i + 1) / 300)), trail = [];
    for (let j = 0; j <= i; j += step) trail.push(px(POS[j]));
    polyline(trail, "rgba(0,200,255,0.67)", 2);
    const pos = POS[i], R = qrot(QUAT[i]);
    circle(px([pos[0], pos[1], 0]), 3, null, "rgba(0,0,0,0.35)");
    const arm = Math.max(0.28, 14.0 / Math.max(V.k, 1e-6)), c = px(pos);
    for (const [sx, sy] of [[1, 1], [1, -1], [-1, 1], [-1, -1]]) {
      const tip = add(pos, mv(R, [arm * 0.75 * sx, arm * 0.75 * sy, 0]));
      const col = sx > 0 ? "rgb(240,90,90)" : "rgb(205,205,220)", tp2 = px(tip);
      line(c, tp2, col, 3); circle(tp2, 3.5, null, col);
    }
    arrow(px(add(pos, mv(R, [arm * 1.6, 0, 0]))), c, "rgb(255,220,90)", 2, 8);
    ctx.restore();
  }

  // --- FPV (draw_fpv) --------------------------------------------------
  function bar(x, y, w, label, frac, txt, col, sym) {
    const bx = x + 34;
    text(label, x, y - 7, 13, "rgb(150,155,170)");
    rect([bx, y - 6], [bx + w, y + 6], "rgb(60,65,80)", null, 1);
    if (sym) { const cx = bx + w / 2; line([cx, y - 6], [cx, y + 6], "rgb(90,95,110)", 1); rect([Math.min(cx, cx + frac * w / 2), y - 4], [Math.max(cx, cx + frac * w / 2), y + 4], null, col); }
    else { rect([bx, y - 4], [bx + Math.max(1, frac * w), y + 4], null, col); }
    text(txt, bx + w + 7, y - 7, 13, "rgb(200,205,215)");
  }
  function drawFpv(ox, oy, i) {
    ctx.save(); ctx.translate(ox, oy);
    text("FPV CAMERA", 10, -2, 13, "#ff8c32", "600");
    const x0 = 10, y0 = 18, sq = 380, cx = x0 + sq / 2, cy = y0 + sq / 2;
    rect([x0, y0], [x0 + sq, y0 + sq], "rgb(70,75,90)", "rgb(13,16,22)", 1);
    for (const f of [1 / 3, 2 / 3]) { line([x0 + sq * f, y0], [x0 + sq * f, y0 + sq], "rgb(28,32,42)", 1); line([x0, y0 + sq * f], [x0 + sq, y0 + sq * f], "rgb(28,32,42)", 1); }
    circle([cx, cy], 16, "rgb(90,95,110)", null, 1);
    line([cx - 26, cy], [cx - 10, cy], "rgb(90,95,110)", 1); line([cx + 10, cy], [cx + 26, cy], "rgb(90,95,110)", 1);
    line([cx, cy - 26], [cx, cy - 10], "rgb(90,95,110)", 1); line([cx, cy + 10], [cx, cy + 26], "rgb(90,95,110)", 1);
    const bb = BBOX[i];
    if (i >= N - 1 && HIT) {
      // capture frame: the drone is on top of the target, so it leaves the FOV —
      // that's an acquisition, not a loss.
      text("TARGET ACQUIRED", cx - 84, cy - 11, 19, "rgb(80,255,120)", "600");
    } else if (bb && bb[3] >= 0.5) {
      const bxp = cx + bb[0] * sq / 2, byp = cy - bb[1] * sq / 2, sp = Math.max(6, bb[2] * sq / 2);
      rect([bxp - sp / 2, byp - sp / 2], [bxp + sp / 2, byp + sp / 2], "rgb(80,255,120)", null, 2);
      text("TARGET", bxp - sp / 2, byp - sp / 2 - 17, 13, "rgb(80,255,120)");
    } else if (bb) { text("TARGET LOST", cx - 62, cy - 11, 19, "rgb(255,90,90)", "600"); }
    const pos = POS[i], dist = Math.hypot(TGT[i][0] - pos[0], TGT[i][1] - pos[1], TGT[i][2] - pos[2]);
    let verdict = HIT ? ("HIT in " + (RUN.time_to_hit || 0).toFixed(2) + "s")
                      : ((RUN.outcome || "miss").toUpperCase() + "  closest " + (RUN.closest_approach || 0).toFixed(2) + "m");
    if (MOVING) verdict += "  |  tgt " + TSPD.toFixed(1) + "m/s";
    const hud = [MODE + "   t=" + T[i].toFixed(2) + "s   dist=" + dist.toFixed(1) + "m",
                 ((RUN.drone && RUN.drone.name) || "?") + "  |  " + (RUN.model || "?"),
                 verdict + "   reward " + Math.round(RUN.total_reward || 0)];
    let ty = y0 + sq + 10;
    for (const l of hud) { text(l, x0, ty, 14, "rgb(200,205,215)"); ty += 19; }
    const act = ACT[i];
    if (act) {
      const bw = sq - 34 - 74, sy0 = ty + 10, thr = (act[0] + 1) / 2;
      bar(x0, sy0 + 6, bw, "THR", thr, (thr * 100).toFixed(0) + "%", "rgba(255,190,80,0.9)", false);
      const names = ["ROL", "PIT", "YAW"];
      for (let k = 1; k <= 3; k++) { const v = act[k]; bar(x0, sy0 + 6 + 17 * k, bw, names[k - 1], v, (v * MAXRATE >= 0 ? "+" : "") + (v * MAXRATE).toFixed(0) + " deg/s", "rgba(0,200,255,0.9)", true); }
    }
    ctx.restore();
  }

  // --- ATTITUDE (draw_att) --------------------------------------------
  function drawAtt(ox, oy, i) {
    ctx.save(); ctx.translate(ox, oy);
    const x0 = 10, y0 = 34, sq = 250, cx = x0 + sq / 2, cy = y0 + sq / 2, k = 62;
    text("ATTITUDE", x0, 8, 15, "rgb(255,200,90)", "600");
    rect([x0, y0], [x0 + sq, y0 + sq], "rgb(70,75,90)", "rgb(13,16,22)", 1);
    const px = v => [cx + k * ((v[0] - v[1]) * C), cy + k * ((v[0] + v[1]) * S - v[2])];
    const ring = [];
    for (let a = 0; a <= Math.PI * 2 + 0.01; a += Math.PI * 2 / 40) ring.push(px([1.35 * Math.cos(a), 1.35 * Math.sin(a), 0]));
    polyline(ring, "rgb(50,55,70)", 1);
    line(px([0, 0, 0]), px([1.35, 0, 0]), "rgb(80,86,105)", 1);
    text("+X", px([1.62, 0, 0])[0], px([1.62, 0, 0])[1], 12, "rgb(80,86,105)");
    line(px([0, 0, 0]), px([0, 1.35, 0]), "rgb(60,66,84)", 1);
    const R = qrot(QUAT[i]);
    for (const [sx, sy] of [[1, 1], [1, -1], [-1, 1], [-1, -1]]) {
      const tip = mv(R, [0.72 * sx, 0.72 * sy, 0]), col = sx > 0 ? "rgb(240,90,90)" : "rgb(205,205,220)", tp = px(tip);
      line(px([0, 0, 0]), tp, col, 4); circle(tp, 5, null, col);
    }
    line(px([0, 0, 0]), px(mv(R, [0, 0, 0.5])), "rgb(120,200,140)", 2);
    arrow(px(mv(R, [1.15, 0, 0])), px([0, 0, 0]), "rgb(255,220,90)", 2, 8);
    const [roll, pitch, yaw] = eulerDeg(QUAT[i]);
    let ty = y0 + sq + 12;
    for (const [nm, val] of [["ROLL", roll], ["PITCH", pitch], ["YAW", yaw]]) {
      text(nm.padEnd(6) + (val >= 0 ? "+" : "") + val.toFixed(1) + " deg", x0, ty, 15, "rgb(200,205,215)"); ty += 22;
    }
    ctx.restore();
  }

  // --- frame + playback ------------------------------------------------
  function draw(i) {
    ctx.setTransform(DPR, 0, 0, DPR, 0, 0);
    ctx.fillStyle = "#0d1016"; ctx.fillRect(0, 0, CW, CH);
    drawIso(6, 20, i); drawFpv(600, 20, i); drawAtt(1010, 0, i);
  }
  function setIdx(i) { idx = Math.max(0, Math.min(N - 1, i)); scrub.value = idx; tlab.textContent = T[Math.round(idx)].toFixed(2) + "s"; draw(Math.round(idx)); }
  function loop(ts) {
    if (!playing) return;
    const dt = (ts - last) / 1000; last = ts;
    idx += (SPEEDS[spdSel.value] || 1) * dt / DT;
    if (idx >= N - 1) { setIdx(N - 1); stop(true); return; }
    setIdx(idx); requestAnimationFrame(loop);
  }
  function play() { if (idx >= N - 1) idx = 0; playing = true; last = performance.now(); playBtn.innerHTML = "&#10073;&#10073; Pause"; requestAnimationFrame(loop); }
  function stop(ended) { playing = false; playBtn.innerHTML = ended ? "&#8635; Replay" : "&#9654; Play"; }

  function precompute() {
    const tr = RUN.traj;
    POS = tr.pos; QUAT = tr.quat; BBOX = tr.bbox || []; ACT = tr.act || []; T = tr.t;
    N = POS.length; DT = RUN.dt || 0.02;
    t0 = RUN.target.pos0 || RUN.target.pos; tvel = RUN.target.vel || [0, 0, 0];
    TRAD = RUN.target.radius || 0.5;
    TSPD = Math.hypot(tvel[0], tvel[1], tvel[2]); MOVING = TSPD > 1e-6; HIT = !!RUN.hit;
    TGT = T.map(t => [t0[0] + tvel[0] * t, t0[1] + tvel[1] * t, t0[2] + tvel[2] * t]);
    FIT = fitView();
  }

  function init() {
    cv = document.getElementById('dg-cv'); ctx = cv.getContext('2d');
    DPR = Math.max(1, Math.min(2, window.devicePixelRatio || 1));
    cv.width = CW * DPR; cv.height = CH * DPR;
    playBtn = document.getElementById('dg-play');
    scrub = document.getElementById('dg-scrub');
    tlab = document.getElementById('dg-time');
    spdSel = document.getElementById('dg-speed');
    playBtn.onclick = () => { playing ? stop(false) : play(); };
    scrub.oninput = () => { stop(false); setIdx(parseInt(scrub.value)); };
    spdSel.onchange = () => { last = performance.now(); };
  }
  function load(run, mode) { RUN = run; MODE = mode || 'REPLAY'; precompute(); scrub.max = N - 1; setIdx(0); play(); }
  function setMaxRate(v) { if (v) MAXRATE = v; }

  return { init, load, setMaxRate };
})();
