/* DroneGym web app — talks to server.py, drives DGViz. */
"use strict";

const S = { presets: {}, models: [], defaultModel: null, battery: {}, runs: [],
            ranked: [], selected: null };

const $ = id => document.getElementById(id);
const OUTCOME_COLOR = { hit: "#50ff78", timeout: "#ffb84d", lost: "#ff6a6a",
  crash: "#ff6a6a", oob: "#ff6a6a", tumble: "#ff6a6a", escaped: "#ffb84d" };

async function getJSON(url, opts) {
  const r = await fetch(url, opts);
  if (!r.ok) throw new Error((await r.json().catch(() => ({}))).detail || r.statusText);
  return r.json();
}

// --- run ranking / labels (mirror dronegym.runner.rank_runs) ---------------
function rankRuns(runs, mode) {
  if (mode === "Total reward")
    return [...runs].sort((a, b) => (b.total_reward || -1e9) - (a.total_reward || -1e9));
  const hits = runs.filter(r => r.hit).sort((a, b) => a.time_to_hit - b.time_to_hit);
  const miss = runs.filter(r => !r.hit).sort((a, b) => (a.closest_approach || 1e9) - (b.closest_approach || 1e9));
  return hits.concat(miss);
}
function runLabel(r, i) {
  const spd = r.target_speed || 0, tag = spd ? ` @${spd.toFixed(0)}m/s` : "";
  const head = r.hit ? `HIT ${r.time_to_hit.toFixed(2)}s${tag}`
    : `${(r.outcome || "miss").toUpperCase()} ${(r.closest_approach || 0).toFixed(1)}m${tag}`;
  return `${String(i).padStart(2)}. ${head}  R${String(Math.round(r.total_reward || 0)).padStart(4)}  ${(r.drone && r.drone.name) || "?"}`;
}

// --- status line ------------------------------------------------------------
function showRun(run, mode, src) {
  const oc = run.outcome || "?", col = OUTCOME_COLOR[oc] || "#c8cdd7";
  const tth = run.time_to_hit ? `${run.time_to_hit.toFixed(2)}s` : "—";
  const spd = run.target_speed || 0;
  const sep = " &nbsp;·&nbsp; ";
  $("srcline").innerHTML =
    `${src}${sep}<span class="badge" style="color:${col}">${oc.toUpperCase()}</span>` +
    `${sep}hit ${tth}${sep}closest ${(run.closest_approach || 0).toFixed(2)} m` +
    `${sep}reward ${Math.round(run.total_reward || 0)}${sep}${spd ? "target " + spd.toFixed(1) + " m/s" : "static"}`;
  DGViz.load(run, mode);
}

// --- replay list ------------------------------------------------------------
function renderRuns() {
  S.ranked = rankRuns(S.runs, $("rank").value);
  const list = $("runlist");
  list.innerHTML = "";
  if (!S.ranked.length) { list.innerHTML = '<div style="color:#7a8091;padding:6px">no runs recorded yet</div>'; return; }
  S.ranked.forEach((r, i) => {
    const b = document.createElement("button");
    b.textContent = runLabel(r, i + 1);
    if (r === S.selected) b.classList.add("on");
    b.onclick = () => { S.selected = r; renderRuns(); showRun(r, "REPLAY", `replay · ${r._file || "run"}`); };
    list.appendChild(b);
  });
}

// --- config from sidebar ----------------------------------------------------
function cfgMode() { return document.querySelector("#cfgMode button.on").dataset.val; }
function scenario() { return document.querySelector("#scen button.on").dataset.val.toLowerCase(); }
function currentCfg() {
  if (cfgMode() === "Preset") return S.presets[$("preset").value];
  return {
    name: "custom",
    mass_g: +$("mass").value, prop_diameter_in: +$("prop").value,
    motor_kv: +$("kv").value, battery_v: S.battery[$("battery").value],
    cam_angle_deg: +$("cam").value, frame_size_mm: +$("frame").value,
  };
}

// --- live run ---------------------------------------------------------------
async function runLive() {
  const btn = $("runlive");
  btn.disabled = true; const label = btn.textContent; btn.textContent = "flying…";
  $("status").textContent = "";
  try {
    const scen = scenario();
    const run = await getJSON("/api/run", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        cfg: currentCfg(), scenario: scen,
        target_speed: scen === "intercept" ? +$("tspeed").value : null,
        model: $("model").value,
      }),
    });
    S.selected = null;
    showRun(run, "LIVE", `LIVE · ${(run.drone && run.drone.name) || "custom"} · ${run.model}`);
    $("status").textContent = `${(run.outcome || "?").toUpperCase()} — saved to replays`;
    S.runs = await getJSON("/api/runs"); renderRuns();
  } catch (e) {
    $("status").style.color = "#ff6a6a"; $("status").textContent = "error: " + e.message;
  } finally { btn.disabled = false; btn.textContent = label; }
}

// --- tabs + training completion ---------------------------------------------
function switchTab(name) {
  const fly = name === "fly";
  $("tab-fly").classList.toggle("on", fly);
  $("tab-train").classList.toggle("on", !fly);
  $("view-fly").hidden = !fly;
  $("view-train").hidden = fly;
}
async function onTrainDone(model, aborted) {
  if (aborted || !model) return;
  try {
    const m = await getJSON("/api/models");         // pull in the fresh checkpoint
    S.models = m.models;
    const sel = $("model"); sel.innerHTML = "";
    S.models.forEach(x => sel.add(new Option(x, x)));
    sel.value = model;
  } catch (e) { /* keep going */ }
  $("trainstatus").innerHTML = 'done — <a id="flynew" style="color:var(--color-cyan);cursor:pointer;text-decoration:underline">▸ fly this checkpoint</a>';
  $("flynew").onclick = () => { switchTab("fly"); runLive(); };
}

// --- wiring -----------------------------------------------------------------
function bindSeg(id, onchange) {
  document.querySelectorAll(`#${id} button`).forEach(b => {
    b.onclick = () => {
      document.querySelectorAll(`#${id} button`).forEach(x => x.classList.remove("on"));
      b.classList.add("on"); onchange(b.dataset.val);
    };
  });
}
function bindRange(id, valId, fmt) {
  const el = $(id), out = $(valId);
  const upd = () => { out.textContent = fmt ? fmt(el.value) : el.value; };
  el.addEventListener("input", upd); upd();
}

async function init() {
  DGViz.init();
  DGTrain.init(onTrainDone);
  $("tab-fly").onclick = () => switchTab("fly");
  $("tab-train").onclick = () => switchTab("train");
  try {
    const meta = await getJSON("/api/meta");
    S.battery = meta.battery; DGViz.setMaxRate(meta.maxRate);
    Object.keys(S.battery).forEach(k => $("battery").add(new Option(k, k)));
    $("battery").value = "6S (22.2V)" in S.battery ? "6S (22.2V)" : Object.keys(S.battery)[0];

    S.presets = await getJSON("/api/presets");
    const pnames = Object.keys(S.presets);
    pnames.forEach(n => $("preset").add(new Option(n, n)));
    if (pnames.includes("freestyle_5inch")) $("preset").value = "freestyle_5inch";
    // seed custom sliders from the default preset
    const base = S.presets[$("preset").value] || {};
    const setv = (id, v) => { if (v != null) $(id).value = v; };
    setv("mass", base.mass_g); setv("prop", base.prop_diameter_in); setv("kv", base.motor_kv);
    setv("cam", base.cam_angle_deg); setv("frame", base.frame_size_mm);

    const m = await getJSON("/api/models");
    S.models = m.models; S.defaultModel = m.default;
    S.models.forEach(x => $("model").add(new Option(x, x)));
    if (S.defaultModel) $("model").value = S.defaultModel;

    S.runs = await getJSON("/api/runs");
    renderRuns();
    if (S.ranked.length) { S.selected = S.ranked[0]; renderRuns(); showRun(S.ranked[0], "REPLAY", `replay · ${S.ranked[0]._file || "run"}`); }
  } catch (e) {
    $("status").style.color = "#ff6a6a"; $("status").textContent = "load error: " + e.message;
  }

  // sidebar interactions
  bindSeg("cfgMode", v => {
    $("presetRow").hidden = v !== "Preset";
    $("customFields").hidden = v !== "Custom";
  });
  bindSeg("scen", v => { $("tspeedRow").hidden = v !== "Intercept"; });
  bindRange("mass", "massV"); bindRange("prop", "propV", v => (+v).toFixed(1));
  bindRange("kv", "kvV"); bindRange("cam", "camV"); bindRange("frame", "frameV");
  bindRange("tspeed", "tspeedV", v => (+v).toFixed(1));
  $("rank").onchange = renderRuns;
  $("runlive").onclick = runLive;
}

window.addEventListener("DOMContentLoaded", init);
