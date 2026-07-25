"""DroneGym GUI.

Configure a drone (preset or custom parameters), fly it live on any trained
model checkpoint, and browse/replay recorded runs ranked by performance.
Left: config + model + replay browser. Right: isometric 3D world view and
the drone's-eye FPV view (bbox on a synthetic frame), side by side.

    py gui.py             launch
    py gui.py --smoke     build UI, render a few frames on synthetic data, exit
"""

import glob
import os
import sys

import numpy as np
import yaml
import dearpygui.dearpygui as dpg

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from dronegym import runner
from dronegym.camera import _quat_to_rot

BATTERY = {"1S (3.7V)": 3.7, "2S (7.4V)": 7.4, "3S (11.1V)": 11.1,
           "4S (14.8V)": 14.8, "6S (22.2V)": 22.2}
SPEEDS = {"0.25x": 0.25, "0.5x": 0.5, "1x": 1.0, "2x": 2.0, "4x": 4.0}
RUNS_DIR = os.path.join(ROOT, "runs")
MODEL_DIRS = ("checkpoints", "models")

ISO_W, ISO_H = 580, 540
FPV_W, FPV_H = 400, 540
FPV_SQ = 380

G = {"mode": "idle", "ep": None, "run": None, "fidx": 0.0, "playing": False,
     "acc": 0.0, "view": None, "grid": (-2, 10, -4, 4), "run_map": {},
     "presets": {}, "models": {}}


# --- isometric projection ----------------------------------------------------
class IsoView:
    C, S = np.cos(np.radians(30)), np.sin(np.radians(30))

    def __init__(self, pts, w, h, margin=55):
        uv = np.array([self.iso(p) for p in pts])
        u0, v0 = uv.min(axis=0)
        u1, v1 = uv.max(axis=0)
        du, dv = max(u1 - u0, 1e-6), max(v1 - v0, 1e-6)
        self.k = min((w - 2 * margin) / du, (h - 2 * margin) / dv)
        self.ox = margin - self.k * u0 + ((w - 2 * margin) - self.k * du) / 2
        self.oy = margin - self.k * v0 + ((h - 2 * margin) - self.k * dv) / 2

    @classmethod
    def iso(cls, p):
        return (p[0] - p[1]) * cls.C, (p[0] + p[1]) * cls.S - p[2]

    def px(self, p):
        u, v = self.iso(p)
        return (self.ox + self.k * u, self.oy + self.k * v)


def fit_view(target, traj=None):
    t = np.asarray(target, dtype=float)
    pts = [(-2, -3, 0), (-2, 3, 0), (max(4, t[0] + 2), -3, 0),
           (0, 0, 4), (t[0], t[1] - 2, 0), (t[0], t[1] + 2, 0),
           (t[0], t[1], t[2] + 1.5)]
    if traj is not None and len(traj) > 1:
        a = np.asarray(traj, dtype=float)
        pts += [a.min(axis=0), a.max(axis=0)]
    P = np.array(pts, dtype=float)
    x0, y0 = P[:, 0].min(), P[:, 1].min()
    x1, y1 = P[:, 0].max(), P[:, 1].max()
    G["grid"] = (x0, x1, y0, y1)
    corners = [(x0, y0, 0), (x0, y1, 0), (x1, y0, 0), (x1, y1, 0)]
    G["view"] = IsoView(np.vstack([P, corners]), ISO_W, ISO_H)


# --- drawing ------------------------------------------------------------------
def draw_iso(pos=None, quat=None, target=None, t_radius=0.5, trail=(), flash=False):
    L = "iso_layer"
    dpg.delete_item(L, children_only=True)
    V = G["view"]
    if V is None:
        return
    x0, x1, y0, y1 = G["grid"]
    for gx in np.arange(np.floor(x0), np.ceil(x1) + 0.1, 2.0):
        dpg.draw_line(V.px((gx, y0, 0)), V.px((gx, y1, 0)), color=(46, 50, 62), parent=L)
    for gy in np.arange(np.floor(y0), np.ceil(y1) + 0.1, 2.0):
        dpg.draw_line(V.px((x0, gy, 0)), V.px((x1, gy, 0)), color=(46, 50, 62), parent=L)

    if target is not None:
        sh, tp = V.px((target[0], target[1], 0)), V.px(target)
        dpg.draw_circle(sh, max(3, V.k * t_radius * 0.6), fill=(0, 0, 0, 90),
                        color=(0, 0, 0, 0), parent=L)
        dpg.draw_line(sh, tp, color=(120, 120, 135, 110), parent=L)
        dpg.draw_circle(tp, max(4, V.k * t_radius), color=(255, 190, 80),
                        fill=(255, 140, 50, 210), thickness=2, parent=L)
        if flash:
            dpg.draw_circle(tp, max(7, V.k * t_radius * 2.2),
                            color=(255, 240, 120, 220), thickness=3, parent=L)
            dpg.draw_text((tp[0] + 12, tp[1] - 26), "HIT!", size=22,
                          color=(255, 240, 120), parent=L)

    if len(trail) > 1:
        step = max(1, len(trail) // 300)
        dpg.draw_polyline([V.px(p) for p in trail[::step]],
                          color=(0, 200, 255, 170), thickness=2, parent=L)

    if pos is not None:
        pos = np.asarray(pos, dtype=float)
        dpg.draw_circle(V.px((pos[0], pos[1], 0)), 3, fill=(0, 0, 0, 90),
                        color=(0, 0, 0, 0), parent=L)
        R = _quat_to_rot(np.asarray(quat, dtype=float))
        arm = max(0.28, 14.0 / max(V.k, 1e-6))       # visible at any zoom
        c = V.px(pos)
        for sx, sy in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
            tip = pos + R @ np.array([arm * 0.75 * sx, arm * 0.75 * sy, 0.0])
            col = (240, 90, 90) if sx > 0 else (205, 205, 220)  # front arms red
            tp2 = V.px(tip)
            dpg.draw_line(c, tp2, color=col, thickness=3, parent=L)
            dpg.draw_circle(tp2, 3.5, fill=col, color=(0, 0, 0, 0), parent=L)
        nose = pos + R @ np.array([arm * 1.6, 0.0, 0.0])
        dpg.draw_arrow(V.px(nose), c, color=(255, 220, 90), thickness=2, size=8, parent=L)


def draw_fpv(bbox=None, hud=()):
    L = "fpv_layer"
    dpg.delete_item(L, children_only=True)
    x0, y0, sq = 10, 10, FPV_SQ
    cx, cy = x0 + sq / 2, y0 + sq / 2
    dpg.draw_rectangle((x0, y0), (x0 + sq, y0 + sq), fill=(13, 16, 22),
                       color=(70, 75, 90), parent=L)
    for f in (1 / 3, 2 / 3):                          # rule-of-thirds guides
        dpg.draw_line((x0 + sq * f, y0), (x0 + sq * f, y0 + sq), color=(28, 32, 42), parent=L)
        dpg.draw_line((x0, y0 + sq * f), (x0 + sq, y0 + sq * f), color=(28, 32, 42), parent=L)
    dpg.draw_circle((cx, cy), 16, color=(90, 95, 110), parent=L)
    dpg.draw_line((cx - 26, cy), (cx - 10, cy), color=(90, 95, 110), parent=L)
    dpg.draw_line((cx + 10, cy), (cx + 26, cy), color=(90, 95, 110), parent=L)
    dpg.draw_line((cx, cy - 26), (cx, cy - 10), color=(90, 95, 110), parent=L)
    dpg.draw_line((cx, cy + 10), (cx, cy + 26), color=(90, 95, 110), parent=L)

    if bbox is not None and bbox[3] >= 0.5:
        bx_px = cx + bbox[0] * sq / 2
        by_px = cy - bbox[1] * sq / 2                 # image +y is up
        s_px = max(6, bbox[2] * sq / 2)
        dpg.draw_rectangle((bx_px - s_px / 2, by_px - s_px / 2),
                           (bx_px + s_px / 2, by_px + s_px / 2),
                           color=(80, 255, 120), thickness=2, parent=L)
        dpg.draw_text((bx_px - s_px / 2, by_px - s_px / 2 - 17), "TARGET",
                      size=13, color=(80, 255, 120), parent=L)
    elif bbox is not None:
        dpg.draw_text((cx - 62, cy - 11), "TARGET LOST", size=19,
                      color=(255, 90, 90), parent=L)

    ty = y0 + sq + 10
    for line in hud:
        dpg.draw_text((x0, ty), line, size=14, color=(200, 205, 215), parent=L)
        ty += 19


# --- config panel -------------------------------------------------------------
def load_presets():
    out = {}
    for p in sorted(glob.glob(os.path.join(ROOT, "configs", "*.yaml"))):
        try:
            with open(p) as f:
                cfg = yaml.safe_load(f)
        except (yaml.YAMLError, OSError):
            continue
        if isinstance(cfg, dict) and "mass_g" in cfg:
            out[cfg.get("name", os.path.basename(p)[:-5])] = cfg
    return out


def apply_preset(name):
    cfg = G["presets"].get(name)
    if not cfg:
        return
    dpg.set_value("w_mass", int(cfg["mass_g"]))
    dpg.set_value("w_prop", float(cfg["prop_diameter_in"]))
    dpg.set_value("w_kv", int(cfg["motor_kv"]))
    dpg.set_value("w_cam", int(cfg["cam_angle_deg"]))
    dpg.set_value("w_frame", int(cfg["frame_size_mm"]))
    dpg.set_value("w_batt", min(BATTERY, key=lambda k: abs(BATTERY[k] - cfg["battery_v"])))


def on_mode_change(sender=None, app_data=None):
    preset = dpg.get_value("cfg_mode") == "Preset"
    dpg.configure_item("w_preset", enabled=preset)
    for t in ("w_mass", "w_prop", "w_kv", "w_batt", "w_cam", "w_frame"):
        dpg.configure_item(t, enabled=not preset)
    if preset:
        apply_preset(dpg.get_value("w_preset"))


def current_cfg():
    preset = dpg.get_value("cfg_mode") == "Preset"
    return {"name": dpg.get_value("w_preset") if preset else "custom",
            "mass_g": float(dpg.get_value("w_mass")),
            "prop_diameter_in": float(dpg.get_value("w_prop")),
            "motor_kv": float(dpg.get_value("w_kv")),
            "battery_v": BATTERY[dpg.get_value("w_batt")],
            "cam_angle_deg": float(dpg.get_value("w_cam")),
            "frame_size_mm": float(dpg.get_value("w_frame"))}


# --- models + live runs --------------------------------------------------------
def status(msg, ok=True):
    dpg.set_value("w_status", msg)
    dpg.configure_item("w_status", color=(140, 220, 140) if ok else (255, 120, 120))


def scan_models():
    paths = []
    for d in MODEL_DIRS:
        paths += sorted(glob.glob(os.path.join(ROOT, d, "*.zip")))
    G["models"] = {os.path.relpath(p, ROOT): p for p in paths}
    items = list(G["models"]) or ["(no trained models found)"]
    dpg.configure_item("w_model", items=items)
    if dpg.get_value("w_model") not in items:
        dpg.set_value("w_model", items[0])


def start_live():
    sel = dpg.get_value("w_model")
    try:
        policy = runner.load_policy(G["models"].get(sel))
    except RuntimeError as e:
        status(str(e), ok=False)
        return
    cfg = current_cfg()
    ep = runner.EpisodeRunner(cfg, policy, model_name=os.path.basename(sel))
    G.update(mode="live", ep=ep, run=None, playing=True, acc=0.0)
    fit_view(ep.target)
    dpg.set_item_label("w_play", "Pause")
    dpg.configure_item("w_scrub", enabled=False)
    status(f"flying {cfg['name']} on {sel} ...")


def finish_live():
    ep = G["ep"]
    path = ep.save(RUNS_DIR)
    run = ep.to_dict()
    refresh_runs()
    n = len(run["traj"]["pos"])
    G.update(mode="replay", run=run, ep=None, fidx=float(n - 1), playing=False)
    dpg.configure_item("w_scrub", max_value=n - 1, enabled=True)
    dpg.set_value("w_scrub", n - 1)
    dpg.set_item_label("w_play", "Replay")
    status(f"{ep.outcome.upper()}  (saved {os.path.basename(path)})",
           ok=ep.outcome == "hit")


# --- replay browser -------------------------------------------------------------
def refresh_runs():
    ranked = runner.rank_runs(runner.load_runs(RUNS_DIR), dpg.get_value("w_rank"))
    G["run_map"] = {}
    items = []
    for i, r in enumerate(ranked, 1):
        if r.get("hit"):
            lab = f"{i:>2}. HIT {r['time_to_hit']:5.2f}s"
        else:
            lab = f"{i:>2}. {r.get('outcome', 'miss').upper()} {r.get('closest_approach', 0):.1f}m"
        lab += f"  R{r.get('total_reward', 0):>5.0f}  {r['drone'].get('name', '?')} [{r.get('model', '?')}]"
        while lab in G["run_map"]:
            lab += " "
        G["run_map"][lab] = r
        items.append(lab)
    dpg.configure_item("w_runs", items=items or ["(no runs recorded yet)"])


def load_replay(run, autoplay=True):
    n = len(run["traj"]["pos"])
    G.update(mode="replay", run=run, ep=None, fidx=0.0, playing=autoplay, acc=0.0)
    fit_view(run["target"]["pos"], traj=run["traj"]["pos"])
    dpg.configure_item("w_scrub", max_value=n - 1, enabled=True)
    dpg.set_value("w_scrub", 0)
    dpg.set_item_label("w_play", "Pause" if autoplay else "Play")


def play_selected():
    run = G["run_map"].get(dpg.get_value("w_runs"))
    if not run:
        status("no run selected", ok=False)
        return
    load_replay(run)
    status(f"replaying {run.get('_file', 'run')}")


# --- playback -------------------------------------------------------------------
def on_play():
    if G["mode"] == "live":
        G["playing"] = not G["playing"]
    elif G["mode"] == "replay" and G["run"]:
        if not G["playing"] and int(G["fidx"]) >= len(G["run"]["traj"]["pos"]) - 1:
            G["fidx"] = 0.0
        G["playing"] = not G["playing"]
    dpg.set_item_label("w_play", "Pause" if G["playing"] else "Play")


def on_scrub(sender, app_data):
    if G["mode"] == "replay" and G["run"]:
        G["fidx"] = float(app_data)
        G["playing"] = False
        dpg.set_item_label("w_play", "Play")


def update():
    dt_wall = dpg.get_delta_time() or 1 / 60
    spd = SPEEDS.get(dpg.get_value("w_speed"), 1.0)

    if G["mode"] == "live" and G["playing"] and G["ep"] is not None:
        G["acc"] += spd * dt_wall
        n = int(G["acc"] / runner.DT)
        G["acc"] -= n * runner.DT
        for _ in range(min(n, 40)):
            G["ep"].step()
            if G["ep"].done:
                finish_live()
                break
    elif G["mode"] == "replay" and G["playing"] and G["run"]:
        n_fr = len(G["run"]["traj"]["pos"])
        G["fidx"] += spd * dt_wall / G["run"]["dt"]
        if G["fidx"] >= n_fr - 1:
            G["fidx"] = float(n_fr - 1)
            G["playing"] = False
            dpg.set_item_label("w_play", "Replay")
        dpg.set_value("w_scrub", int(G["fidx"]))
    render()


def render():
    if G["mode"] == "live" and G["ep"] is not None:
        ep = G["ep"]
        d = float(np.linalg.norm(ep.target - ep.state["pos"]))
        draw_iso(ep.state["pos"], ep.state["quat"], ep.target, ep.target_radius,
                 trail=ep.traj["pos"])
        draw_fpv(ep.traj["bbox"][-1],
                 [f"LIVE   t={ep.t:5.2f}s   dist={d:4.1f}m",
                  f"{ep.cfg['name']}  |  {ep.model_name}",
                  f"reward {ep.total_reward:8.1f}"])
    elif G["mode"] == "replay" and G["run"]:
        run, tr = G["run"], G["run"]["traj"]
        i = min(int(G["fidx"]), len(tr["pos"]) - 1)
        pos, tgt = np.array(tr["pos"][i]), np.array(run["target"]["pos"])
        at_end = i >= len(tr["pos"]) - 1
        if run.get("hit"):
            verdict = f"HIT in {run['time_to_hit']:.2f}s"
        else:
            verdict = f"{run.get('outcome', 'miss').upper()}  closest {run.get('closest_approach', 0):.2f}m"
        draw_iso(pos, tr["quat"][i], tgt, run["target"]["radius"],
                 trail=tr["pos"][:i + 1], flash=at_end and run.get("hit", False))
        draw_fpv(tr["bbox"][i],
                 [f"REPLAY   t={tr['t'][i]:5.2f}s   dist={float(np.linalg.norm(tgt - pos)):4.1f}m",
                  f"{run['drone'].get('name', '?')}  |  {run.get('model', '?')}",
                  f"{verdict}   reward {run.get('total_reward', 0):.0f}"])
    else:
        draw_iso()
        draw_fpv(None, ["Configure a drone, pick a trained model, START LIVE RUN.",
                        "Or select a saved run below and PLAY SELECTED."])


# --- UI construction --------------------------------------------------------------
def build():
    presets = list(G["presets"]) or ["(none found)"]
    default_preset = "freestyle_5inch" if "freestyle_5inch" in G["presets"] else presets[0]

    with dpg.window(tag="main"):
        with dpg.group(horizontal=True):
            with dpg.child_window(width=352):
                dpg.add_text("DRONE", color=(255, 200, 90))
                dpg.add_radio_button(("Preset", "Custom"), tag="cfg_mode",
                                     horizontal=True, default_value="Preset",
                                     callback=on_mode_change)
                dpg.add_combo(presets, tag="w_preset", label="preset", width=210,
                              default_value=default_preset,
                              callback=lambda s, a: apply_preset(a))
                dpg.add_slider_int(tag="w_mass", label="mass g", width=210,
                                   min_value=20, max_value=3000, default_value=650)
                dpg.add_slider_float(tag="w_prop", label="prop in", width=210,
                                     min_value=1.0, max_value=12.0,
                                     default_value=5.1, format="%.1f")
                dpg.add_slider_int(tag="w_kv", label="motor KV", width=210,
                                   min_value=700, max_value=26000, default_value=1850)
                dpg.add_combo(list(BATTERY), tag="w_batt", label="battery", width=210,
                              default_value="6S (22.2V)")
                dpg.add_slider_int(tag="w_cam", label="cam uptilt", width=210,
                                   min_value=0, max_value=45, default_value=30)
                dpg.add_slider_int(tag="w_frame", label="frame mm", width=210,
                                   min_value=60, max_value=500, default_value=220)
                dpg.add_separator()
                dpg.add_text("MODEL", color=(255, 200, 90))
                dpg.add_combo([], tag="w_model", label="checkpoint", width=210)
                with dpg.group(horizontal=True):
                    dpg.add_button(label="Rescan", callback=lambda: scan_models())
                    dpg.add_button(label="START LIVE RUN", callback=lambda: start_live())
                dpg.add_text("", tag="w_status", wrap=330)
                dpg.add_separator()
                dpg.add_text("REPLAYS", color=(255, 200, 90))
                dpg.add_combo(("Fastest hit", "Total reward"), tag="w_rank",
                              label="rank by", width=210, default_value="Fastest hit",
                              callback=lambda s, a: refresh_runs())
                dpg.add_listbox([], tag="w_runs", num_items=9, width=332)
                dpg.add_button(label="PLAY SELECTED", width=332,
                               callback=lambda: play_selected())
            with dpg.group():
                with dpg.group(horizontal=True):
                    with dpg.drawlist(width=ISO_W, height=ISO_H, tag="iso_dl"):
                        dpg.add_draw_layer(tag="iso_layer")
                    with dpg.drawlist(width=FPV_W, height=FPV_H, tag="fpv_dl"):
                        dpg.add_draw_layer(tag="fpv_layer")
                with dpg.group(horizontal=True):
                    dpg.add_button(label="Play", tag="w_play", width=72, callback=on_play)
                    dpg.add_slider_int(tag="w_scrub", width=560, min_value=0,
                                       max_value=1, callback=on_scrub)
                    dpg.add_combo(list(SPEEDS), tag="w_speed", width=72,
                                  default_value="1x")


def _inject_smoke_replay():
    """--smoke only: synthetic trajectory to exercise the renderer. Not a
    flight policy — pure test scaffolding, never saved to runs/."""
    n = 150
    run = {"drone": {"name": "smoke-test"}, "model": "none", "dt": 0.02,
           "outcome": "hit", "hit": True, "time_to_hit": 3.0,
           "closest_approach": 0.5, "total_reward": 100.0,
           "target": {"pos": [8, 2, 2], "radius": 0.5},
           "traj": {"pos": [[8 * i / n, 2 * np.sin(3 * i / n), 1.5 + np.sin(2 * i / n)] for i in range(n)],
                    "quat": [[1, 0, 0, 0]] * n,
                    "bbox": [[0.4 * np.sin(i / 20), 0.15 * np.cos(i / 17),
                              0.05 + 0.5 * i / n, 1.0] for i in range(n)],
                    "t": [round(i * 0.02, 3) for i in range(n)]}}
    load_replay(run)


def main():
    smoke = "--smoke" in sys.argv
    G["presets"] = load_presets()
    dpg.create_context()
    dpg.create_viewport(title="DroneGym", width=1340, height=880)
    build()
    dpg.setup_dearpygui()
    dpg.show_viewport()
    dpg.set_primary_window("main", True)
    fit_view([8.0, 0.0, 2.0])
    apply_preset(dpg.get_value("w_preset"))
    on_mode_change()
    scan_models()
    refresh_runs()

    if smoke:
        _inject_smoke_replay()
        for _ in range(90):
            update()
            dpg.render_dearpygui_frame()
        print("SMOKE OK")
        dpg.destroy_context()
        return

    while dpg.is_dearpygui_running():
        update()
        dpg.render_dearpygui_frame()
    dpg.destroy_context()


if __name__ == "__main__":
    main()
