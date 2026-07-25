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
import re
import shutil
import subprocess
import sys
import time

import numpy as np
import yaml
import dearpygui.dearpygui as dpg

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from dronegym import runner
from dronegym.camera import _quat_to_rot, get_bbox

BATTERY = {"1S (3.7V)": 3.7, "2S (7.4V)": 7.4, "3S (11.1V)": 11.1,
           "4S (14.8V)": 14.8, "6S (22.2V)": 22.2}
SPEEDS = {"0.25x": 0.25, "0.5x": 0.5, "1x": 1.0, "2x": 2.0, "4x": 4.0}
RUNS_DIR = os.path.join(ROOT, "runs")
MODEL_DIRS = ("checkpoints", "models")

ISO_W, ISO_H = 580, 560
FPV_W, FPV_H = 400, 560
FPV_SQ = 380
ATT_W, ATT_H = 270, 560
ATT_SQ = 250
from dronegym.config import DroneConfig
MAX_RATE_DEG = np.degrees(DroneConfig.__dataclass_fields__["max_body_rate"].default)

G = {"mode": "idle", "ep": None, "run": None, "fidx": 0.0, "playing": False,
     "acc": 0.0, "view": None, "grid": (-2, 10, -4, 4), "run_map": {},
     "presets": {}, "models": {}, "dragging": None}


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


def fit_view(target, traj=None, extra=None):
    t = np.asarray(target, dtype=float)
    pts = [(-2, -3, 0), (-2, 3, 0), (max(4, t[0] + 2), -3, 0),
           (0, 0, 4), (t[0], t[1] - 2, 0), (t[0], t[1] + 2, 0),
           (t[0], t[1], t[2] + 1.5)]
    if traj is not None and len(traj) > 1:
        a = np.asarray(traj, dtype=float)
        pts += [a.min(axis=0), a.max(axis=0)]
    if extra:
        pts += [np.asarray(p, dtype=float) for p in extra]
    P = np.array(pts, dtype=float)
    x0, y0 = P[:, 0].min(), P[:, 1].min()
    x1, y1 = P[:, 0].max(), P[:, 1].max()
    G["grid"] = (x0, x1, y0, y1)
    corners = [(x0, y0, 0), (x0, y1, 0), (x1, y0, 0), (x1, y1, 0)]
    G["view"] = IsoView(np.vstack([P, corners]), ISO_W, ISO_H)


# --- drawing ------------------------------------------------------------------
def draw_iso(pos=None, quat=None, target=None, t_radius=0.5, trail=(), flash=False,
             tpath=None, tvel=None):
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

    if tpath is not None:                  # intercept: the target's flight line
        dpg.draw_line(V.px(tpath[0]), V.px(tpath[1]),
                      color=(255, 140, 50, 70), thickness=1, parent=L)

    if target is not None:
        sh, tp = V.px((target[0], target[1], 0)), V.px(target)
        dpg.draw_circle(sh, max(3, V.k * t_radius * 0.6), fill=(0, 0, 0, 90),
                        color=(0, 0, 0, 0), parent=L)
        dpg.draw_line(sh, tp, color=(120, 120, 135, 110), parent=L)
        dpg.draw_circle(tp, max(4, V.k * t_radius), color=(255, 190, 80),
                        fill=(255, 140, 50, 210), thickness=2, parent=L)
        if tvel is not None and np.linalg.norm(tvel) > 1e-6:
            head = np.asarray(target) + np.asarray(tvel) / np.linalg.norm(tvel) * 1.4
            dpg.draw_arrow(V.px(head), tp, color=(255, 140, 50, 200),
                           thickness=2, size=7, parent=L)
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


def _draw_bar(L, x, y, w, label, frac, text, color, symmetric):
    """One stick channel: label, bar, value text. frac in [-1,1] (or [0,1])."""
    bx = x + 34
    dpg.draw_text((x, y - 7), label, size=13, color=(150, 155, 170), parent=L)
    dpg.draw_rectangle((bx, y - 6), (bx + w, y + 6), color=(60, 65, 80), parent=L)
    if symmetric:
        cx = bx + w / 2
        dpg.draw_line((cx, y - 6), (cx, y + 6), color=(90, 95, 110), parent=L)
        dpg.draw_rectangle((min(cx, cx + frac * w / 2), y - 4),
                           (max(cx, cx + frac * w / 2), y + 4),
                           fill=color, color=(0, 0, 0, 0), parent=L)
    else:
        dpg.draw_rectangle((bx, y - 4), (bx + max(1, frac * w), y + 4),
                           fill=color, color=(0, 0, 0, 0), parent=L)
    dpg.draw_text((bx + w + 7, y - 7), text, size=13, color=(200, 205, 215), parent=L)


def draw_sticks(L, x0, ty, act):
    """Throttle / roll / pitch / yaw commands for the current frame."""
    if act is None:
        dpg.draw_text((x0, ty), "(no stick data in this run)", size=13,
                      color=(120, 125, 140), parent=L)
        return
    bar_w = FPV_SQ - 34 - 74
    thr = (float(act[0]) + 1.0) / 2.0
    _draw_bar(L, x0, ty + 6, bar_w, "THR", thr, f"{thr * 100:3.0f}%",
              (255, 190, 80, 230), symmetric=False)
    for i, (name, sgn) in enumerate((("ROL", 1), ("PIT", 1), ("YAW", 1)), start=1):
        v = float(act[i]) * sgn
        _draw_bar(L, x0, ty + 6 + 17 * i, bar_w, name, v,
                  f"{v * MAX_RATE_DEG:+5.0f} deg/s", (0, 200, 255, 230),
                  symmetric=True)


def draw_fpv(bbox=None, hud=(), act=None):
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
    if bbox is not None:                              # idle screen: no sticks
        draw_sticks(L, x0, ty + 4, act)


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


def _euler_deg(q):
    """Quaternion -> (roll, pitch, yaw) degrees. Aviation display signs:
    roll + = right bank, pitch + = nose up, yaw + = nose left (CCW from +x)."""
    R = _quat_to_rot(np.asarray(q, dtype=float))
    roll = np.degrees(np.arctan2(R[2, 1], R[2, 2]))
    pitch = np.degrees(np.arcsin(np.clip(R[2, 0], -1.0, 1.0)))
    yaw = np.degrees(np.arctan2(R[1, 0], R[0, 0]))
    return roll, pitch, yaw


def draw_att(quat=None):
    """Centered 3D drone at fixed scale: pure attitude, no translation."""
    L = "att_layer"
    dpg.delete_item(L, children_only=True)
    x0, y0, sq = 10, 34, ATT_SQ
    cx, cy = x0 + sq / 2, y0 + sq / 2
    dpg.draw_text((x0, 8), "ATTITUDE", size=15, color=(255, 200, 90), parent=L)
    dpg.draw_rectangle((x0, y0), (x0 + sq, y0 + sq), fill=(13, 16, 22),
                       color=(70, 75, 90), parent=L)
    k = 62.0

    def px(v):                             # same iso camera as the world view
        u = (v[0] - v[1]) * IsoView.C
        w = (v[0] + v[1]) * IsoView.S - v[2]
        return (cx + k * u, cy + k * w)

    # world-fixed reference: ground ring + world +x / +y axes
    ring = [px((1.35 * np.cos(a), 1.35 * np.sin(a), 0))
            for a in np.linspace(0, 2 * np.pi, 41)]
    dpg.draw_polyline(ring, color=(50, 55, 70), parent=L)
    dpg.draw_line(px((0, 0, 0)), px((1.35, 0, 0)), color=(80, 86, 105), parent=L)
    dpg.draw_text(px((1.62, 0, 0)), "+X", size=12, color=(80, 86, 105), parent=L)
    dpg.draw_line(px((0, 0, 0)), px((0, 1.35, 0)), color=(60, 66, 84), parent=L)

    if quat is None:
        quat = (1.0, 0.0, 0.0, 0.0)
    R = _quat_to_rot(np.asarray(quat, dtype=float))
    for sx, sy in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
        tip = R @ np.array([0.72 * sx, 0.72 * sy, 0.0])
        col = (240, 90, 90) if sx > 0 else (205, 205, 220)
        tp = px(tip)
        dpg.draw_line(px((0, 0, 0)), tp, color=col, thickness=4, parent=L)
        dpg.draw_circle(tp, 5, fill=col, color=(0, 0, 0, 0), parent=L)
    up = R @ np.array([0.0, 0.0, 0.5])     # body-z strut makes tilt readable
    dpg.draw_line(px((0, 0, 0)), px(up), color=(120, 200, 140), thickness=2, parent=L)
    nose = R @ np.array([1.15, 0.0, 0.0])
    dpg.draw_arrow(px(nose), px((0, 0, 0)), color=(255, 220, 90),
                   thickness=2, size=8, parent=L)

    roll, pitch, yaw = _euler_deg(quat)
    ty = y0 + sq + 12
    for name, val in (("ROLL", roll), ("PITCH", pitch), ("YAW", yaw)):
        dpg.draw_text((x0, ty), f"{name:<6s}{val:+7.1f} deg", size=15,
                      color=(200, 205, 215), parent=L)
        ty += 22


def run_target(run):
    """Target info from a run dict; tolerates the pre-intercept format."""
    t = run["target"]
    pos0 = np.array(t.get("pos0", t.get("pos")), dtype=float)
    vel = np.array(t.get("vel", [0, 0, 0]), dtype=float)
    return pos0, vel, t.get("radius", 0.5)


def on_scen_change(sender=None, app_data=None):
    intercept = dpg.get_value("w_scen") == "Intercept"
    dpg.configure_item("w_tspeed", enabled=intercept)
    dpg.configure_item("w_thead", enabled=intercept)


def on_custom_change(sender=None, app_data=None):
    on = dpg.get_value("w_custom")
    dpg.configure_item("grp_custom", show=on)
    if on:                                 # enter the scenario editor
        G.update(mode="idle", ep=None, run=None, playing=False, dragging=None)
        dpg.set_item_label("w_play", "Play")


def read_setup():
    """Scenario-editor widget values -> EpisodeRunner's `custom` dict."""
    return {"drone_pos": list(dpg.get_value("w_dpos"))[:3],
            "drone_rpy_deg": list(dpg.get_value("w_datt"))[:3],
            "target_pos": list(dpg.get_value("w_tpos"))[:3],
            "target_heading_deg": float(dpg.get_value("w_thead")),
            "target_speed": float(dpg.get_value("w_tspeed"))}


# --- drag-and-drop placement in the iso view ------------------------------------
def _setup_active():
    return (dpg.get_value("w_custom") and G["mode"] == "idle"
            and G["view"] is not None)


def _iso_mouse():
    mx, my = dpg.get_mouse_pos(local=False)
    rx, ry = dpg.get_item_rect_min("iso_dl")
    return mx - rx, my - ry


def on_mouse_down(sender=None, app_data=None):
    if not _setup_active():
        return
    sx, sy = _iso_mouse()
    if not (0 <= sx <= ISO_W and 0 <= sy <= ISO_H):
        return
    V, s = G["view"], read_setup()
    for name, p in (("target", s["target_pos"]), ("drone", s["drone_pos"])):
        px, py = V.px(p)
        if (px - sx) ** 2 + (py - sy) ** 2 <= 18 ** 2:
            G["dragging"] = name
            return


def on_mouse_move(sender=None, app_data=None):
    tag = {"drone": "w_dpos", "target": "w_tpos"}.get(G["dragging"])
    if not tag or not _setup_active():
        return
    sx, sy = _iso_mouse()
    V = G["view"]
    vals = list(dpg.get_value(tag))
    # invert the iso projection on the horizontal plane at the object's own z:
    # u = (x - y) C ; v = (x + y) S - z
    a = (sx - V.ox) / V.k / IsoView.C
    b = ((sy - V.oy) / V.k + vals[2]) / IsoView.S
    vals[0] = float(np.clip((a + b) / 2.0, -30.0, 30.0))
    vals[1] = float(np.clip((b - a) / 2.0, -30.0, 30.0))
    dpg.set_value(tag, vals)


def on_mouse_release(sender=None, app_data=None):
    G["dragging"] = None


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


# --- model import / export -------------------------------------------------------
def _file_dialog(tag, callback, default_filename=""):
    """(Re)create a modal .zip file dialog and show it."""
    if dpg.does_item_exist(tag):
        dpg.delete_item(tag)
    with dpg.file_dialog(directory_selector=False, show=True, modal=True,
                         callback=callback, tag=tag, width=720, height=420,
                         default_path=os.path.expanduser("~"),
                         default_filename=default_filename):
        dpg.add_file_extension(".zip", color=(140, 220, 140, 255))
        dpg.add_file_extension(".*")


def open_import():
    _file_dialog("fd_import", on_import_file)


def open_export():
    sel = dpg.get_value("w_model")
    src = G["models"].get(sel)
    if not src:
        status("no model selected to export", ok=False)
        return
    G["export_src"] = src
    _file_dialog("fd_export", on_export_file,
                 default_filename=os.path.basename(src))


def on_import_file(sender, app_data):
    path = (app_data or {}).get("file_path_name", "")
    if not path or not os.path.isfile(path):
        status("import cancelled: file not found", ok=False)
        return
    base = os.path.splitext(os.path.basename(path))[0]
    ckpt_dir = os.path.join(ROOT, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    dst = os.path.join(ckpt_dir, base + ".zip")
    if os.path.abspath(path) != os.path.abspath(dst):
        if os.path.exists(dst):                    # never clobber an existing model
            dst = os.path.join(ckpt_dir, f"{base}_{time.strftime('%H%M%S')}.zip")
        shutil.copyfile(path, dst)
    scan_models()
    dpg.set_value("w_model", os.path.relpath(dst, ROOT))
    try:
        runner.load_policy(dst)
        status(f"imported {os.path.basename(dst)} - verified, ready to fly")
    except RuntimeError as e:                      # kept on disk, flagged clearly
        status(f"imported {os.path.basename(dst)}, but it failed verification: {e}",
               ok=False)


def on_export_file(sender, app_data):
    path = (app_data or {}).get("file_path_name", "")
    src = G.get("export_src")
    if not path or not src:
        return
    if not path.lower().endswith(".zip"):
        path += ".zip"
    try:
        shutil.copyfile(src, path)
        status(f"exported {os.path.basename(src)} -> {path}")
    except OSError as e:
        status(f"export failed: {e}", ok=False)


def start_live():
    sel = dpg.get_value("w_model")
    try:
        policy = runner.load_policy(G["models"].get(sel))
    except RuntimeError as e:
        status(str(e), ok=False)
        return
    cfg = current_cfg()
    intercept = dpg.get_value("w_scen") == "Intercept"
    ep = runner.EpisodeRunner(
        cfg, policy, model_name=os.path.basename(sel),
        scenario="intercept" if intercept else "static",
        target_speed=float(dpg.get_value("w_tspeed")) if intercept else None,
        custom=read_setup() if dpg.get_value("w_custom") else None)
    G.update(mode="live", ep=ep, run=None, playing=True, acc=0.0)
    fit_view(ep.target0,
             extra=[np.array(ep.state["pos"]),
                    ep.target0 + ep.target_vel * min(runner.MAX_T, 8.0)])
    dpg.set_item_label("w_play", "Pause")
    dpg.configure_item("w_scrub", enabled=False)
    spd = float(np.linalg.norm(ep.target_vel))
    status(f"flying {cfg['name']} on {sel}"
           + (f"  |  intercept @{spd:.1f} m/s ..." if intercept else " ..."))


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


# --- in-GUI training ------------------------------------------------------------
TRAIN_FPS_EST = 930          # measured on this machine, static5min run


def tstatus(msg, ok=True):
    dpg.set_value("w_tstatus", msg)
    dpg.configure_item("w_tstatus", color=(140, 220, 140) if ok else (255, 120, 120))


def update_est(sender=None, app_data=None):
    steps = int(dpg.get_value("w_tsteps"))
    dpg.set_value("w_test",
                  f"~{steps / TRAIN_FPS_EST / 60.0:.1f} min at ~{TRAIN_FPS_EST} steps/s")


def _tail_progress(log_path):
    """Latest '[throughput] N steps' figure from the training log, or None."""
    try:
        with open(log_path, "r", errors="ignore") as f:
            f.seek(0, 2)
            f.seek(max(0, f.tell() - 6000))
            txt = f.read()
    except OSError:
        return None
    steps = None
    for line in txt.splitlines():
        if "[throughput]" in line:
            m = re.search(r"([\d,]+) steps", line)
            if m:
                steps = int(m.group(1).replace(",", ""))
    return steps


def start_training():
    if G.get("train_proc") is not None:
        return
    cfg = current_cfg()
    steps = int(dpg.get_value("w_tsteps"))
    stamp = time.strftime("%H%M%S")
    run_name = f"gui_{cfg['name']}_{stamp}"

    if dpg.get_value("cfg_mode") == "Preset":
        cfg_arg = os.path.join(ROOT, "configs", f"{cfg['name']}.yaml")
    else:
        # custom slider values: write a config train.py can load. NOT into
        # configs/ — presets.all_presets() asserts exactly 5 files there.
        gui_cfg_dir = os.path.join(ROOT, "runs", "gui_configs")
        os.makedirs(gui_cfg_dir, exist_ok=True)
        cfg_arg = os.path.join(gui_cfg_dir, f"{run_name}.yaml")
        with open(cfg_arg, "w") as f:
            yaml.safe_dump(cfg, f)

    os.makedirs(os.path.join(ROOT, "runs"), exist_ok=True)
    log_path = os.path.join(ROOT, "runs", f"{run_name}.log")
    logf = open(log_path, "w")
    cmd = [sys.executable, os.path.join(ROOT, "train.py"), "--config", cfg_arg,
           "--steps", str(steps), "--run-name", run_name, "--difficulty", "0"]
    flags = 0x08000000 if os.name == "nt" else 0        # CREATE_NO_WINDOW
    G["train_proc"] = subprocess.Popen(cmd, cwd=ROOT, stdout=logf,
                                       stderr=subprocess.STDOUT,
                                       creationflags=flags)
    G["train_logf"] = logf
    G["train_meta"] = {"run": run_name, "steps": steps, "t0": time.time(),
                       "log": log_path, "cfg_name": cfg["name"], "stamp": stamp}
    dpg.configure_item("w_train", enabled=False)
    tstatus(f"training {run_name} ({steps:,} steps @ level 0)...")


def poll_training():
    proc = G.get("train_proc")
    if proc is None:
        return
    now = time.time()
    if now - G.get("train_poll_t", 0.0) < 1.0:
        return
    G["train_poll_t"] = now
    meta = G["train_meta"]

    if proc.poll() is None:
        done = _tail_progress(meta["log"]) or 0
        pct = min(99.0, 100.0 * done / meta["steps"])
        tstatus(f"training... {pct:3.0f}%   {(now - meta['t0']) / 60.0:.1f} min elapsed")
        return

    rc = proc.returncode
    G["train_proc"] = None
    G["train_logf"].close()
    dpg.configure_item("w_train", enabled=True)
    if rc != 0:
        tstatus(f"training failed (exit {rc}) - see runs/{meta['run']}.log", ok=False)
        return
    run_dir = os.path.join(ROOT, "runs", meta["run"])
    src = os.path.join(run_dir, "best_model.zip")       # absent on short runs:
    if not os.path.isfile(src):                         # eval fires every 50k steps
        src = os.path.join(run_dir, "final_model.zip")
    dst = os.path.join(ROOT, "checkpoints",
                       f"{meta['cfg_name']}_{meta['stamp']}_{meta['steps'] // 1000}k.zip")
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copyfile(src, dst)
    scan_models()
    dpg.set_value("w_model", os.path.relpath(dst, ROOT))
    tstatus(f"done: {os.path.basename(dst)} selected and ready to fly")


# --- replay browser -------------------------------------------------------------
def refresh_runs():
    ranked = runner.rank_runs(runner.load_runs(RUNS_DIR), dpg.get_value("w_rank"))
    G["run_map"] = {}
    items = []
    for i, r in enumerate(ranked, 1):
        spd = r.get("target_speed") or 0
        tag = f" @{spd:.0f}m/s" if spd else ""
        if r.get("hit"):
            lab = f"{i:>2}. HIT {r['time_to_hit']:5.2f}s{tag}"
        else:
            lab = f"{i:>2}. {r.get('outcome', 'miss').upper()} {r.get('closest_approach', 0):.1f}m{tag}"
        lab += f"  R{r.get('total_reward', 0):>5.0f}  {r['drone'].get('name', '?')} [{r.get('model', '?')}]"
        while lab in G["run_map"]:
            lab += " "
        G["run_map"][lab] = r
        items.append(lab)
    dpg.configure_item("w_runs", items=items or ["(no runs recorded yet)"])


def load_replay(run, autoplay=True):
    n = len(run["traj"]["pos"])
    G.update(mode="replay", run=run, ep=None, fidx=0.0, playing=autoplay, acc=0.0)
    t0, tvel, _ = run_target(run)
    fit_view(t0, traj=run["traj"]["pos"],
             extra=[t0 + tvel * run["traj"]["t"][-1]])
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
    poll_training()
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
        tspd = float(np.linalg.norm(ep.target_vel))
        draw_iso(ep.state["pos"], ep.state["quat"], ep.target, ep.target_radius,
                 trail=ep.traj["pos"],
                 tpath=(ep.target0, ep.target0 + ep.target_vel * runner.MAX_T)
                 if tspd > 1e-6 else None,
                 tvel=ep.target_vel if tspd > 1e-6 else None)
        hud2 = f"{ep.cfg['name']}  |  {ep.model_name}"
        if tspd > 1e-6:
            hud2 += f"  |  tgt {tspd:.1f}m/s"
        draw_fpv(ep.traj["bbox"][-1],
                 [f"LIVE   t={ep.t:5.2f}s   dist={d:4.1f}m", hud2,
                  f"reward {ep.total_reward:8.1f}"],
                 act=ep.traj["act"][-1])
        draw_att(ep.state["quat"])
    elif G["mode"] == "replay" and G["run"]:
        run, tr = G["run"], G["run"]["traj"]
        i = min(int(G["fidx"]), len(tr["pos"]) - 1)
        pos = np.array(tr["pos"][i])
        t0, tvel, trad = run_target(run)
        tgt = t0 + tvel * tr["t"][i]
        tspd = float(np.linalg.norm(tvel))
        at_end = i >= len(tr["pos"]) - 1
        if run.get("hit"):
            verdict = f"HIT in {run['time_to_hit']:.2f}s"
        else:
            verdict = f"{run.get('outcome', 'miss').upper()}  closest {run.get('closest_approach', 0):.2f}m"
        if tspd > 1e-6:
            verdict += f"  |  tgt {tspd:.1f}m/s"
        draw_iso(pos, tr["quat"][i], tgt, trad,
                 trail=tr["pos"][:i + 1], flash=at_end and run.get("hit", False),
                 tpath=(t0, t0 + tvel * tr["t"][-1]) if tspd > 1e-6 else None,
                 tvel=tvel if tspd > 1e-6 else None)
        acts = tr.get("act")
        draw_fpv(tr["bbox"][i],
                 [f"REPLAY   t={tr['t'][i]:5.2f}s   dist={float(np.linalg.norm(tgt - pos)):4.1f}m",
                  f"{run['drone'].get('name', '?')}  |  {run.get('model', '?')}",
                  f"{verdict}   reward {run.get('total_reward', 0):.0f}"],
                 act=acts[i] if acts and i < len(acts) else None)
        draw_att(tr["quat"][i])
    elif dpg.get_value("w_custom"):        # scenario editor preview
        s = read_setup()
        q = runner.custom_quat(s["drone_rpy_deg"])
        tp = np.asarray(s["target_pos"], dtype=float)
        tvel = tpath = None
        if dpg.get_value("w_scen") == "Intercept":
            h = np.radians(s["target_heading_deg"])
            tvel = np.array([np.cos(h), np.sin(h), 0.0]) * s["target_speed"]
            tpath = (tp, tp + tvel * runner.MAX_T)
        if not G["dragging"]:              # freeze the camera while dragging
            fit_view(tp, extra=[s["drone_pos"]] + ([tpath[1]] if tpath else []))
        draw_iso(pos=s["drone_pos"], quat=q, target=tp,
                 t_radius=runner.TARGET_RADIUS, tpath=tpath, tvel=tvel)
        bbox = get_bbox(s["drone_pos"], q, tp, runner.TARGET_RADIUS,
                        current_cfg()["cam_angle_deg"])
        d = float(np.linalg.norm(tp - np.asarray(s["drone_pos"])))
        draw_fpv(bbox, [f"SETUP   dist={d:4.1f}m   drag drone/target in 3D view",
                        "z + attitude via the SCENARIO fields",
                        "START LIVE RUN flies this exact setup"])
        draw_att(q)
    else:
        draw_iso()
        draw_fpv(None, ["Configure a drone, pick a trained model, START LIVE RUN.",
                        "Or select a saved run below and PLAY SELECTED."])
        draw_att()


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
                dpg.add_text("SCENARIO", color=(255, 200, 90))
                dpg.add_radio_button(("Static", "Intercept"), tag="w_scen",
                                     horizontal=True, default_value="Static",
                                     callback=on_scen_change)
                dpg.add_slider_float(tag="w_tspeed", label="target m/s", width=210,
                                     min_value=2.0, max_value=10.0,
                                     default_value=5.0, format="%.1f", enabled=False)
                dpg.add_checkbox(label="custom setup (drag to place)",
                                 tag="w_custom", callback=on_custom_change)
                with dpg.group(tag="grp_custom", show=False):
                    dpg.add_drag_floatx(tag="w_dpos", label="drone x y z", size=3,
                                        width=210, speed=0.1, clamped=True,
                                        min_value=-30.0, max_value=30.0,
                                        default_value=[0.0, 0.0, 10.0, 0.0])
                    dpg.add_drag_floatx(tag="w_datt", label="roll pitch yaw",
                                        size=3, width=210, speed=1.0, clamped=True,
                                        min_value=-180.0, max_value=180.0,
                                        default_value=[0.0, 0.0, 0.0, 0.0])
                    dpg.add_drag_floatx(tag="w_tpos", label="target x y z", size=3,
                                        width=210, speed=0.1, clamped=True,
                                        min_value=-30.0, max_value=30.0,
                                        default_value=[5.0, 0.0, 10.0, 0.0])
                    dpg.add_drag_float(tag="w_thead", label="tgt heading deg",
                                       width=210, speed=1.0, clamped=True,
                                       min_value=-180.0, max_value=180.0,
                                       default_value=180.0, enabled=False)
                dpg.add_separator()
                dpg.add_text("MODEL", color=(255, 200, 90))
                dpg.add_combo([], tag="w_model", label="checkpoint", width=210)
                with dpg.group(horizontal=True):
                    dpg.add_button(label="Rescan", callback=lambda: scan_models())
                    dpg.add_button(label="START LIVE RUN", callback=lambda: start_live())
                with dpg.group(horizontal=True):
                    dpg.add_button(label="Import model...", width=163,
                                   callback=lambda: open_import())
                    dpg.add_button(label="Export model...", width=163,
                                   callback=lambda: open_export())
                dpg.add_text("", tag="w_status", wrap=330)
                dpg.add_separator()
                dpg.add_text("TRAINING", color=(255, 200, 90))
                dpg.add_slider_int(tag="w_tsteps", label="steps", width=210,
                                   min_value=50_000, max_value=1_000_000,
                                   default_value=280_000, callback=update_est)
                dpg.add_text("", tag="w_test", color=(150, 155, 170))
                dpg.add_button(label="TRAIN MODEL", tag="w_train", width=332,
                               callback=lambda: start_training())
                dpg.add_text("", tag="w_tstatus", wrap=330)
            with dpg.group():
                with dpg.group(horizontal=True):
                    with dpg.drawlist(width=ISO_W, height=ISO_H, tag="iso_dl"):
                        dpg.add_draw_layer(tag="iso_layer")
                    with dpg.drawlist(width=FPV_W, height=FPV_H, tag="fpv_dl"):
                        dpg.add_draw_layer(tag="fpv_layer")
                    with dpg.drawlist(width=ATT_W, height=ATT_H, tag="att_dl"):
                        dpg.add_draw_layer(tag="att_layer")
                with dpg.group(horizontal=True):
                    dpg.add_button(label="Play", tag="w_play", width=72, callback=on_play)
                    dpg.add_slider_int(tag="w_scrub", width=1090, min_value=0,
                                       max_value=1, callback=on_scrub)
                    dpg.add_combo(list(SPEEDS), tag="w_speed", width=72,
                                  default_value="1x")
                dpg.add_spacer(height=4)
                with dpg.group(horizontal=True):
                    with dpg.group():
                        dpg.add_text("REPLAYS", color=(255, 200, 90))
                        dpg.add_combo(("Fastest hit", "Total reward"), tag="w_rank",
                                      label="rank by", width=148,
                                      default_value="Fastest hit",
                                      callback=lambda s, a: refresh_runs())
                        dpg.add_button(label="PLAY SELECTED", width=232,
                                       callback=lambda: play_selected())
                    dpg.add_listbox([], tag="w_runs", num_items=8, width=1014)

    with dpg.handler_registry():
        dpg.add_mouse_down_handler(callback=on_mouse_down)
        dpg.add_mouse_move_handler(callback=on_mouse_move)
        dpg.add_mouse_release_handler(callback=on_mouse_release)


def _inject_smoke_replay():
    """--smoke only: synthetic trajectory to exercise the renderer. Not a
    flight policy — pure test scaffolding, never saved to runs/."""
    n = 150
    run = {"drone": {"name": "smoke-test"}, "model": "none", "dt": 0.02,
           "scenario": "intercept", "target_speed": 1.5,
           "outcome": "hit", "hit": True, "time_to_hit": 3.0,
           "closest_approach": 0.5, "total_reward": 100.0,
           "target": {"pos0": [8, 2, 2], "vel": [-1.3, 0.7, 0], "radius": 0.5},
           "traj": {"pos": [[8 * i / n, 2 * np.sin(3 * i / n), 1.5 + np.sin(2 * i / n)] for i in range(n)],
                    "quat": [[1, 0, 0, 0]] * n,
                    "bbox": [[0.4 * np.sin(i / 20), 0.15 * np.cos(i / 17),
                              0.05 + 0.5 * i / n, 1.0] for i in range(n)],
                    "act": [[0.2 * np.sin(i / 15) - 0.4, 0.5 * np.sin(i / 9),
                             0.4 * np.cos(i / 12), 0.6 * np.sin(i / 22)] for i in range(n)],
                    "t": [round(i * 0.02, 3) for i in range(n)]}}
    load_replay(run)


def main():
    smoke = "--smoke" in sys.argv
    G["presets"] = load_presets()
    dpg.create_context()
    dpg.create_viewport(title="DroneGym", width=1660, height=880)
    build()
    dpg.setup_dearpygui()
    dpg.show_viewport()
    dpg.set_primary_window("main", True)
    fit_view([8.0, 0.0, 2.0])
    apply_preset(dpg.get_value("w_preset"))
    on_mode_change()
    on_scen_change()
    scan_models()
    refresh_runs()
    update_est()

    if smoke:
        _inject_smoke_replay()                # new format, moving target
        for _ in range(60):
            update()
            dpg.render_dearpygui_frame()
        disk = runner.load_runs(RUNS_DIR)     # old-format compat, if present
        if disk:
            load_replay(disk[0])
            for _ in range(30):
                update()
                dpg.render_dearpygui_frame()
        # scenario-editor preview, static then intercept
        dpg.set_value("w_custom", True)
        on_custom_change()
        for scen in ("Static", "Intercept"):
            dpg.set_value("w_scen", scen)
            for _ in range(15):
                update()
                dpg.render_dearpygui_frame()
        # import/export round-trip through the real callbacks, then clean up
        src = G["models"].get(dpg.get_value("w_model"))
        if src:
            G["export_src"] = src
            exp = os.path.join(os.environ.get("TEMP", ROOT),
                               "dronegym_smoke_export.zip")
            on_export_file(None, {"file_path_name": exp})
            assert os.path.isfile(exp), "export failed"
            on_import_file(None, {"file_path_name": exp})
            imported = os.path.join(ROOT, "checkpoints", "dronegym_smoke_export.zip")
            assert os.path.isfile(imported), "import failed"
            os.remove(imported)
            os.remove(exp)
            scan_models()
            print("import/export round-trip OK")
        print("SMOKE OK")
        dpg.destroy_context()
        return

    while dpg.is_dearpygui_running():
        update()
        dpg.render_dearpygui_frame()
    dpg.destroy_context()


if __name__ == "__main__":
    main()
