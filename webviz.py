"""Pure viz + data helpers for the Streamlit web app (streamlit_app.py).

Everything here is Streamlit-free so it can be smoke-tested headlessly. The one
public entry point the UI cares about is `build_figure(run)`, which turns a run
dict (the exact shape runner.EpisodeRunner.to_dict() / runs/*.json produce) into
a single animated Plotly figure: an isometric-style 3D world view next to the
drone's-eye FPV camera, with a native play/pause + scrub slider that runs
entirely in the browser (no server round-trip per frame).

This mirrors gui.py's draw_iso / draw_fpv / draw_att, reimplemented in Plotly.
"""

import glob
import os

import numpy as np
import plotly.graph_objects as go
import yaml
from plotly.subplots import make_subplots

from dronegym import runner
from dronegym.camera import _quat_to_rot

ROOT = os.path.dirname(os.path.abspath(__file__))

# Where committed demo assets live, plus the local RL artifact dirs. The web
# host only has the committed ones; a local dev machine also has runs/.
MODEL_GLOBS = ("assets/model/*.zip", "models/*.zip", "checkpoints/*.zip",
               "runs/*/best_model.zip", "runs/*/ckpt/*.zip")
RUN_DIRS = ("assets/replays", "runs")

MAX_FRAMES = 140            # decimate longer trajectories for smooth playback
BATTERY = {"1S (3.7V)": 3.7, "2S (7.4V)": 7.4, "3S (11.1V)": 11.1,
           "4S (14.8V)": 14.8, "6S (22.2V)": 22.2}


# --- data helpers ------------------------------------------------------------
def load_presets(root=ROOT):
    out = {}
    for p in sorted(glob.glob(os.path.join(root, "configs", "*.yaml"))):
        try:
            with open(p) as f:
                cfg = yaml.safe_load(f)
        except (yaml.YAMLError, OSError):
            continue
        if isinstance(cfg, dict) and "mass_g" in cfg:
            out[cfg.get("name", os.path.basename(p)[:-5])] = cfg
    return out


def scan_models(root=ROOT):
    """label -> absolute path, for every checkpoint we can find."""
    out = {}
    for pat in MODEL_GLOBS:
        for p in sorted(glob.glob(os.path.join(root, pat))):
            out[os.path.relpath(p, root).replace("\\", "/")] = p
    return out


def load_runs(root=ROOT):
    """Every recorded run dict, de-duplicated by filename, newest dir wins."""
    seen, runs = set(), []
    for d in RUN_DIRS:
        for r in runner.load_runs(os.path.join(root, d)):
            key = r.get("_file")
            if key in seen:
                continue
            seen.add(key)
            runs.append(r)
    return runs


def run_label(r, i):
    """Human-readable list label, matching gui.refresh_runs()."""
    spd = r.get("target_speed") or 0
    tag = f" @{spd:.0f}m/s" if spd else ""
    if r.get("hit"):
        head = f"HIT {r['time_to_hit']:.2f}s{tag}"
    else:
        head = f"{r.get('outcome', 'miss').upper()} {r.get('closest_approach', 0):.1f}m{tag}"
    return (f"{i:>2}. {head}  R{r.get('total_reward', 0):>5.0f}  "
            f"{r['drone'].get('name', '?')} [{r.get('model', '?')}]")


def run_target(run):
    t = run["target"]
    pos0 = np.array(t.get("pos0", t.get("pos")), dtype=float)
    vel = np.array(t.get("vel", [0, 0, 0]), dtype=float)
    return pos0, vel, float(t.get("radius", 0.5))


def euler_deg(q):
    """Quaternion -> (roll, pitch, yaw) in degrees (aviation signs)."""
    R = _quat_to_rot(np.asarray(q, dtype=float))
    roll = np.degrees(np.arctan2(R[2, 1], R[2, 2]))
    pitch = np.degrees(np.arcsin(np.clip(R[2, 0], -1.0, 1.0)))
    yaw = np.degrees(np.arctan2(R[1, 0], R[0, 0]))
    return roll, pitch, yaw


# --- geometry ----------------------------------------------------------------
def _arms(pos, quat, arm=0.5):
    """4 rotor arms as one polyline (None-separated) in world coords."""
    R = _quat_to_rot(np.asarray(quat, dtype=float))
    xs, ys, zs = [], [], []
    for sx, sy in ((1, 1), (-1, -1), (1, -1), (-1, 1)):
        tip = pos + R @ np.array([arm * sx, arm * sy, 0.0])
        xs += [pos[0], tip[0], None]
        ys += [pos[1], tip[1], None]
        zs += [pos[2], tip[2], None]
    return xs, ys, zs


def _nose(pos, quat, arm=0.9):
    R = _quat_to_rot(np.asarray(quat, dtype=float))
    tip = pos + R @ np.array([arm, 0.0, 0.0])
    return [pos[0], tip[0]], [pos[1], tip[1]], [pos[2], tip[2]]


def _bbox_rect(bbox):
    """FPV bounding box as a closed rectangle in [-1,1] image coords, or None."""
    if bbox is None or bbox[3] < 0.5:
        return None
    cx, cy, s = float(bbox[0]), float(bbox[1]), max(0.04, float(bbox[2]))
    xs = [cx - s, cx + s, cx + s, cx - s, cx - s]
    ys = [cy - s, cy - s, cy + s, cy + s, cy - s]
    return xs, ys


def _ground_grid(x0, x1, y0, y1, step=2.0):
    xs, ys, zs = [], [], []
    for gx in np.arange(np.floor(x0), np.ceil(x1) + 0.1, step):
        xs += [gx, gx, None]; ys += [y0, y1, None]; zs += [0, 0, None]
    for gy in np.arange(np.floor(y0), np.ceil(y1) + 0.1, step):
        xs += [x0, x1, None]; ys += [gy, gy, None]; zs += [0, 0, None]
    return xs, ys, zs


# --- the figure --------------------------------------------------------------
CYAN, ORANGE, RED, YEL, GREY = ("#00c8ff", "#ff8c32", "#f05a5a",
                                "#ffdc5a", "#2e323e")


def build_figure(run, height=560):
    """Animated Plotly figure for one run. World 3D + FPV, one shared slider."""
    tr = run["traj"]
    pos = np.asarray(tr["pos"], dtype=float)
    quat = np.asarray(tr["quat"], dtype=float)
    bbox = tr.get("bbox")
    tt = np.asarray(tr["t"], dtype=float)
    n = len(pos)
    t0, tvel, trad = run_target(run)
    tgt = t0[None, :] + tvel[None, :] * tt[:, None]         # (n,3) target path
    moving = float(np.linalg.norm(tvel)) > 1e-6

    idx = list(range(0, n, max(1, n // MAX_FRAMES))) or [0]
    if idx[-1] != n - 1:
        idx.append(n - 1)

    # world bounds -> ground grid + equal-aspect scene
    allp = np.vstack([pos, tgt, [[0, 0, 0]]])
    x0, y0, z0 = allp.min(axis=0)
    x1, y1, z1 = allp.max(axis=0)
    gx, gy, gz = _ground_grid(x0, x1, y0, y1)

    fig = make_subplots(
        rows=1, cols=2, column_widths=[0.62, 0.38],
        specs=[[{"type": "scene"}, {"type": "xy"}]],
        subplot_titles=("WORLD", "FPV CAMERA"), horizontal_spacing=0.04)

    def scene(**kw):
        return dict(mode="lines", showlegend=False, **kw)

    # 0 ground grid (static)
    fig.add_trace(go.Scatter3d(x=gx, y=gy, z=gz, line=dict(color=GREY, width=1),
                               **scene()), row=1, col=1)
    # 1 target flight line (static; empty if static scenario)
    if moving:
        end = t0 + tvel * tt[-1]
        fig.add_trace(go.Scatter3d(x=[t0[0], end[0]], y=[t0[1], end[1]],
                                   z=[t0[2], end[2]],
                                   line=dict(color=ORANGE, width=2, dash="dot"),
                                   opacity=0.5, **scene()), row=1, col=1)
    else:
        fig.add_trace(go.Scatter3d(x=[], y=[], z=[], **scene()), row=1, col=1)

    i0 = 0
    ax, ay, az = _arms(pos[i0], quat[i0])
    nx, ny, nz = _nose(pos[i0], quat[i0])
    # 2 trail  3 arms  4 nose  5 target
    fig.add_trace(go.Scatter3d(x=[pos[i0, 0]], y=[pos[i0, 1]], z=[pos[i0, 2]],
                               line=dict(color=CYAN, width=4), **scene()), row=1, col=1)
    fig.add_trace(go.Scatter3d(x=ax, y=ay, z=az, line=dict(color=RED, width=6),
                               **scene()), row=1, col=1)
    fig.add_trace(go.Scatter3d(x=nx, y=ny, z=nz, line=dict(color=YEL, width=4),
                               **scene()), row=1, col=1)
    fig.add_trace(go.Scatter3d(
        x=[tgt[i0, 0]], y=[tgt[i0, 1]], z=[tgt[i0, 2]], mode="markers",
        marker=dict(size=8, color=ORANGE, symbol="circle"),
        showlegend=False), row=1, col=1)

    # 6 FPV frame + crosshair (static)  7 FPV bbox (dynamic)
    fr = [-1, 1, 1, -1, -1]
    fc = [-1, -1, 1, 1, -1]
    frame_xs = fr + [None, -0.12, 0.12, None, 0, 0]
    frame_ys = fc + [None, 0, 0, None, -0.12, 0.12]
    fig.add_trace(go.Scatter(x=frame_xs, y=frame_ys, mode="lines",
                             line=dict(color="#5a6070", width=1),
                             showlegend=False), row=1, col=2)
    rect = _bbox_rect(bbox[i0] if bbox else None)
    fig.add_trace(go.Scatter(x=rect[0] if rect else [], y=rect[1] if rect else [],
                             mode="lines", line=dict(color="#50ff78", width=2),
                             showlegend=False), row=1, col=2)

    # --- animation frames ---
    frames = []
    for i in idx:
        ax, ay, az = _arms(pos[i], quat[i])
        nx, ny, nz = _nose(pos[i], quat[i])
        rect = _bbox_rect(bbox[i] if bbox else None)
        d = float(np.linalg.norm(tgt[i] - pos[i]))
        roll, pitch, yaw = euler_deg(quat[i])
        lost = bbox is not None and bbox[i][3] < 0.5
        hud = (f"t={tt[i]:4.2f}s   dist={d:4.1f}m   "
               f"roll {roll:+.0f}  pitch {pitch:+.0f}  yaw {yaw:+.0f}"
               + ("   <span style='color:#f05a5a'>TARGET LOST</span>" if lost else ""))
        frames.append(go.Frame(name=str(i), traces=[2, 3, 4, 5, 7], data=[
            go.Scatter3d(x=pos[:i + 1, 0], y=pos[:i + 1, 1], z=pos[:i + 1, 2]),
            go.Scatter3d(x=ax, y=ay, z=az),
            go.Scatter3d(x=nx, y=ny, z=nz),
            go.Scatter3d(x=[tgt[i, 0]], y=[tgt[i, 1]], z=[tgt[i, 2]]),
            go.Scatter(x=rect[0] if rect else [], y=rect[1] if rect else []),
        ], layout=go.Layout(title=dict(text=hud))))
    fig.frames = frames

    step_ms = max(20, int(run.get("dt", 0.02) * (idx[1] - idx[0] if len(idx) > 1 else 1) * 1000))
    slider = dict(active=0, y=0, x=0.02, len=0.6, pad=dict(t=0),
                  currentvalue=dict(visible=False),
                  steps=[dict(method="animate", label="",
                              args=[[str(i)], dict(mode="immediate",
                                    frame=dict(duration=0, redraw=True),
                                    transition=dict(duration=0))]) for i in idx])
    play = dict(type="buttons", showactive=False, y=0, x=0.0, xanchor="right",
                pad=dict(t=0, r=8), buttons=[
        dict(label="▶", method="animate",
             args=[None, dict(frame=dict(duration=step_ms, redraw=True),
                              fromcurrent=True, transition=dict(duration=0))]),
        dict(label="⏸", method="animate",
             args=[[None], dict(mode="immediate",
                                frame=dict(duration=0, redraw=True))])])

    fig.update_layout(
        height=height, margin=dict(l=0, r=0, t=34, b=28),
        paper_bgcolor="#0d1016", font=dict(color="#c8cdd7", size=12),
        title=dict(text="", x=0.02, xanchor="left", y=0.99,
                   font=dict(size=13, color="#c8cdd7")),
        updatemenus=[play], sliders=[slider],
        scene=dict(aspectmode="data", bgcolor="#0d1016",
                   xaxis=dict(title="x", color="#5a6070", backgroundcolor="#0d1016"),
                   yaxis=dict(title="y", color="#5a6070", backgroundcolor="#0d1016"),
                   zaxis=dict(title="z", color="#5a6070", backgroundcolor="#0d1016"),
                   camera=dict(eye=dict(x=1.6, y=1.6, z=1.1))))
    fig.update_xaxes(range=[-1.12, 1.12], visible=False, row=1, col=2,
                     scaleanchor="y2", scaleratio=1)
    fig.update_yaxes(range=[-1.12, 1.12], visible=False, row=1, col=2)
    for a in fig.layout.annotations:              # subplot titles
        a.font.color = "#ff8c32"
        a.font.size = 13
    return fig
