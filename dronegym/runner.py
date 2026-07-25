"""Episode runner: steps physics + camera into the 12-dim obs contract,
records trajectories to runs/*.json, ranks them, loads policies.

Used by the GUI for live runs and replays; evaluate.py will reuse it.
Automatically prefers the real dronegym.physics / dronegym.rewards modules
when they exist; falls back to the stubs so this branch works standalone.
"""

import json
import os
import time

import numpy as np

from dronegym.camera import get_bbox, _quat_to_rot

# --- backend selection: real modules win the moment teammates push them ----
from dronegym import stub_physics

try:
    from dronegym import physics as _phys
    PHYS = _phys if all(hasattr(_phys, f) for f in ("derive_params", "make_state", "step")) else stub_physics
except ImportError:
    PHYS = stub_physics

try:
    from dronegym.rewards import compute_reward as _real_reward
except ImportError:
    _real_reward = None

RUNS_DIR = "runs"
DT = 0.02
MAX_T = 12.0
HIT_MARGIN = 0.15


class EpisodeRunner:
    """One episode, steppable frame-by-frame (for the GUI) or all at once."""

    def __init__(self, cfg, policy, target_pos=None, target_radius=0.5,
                 model_name="scripted", rng=None):
        rng = rng or np.random.default_rng()
        self.cfg = dict(cfg)
        self.policy = policy
        self.model_name = model_name
        self.params = PHYS.derive_params(cfg)

        if target_pos is None:
            bearing = rng.uniform(-0.5, 0.5)          # rad, roughly ahead
            dist = rng.uniform(7.0, 10.0)
            target_pos = [dist * np.cos(bearing), dist * np.sin(bearing),
                          rng.uniform(1.5, 3.5)]
        self.target = np.asarray(target_pos, dtype=float)
        self.target_radius = target_radius
        self.drone_radius = cfg["frame_size_mm"] / 2000.0   # frame half-span, m

        self.state = PHYS.make_state([0.0, 0.0, 1.5], yaw=0.0)
        self.t = 0.0
        self.done = False
        self.outcome = None                            # 'hit' | 'crash' | 'timeout'
        self.total_reward = 0.0
        self.closest = self._dist()
        self.traj = {"pos": [], "quat": [], "bbox": [], "t": [], "act": []}
        self._record_frame()

    # -- helpers -------------------------------------------------------------
    def _dist(self):
        return float(np.linalg.norm(self.target - self.state["pos"]))

    def bbox(self):
        return get_bbox(self.state["pos"], self.state["quat"], self.target,
                        self.target_radius, self.cfg["cam_angle_deg"])

    def obs(self):
        s = self.state
        R = _quat_to_rot(s["quat"])
        g_body = R.T @ np.array([0.0, 0.0, -1.0])
        v_body = R.T @ s["vel"]
        return np.concatenate([self.bbox(), s["rates"], g_body,
                               [v_body[0], v_body[2]]]).astype(np.float32)

    def _record_frame(self, action=None):
        self.traj["pos"].append(self.state["pos"].tolist())
        self.traj["quat"].append(self.state["quat"].tolist())
        self.traj["bbox"].append(self.bbox().tolist())
        self.traj["t"].append(round(self.t, 4))
        # stick commands that produced this frame; frame 0 = at rest
        act = [-1.0, 0.0, 0.0, 0.0] if action is None else \
            [float(a) for a in np.clip(np.asarray(action, dtype=float), -1, 1)]
        self.traj["act"].append(act)

    # -- stepping ------------------------------------------------------------
    def step(self):
        if self.done:
            return
        prev_dist = self._dist()
        action = self.policy(self.obs())
        self.state = PHYS.step(self.state, action, self.params, DT)
        self.t += DT

        dist = self._dist()
        self.closest = min(self.closest, dist)
        hit = dist <= self.target_radius + self.drone_radius + HIT_MARGIN
        crash = self.state["pos"][2] <= 0.05
        timeout = self.t >= MAX_T

        if _real_reward is not None:
            r = _real_reward(prev_dist, dist, self.bbox(), hit, crash)
        else:  # placeholder shaping until rewards.py lands
            b = self.bbox()
            centering = b[3] * max(0.0, 1.0 - abs(b[0]) - abs(b[1]))
            r = (2.5 * (prev_dist - dist) + 0.3 * centering - 0.05 * (1 - b[3])
                 - 0.01 + (100.0 if hit else 0.0) - (50.0 if crash else 0.0))
        self.total_reward += float(r)

        self._record_frame(action)
        if hit or crash or timeout:
            self.done = True
            self.outcome = "hit" if hit else ("crash" if crash else "timeout")

    def run(self):
        while not self.done:
            self.step()
        return self

    # -- recording -----------------------------------------------------------
    def to_dict(self):
        return {
            "drone": self.cfg,
            "model": self.model_name,
            "dt": DT,
            "outcome": self.outcome,
            "hit": self.outcome == "hit",
            "time_to_hit": round(self.t, 3) if self.outcome == "hit" else None,
            "closest_approach": round(self.closest, 3),
            "total_reward": round(self.total_reward, 2),
            "target": {"pos": self.target.tolist(), "radius": self.target_radius},
            "traj": self.traj,
        }

    def save(self, runs_dir=RUNS_DIR):
        os.makedirs(runs_dir, exist_ok=True)
        name = f"run_{time.strftime('%H%M%S')}_{self.cfg.get('name', 'custom')}_{self.outcome}.json"
        path = os.path.join(runs_dir, name)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f)
        return path


# --- run library ------------------------------------------------------------
def load_runs(runs_dir=RUNS_DIR):
    runs = []
    if not os.path.isdir(runs_dir):
        return runs
    for fn in sorted(os.listdir(runs_dir)):
        if fn.endswith(".json"):
            try:
                with open(os.path.join(runs_dir, fn)) as f:
                    run = json.load(f)
                run["_file"] = fn
                runs.append(run)
            except (json.JSONDecodeError, OSError):
                continue
    return runs


def rank_runs(runs, mode="Fastest hit"):
    if mode == "Total reward":
        return sorted(runs, key=lambda r: -r.get("total_reward", -1e9))
    hits = sorted([r for r in runs if r.get("hit")], key=lambda r: r["time_to_hit"])
    misses = sorted([r for r in runs if not r.get("hit")],
                    key=lambda r: r.get("closest_approach", 1e9))
    return hits + misses


# --- policies ----------------------------------------------------------------
def load_policy(model_path):
    """SB3 checkpoint path -> policy fn. Raises RuntimeError with a clear
    message if the model can't be loaded (GUI shows it in the status line)."""
    if not model_path or not os.path.isfile(model_path):
        raise RuntimeError(f"model not found: {model_path}")
    try:
        from stable_baselines3 import PPO
    except ImportError:
        raise RuntimeError("stable-baselines3 is not installed "
                           "(pip install stable-baselines3)")
    try:
        model = PPO.load(model_path, device="cpu")
    except Exception as e:
        raise RuntimeError(f"could not load {os.path.basename(model_path)}: {e}")

    def policy(obs):
        action, _ = model.predict(obs, deterministic=True)
        return action
    return policy
