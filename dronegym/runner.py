"""Episode runner: the bridge between the GUI and the real training env.

Wraps DroneTargetEnv (the single source of truth for physics stepping, obs
building, actions, rewards, termination) and replicates train.py's
VecFrameStack so a policy sees exactly what it saw in training:
OBS_DIM x N_STACK = 12 x 4 = 48 dims, newest frame last, zero-padded at reset.
Records every flight to runs/*.json for the GUI's replay browser.

The intercept scenario (README v2) lives here as a small subclass until it
graduates into env.py.
"""

import json
import os
import time

import numpy as np

from dronegym import physics
from dronegym.env import TARGET_MIN_Z, DroneTargetEnv
from dronegym.state import make_state, quat_from_euler
from dronegym.task import (ACTION_REPEAT, MAX_EPISODE_STEPS, N_STACK, OBS_DIM,
                           PHYSICS_DT, TARGET_RADIUS)

RUNS_DIR = "runs"
DT = ACTION_REPEAT * PHYSICS_DT            # policy-rate timestep: 0.02 s
MAX_T = MAX_EPISODE_STEPS * DT             # 10 s episode cap


def custom_quat(rpy_deg):
    """Aviation display convention (roll right+, pitch up+, yaw left+) -> quat.

    state.quat_from_euler's pitch is rotation about body +y, which is nose-DOWN
    positive in our frame; the GUI's attitude panel shows nose-up positive, so
    the sign flips here to keep 'what you typed' == 'what the panel reads'.
    """
    r, p, y = np.radians(np.asarray(rpy_deg, dtype=float))
    return quat_from_euler(r, -p, y)


class InterceptTargetEnv(DroneTargetEnv):
    """README v2: constant-velocity target + `escaped` termination.

    Reuses the static env's spawn machinery, then pushes the target out to
    intercept range along its sampled bearing (which preserves the visibility
    contract) and aims it to cross within ~6 m of the drone.
    """

    ESCAPE_RANGE = 40.0

    def __init__(self, cfg, difficulty=0, target_speed=None):
        super().__init__(cfg, difficulty)
        self._speed_req = target_speed
        self.target_vel = np.zeros(3)

    def reset(self, *, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        rng = self.np_random
        drone = self.state[physics.POS]

        bearing = self.target_pos - drone
        bearing /= max(np.linalg.norm(bearing), 1e-9)
        d = rng.uniform(12.0, 20.0)
        if bearing[2] < 0.0:                # don't push the target underground
            d = min(d, (TARGET_MIN_Z - drone[2]) / bearing[2])
        self.target_pos = drone + bearing * max(d, 2.0)

        speed = float(self._speed_req) if self._speed_req else rng.uniform(2.0, 10.0)
        aim = drone + np.array([rng.uniform(-6, 6), rng.uniform(-6, 6), 0.0])
        heading = aim - self.target_pos
        heading[2] = 0.0
        n = np.linalg.norm(heading)
        heading = heading / n if n > 1e-6 else np.array([1.0, 0.0, 0.0])
        self.target_vel = heading * speed

        obs, bbox = self._build_obs()       # target moved: rebuild first frame
        self._ever_seen = bool(bbox[3] > 0.5)
        info["distance"] = self._distance()
        return obs, info

    def step(self, action):
        self.target_pos = self.target_pos + self.target_vel * DT
        prev_dist = self._distance()
        obs, reward, terminated, truncated, info = super().step(action)
        if (not (terminated or truncated)
                and info["distance"] > self.ESCAPE_RANGE
                and info["distance"] > prev_dist):
            terminated = True
            info["event"] = "escaped"
            info["success"] = False
            info["terms_sum"] = dict(self._terms_sum)
        return obs, reward, terminated, truncated, info


class EpisodeRunner:
    """One episode, steppable frame-by-frame (for the GUI) or all at once."""

    def __init__(self, cfg, policy, scenario="static", target_speed=None,
                 difficulty=0, model_name="?", seed=None, custom=None):
        self.cfg = dict(cfg)               # raw dict, kept for run JSON / labels
        self.policy = policy
        self.model_name = model_name
        self.scenario = scenario

        dcfg = physics.derive_params(dict(cfg))
        if scenario == "intercept":
            self.env = InterceptTargetEnv(dcfg, difficulty, target_speed)
        else:
            self.env = DroneTargetEnv(dcfg, difficulty)

        obs, info = self.env.reset(seed=seed)
        if custom:                          # GUI scenario editor overrides spawn
            self._apply_custom(custom)
            obs, bbox = self.env._build_obs()
            self.env._ever_seen = bool(bbox[3] > 0.5)
            info = dict(info, distance=self.env._distance())
        # VecFrameStack semantics: zeros at reset except the newest slot (last).
        self._stack = np.zeros(OBS_DIM * N_STACK, dtype=np.float32)
        self._stack[-OBS_DIM:] = obs

        self.target0 = self.env.target_pos.copy()
        self.target_vel = np.asarray(getattr(self.env, "target_vel", np.zeros(3)),
                                     dtype=float)
        self.target_radius = TARGET_RADIUS
        self.t = 0.0
        self.done = False
        self.outcome = None    # 'hit' | 'crash' | 'oob' | 'tumble' | 'lost'
        #                        | 'escaped' | 'timeout'
        self.total_reward = 0.0
        self.closest = info["distance"]
        self.traj = {"pos": [], "quat": [], "bbox": [], "t": [], "act": []}
        self._record_frame(obs[0:4], None)

    def _apply_custom(self, custom):
        """Overwrite the sampled spawn with the GUI scenario editor's values.

        custom = {drone_pos, drone_rpy_deg (aviation display signs),
                  target_pos, target_heading_deg, target_speed}
        """
        env = self.env
        pos = np.asarray(custom["drone_pos"], dtype=float)
        pos[2] = max(pos[2], 0.3)                       # never spawn underground
        env.state = make_state(pos, quat=custom_quat(custom["drone_rpy_deg"]),
                               motor=env.hover_thrust)
        tp = np.asarray(custom["target_pos"], dtype=float)
        tp[2] = max(tp[2], TARGET_MIN_Z)
        env.target_pos = tp
        if isinstance(env, InterceptTargetEnv):
            h = np.radians(float(custom.get("target_heading_deg", 0.0)))
            spd = float(custom.get("target_speed") or 5.0)
            env.target_vel = np.array([np.cos(h), np.sin(h), 0.0]) * spd

    # -- gui-facing views ------------------------------------------------------
    @property
    def state(self):
        return {"pos": self.env.state[physics.POS],
                "quat": self.env.state[physics.QUAT]}

    @property
    def target(self):
        return self.env.target_pos

    def obs(self):
        """Stacked observation exactly as the policy sees it (48,)."""
        return self._stack.copy()

    # -- stepping ---------------------------------------------------------------
    def _record_frame(self, bbox, action):
        s = self.env.state
        self.traj["pos"].append(s[physics.POS].tolist())
        self.traj["quat"].append(s[physics.QUAT].tolist())
        self.traj["bbox"].append([float(b) for b in bbox])
        self.traj["t"].append(round(self.t, 4))
        # hover-centred contract: 0 == hover, so a resting frame records zeros
        act = [0.0] * 4 if action is None else \
            [float(a) for a in np.clip(np.asarray(action, dtype=float), -1, 1)]
        self.traj["act"].append(act)

    def step(self):
        if self.done:
            return
        action = self.policy(self._stack)
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._stack = np.roll(self._stack, -OBS_DIM)
        self._stack[-OBS_DIM:] = obs
        self.t += DT
        self.total_reward += float(reward)
        self.closest = min(self.closest, info["distance"])
        self._record_frame(obs[0:4], action)
        if terminated or truncated:
            self.done = True
            event = info.get("event") or "timeout"
            self.outcome = "hit" if event == "capture" else event

    def run(self):
        while not self.done:
            self.step()
        return self

    # -- recording ---------------------------------------------------------------
    def to_dict(self):
        return {
            "drone": self.cfg,
            "model": self.model_name,
            "dt": DT,
            "scenario": self.scenario,
            "target_speed": round(float(np.linalg.norm(self.target_vel)), 2),
            "outcome": self.outcome,
            "hit": self.outcome == "hit",
            "time_to_hit": round(self.t, 3) if self.outcome == "hit" else None,
            "closest_approach": round(self.closest, 3),
            "total_reward": round(self.total_reward, 2),
            "target": {"pos0": self.target0.tolist(),
                       "vel": self.target_vel.tolist(),
                       "radius": self.target_radius},
            "traj": self.traj,
        }

    def save(self, runs_dir=RUNS_DIR):
        os.makedirs(runs_dir, exist_ok=True)
        name = f"run_{time.strftime('%H%M%S')}_{self.cfg.get('name', 'custom')}_{self.outcome}.json"
        path = os.path.join(runs_dir, name)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f)
        return path


# --- run library --------------------------------------------------------------
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


# --- policies -------------------------------------------------------------------
def load_policy(model_path):
    """SB3 checkpoint path -> policy fn over stacked obs. Raises RuntimeError
    with a clear message if the model can't be loaded (GUI shows it)."""
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

    expected = OBS_DIM * N_STACK
    got = int(np.prod(model.observation_space.shape))
    if got != expected:
        raise RuntimeError(f"{os.path.basename(model_path)} expects {got}-dim obs, "
                           f"runner builds {expected} (12 x {N_STACK} stack)")

    def policy(obs):
        action, _ = model.predict(obs, deterministic=True)
        return action
    return policy
