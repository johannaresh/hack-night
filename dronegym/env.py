"""Gymnasium env: fly a quadrotor at a target it can only see as a bbox.

Design notes (the non-obvious parts — see EXECUTION_PLAN.md sections 2.4 / 3 / 9):

  * Hover-centred thrust. `a[0] == 0` commands EXACT hover. A fresh Gaussian
    policy outputs ~0 mean, so the naive [-1,1] -> [0, max_thrust] mapping would
    command 2-4x hover on step one and the drone would rocket out of the arena
    before it ever learned anything.

  * Target placement back-projects along the CAMERA axis, not body +x. With a
    30 deg uptilt, a target on body +x sits a third of the way down the frame
    and level 3's bearings would fall out of it entirely. Sampling image coords
    and back-projecting through the camera basis makes the visibility contract
    ("obs[3] == 1 at spawn for levels 0-2") hold for any uptilt and any spawn
    attitude, because the tilt is absorbed by the same rotation that produced it.

  * The ground constraint SHORTENS the ray instead of clamping target z. A clamp
    would move the target off the sampled bearing and silently break that same
    contract; shortening keeps the bearing exactly.

  * Only the timeout is `truncated`. Everything else is `terminated`. Getting
    this backwards makes PPO bootstrap the value function past a crash, which is
    invisible in the reward curves and poisons the whole run.

  * Capture uses the privileged true distance, never bbox_size (which goes as
    1/depth and is zeroed the instant the target's centre leaves the frame).

Observation (12,), float32, clipped to [-5, 5]:
    0-3  bbox x, y, size, visible   (camera.py, unscaled)
    4-6  body rates / 10
    7-9  gravity in body frame / G  (unit vector = attitude)
    10   forward velocity / 15
    11   vertical velocity / 15
"""

import math
import time

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from dronegym import stub_physics
from dronegym.camera import _quat_to_rot, get_bbox
from dronegym.rewards import WEIGHTS, compute_reward
from dronegym.stub_physics import (G, POS, QUAT, RATES, VEL, make_state,
                                   quat_from_euler)
from dronegym.task import (ACT_DIM, ACTION_REPEAT, ARENA_RADIUS, CAPTURE_RADIUS,
                           FOV_DEG, LOST_GRACE_INITIAL, LOST_GRACE_STEPS,
                           MAX_EPISODE_STEPS, OBS_DIM, TARGET_RADIUS)

MAX_DIFFICULTY = 3

# Initial-state distribution per curriculum level. The reward never changes;
# only this table widens (EXECUTION_PLAN section 1).
#   d        : drone->target distance range [m]
#   img      : half-width of the sampled normalized image coords
#   tilt_deg : roll/pitch spawn range, +- deg
#   speed    : initial speed cap [m/s], random direction
CURRICULUM = {
    0: {"d": (5.0, 5.0), "img": 0.0, "tilt_deg": 0.0, "speed": 0.0},
    1: {"d": (5.0, 15.0), "img": 0.5, "tilt_deg": 10.0, "speed": 0.0},
    2: {"d": (5.0, 25.0), "img": 0.9, "tilt_deg": 20.0, "speed": 3.0},
    3: {"d": (5.0, 30.0), "img": 0.9, "tilt_deg": 30.0, "speed": 5.0},
}

OFFSCREEN_PROB = 0.3        # level 3: fraction of spawns outside the frame
OFFSCREEN_BX = (1.2, 3.0)   # |bx| range for those (up to ~79 deg off axis)
OFFSCREEN_BY = 0.5

TARGET_MIN_Z = 1.0          # target never inside the ground (capture radius is 1.0)
MIN_TARGET_DIST = 2.0       # shortening the ray below this -> resample
PLACEMENT_TRIES = 8

SPAWN_Z = (8.0, 12.0)       # room below for dives, far from the OOB sphere

RATE_SCALE = 10.0
VEL_SCALE = 15.0
OBS_LIMIT = 5.0

GRAVITY_WORLD = np.array([0.0, 0.0, -G])


class DroneTargetEnv(gym.Env):
    """Chase a target sphere using bbox observations.

    action = [thrust, roll_rate, pitch_rate, yaw_rate] in [-1, 1];
    thrust is hover-centred, rates scale to +-cfg.max_rate.
    """

    metadata = {"render_modes": []}

    def __init__(self, cfg, difficulty: int = 0):
        super().__init__()
        self.cfg = cfg
        self.difficulty = int(np.clip(difficulty, 0, MAX_DIFFICULTY))

        self.action_space = spaces.Box(-1.0, 1.0, (ACT_DIM,), dtype=np.float32)
        self.observation_space = spaces.Box(-OBS_LIMIT, OBS_LIMIT, (OBS_DIM,),
                                            dtype=np.float32)

        # Swap point for moterodiaz's physics.py: reassign this one attribute.
        self._physics_step = stub_physics.step

        self.state = None
        self.target_pos = None
        self._prev_action = np.zeros(ACT_DIM)
        self._steps = 0
        self._invisible = 0
        self._ever_seen = False
        self._terms_sum = dict.fromkeys(WEIGHTS, 0.0)

    # ------------------------------------------------------------------ api

    def set_difficulty(self, level: int):
        """Curriculum hook (train.py calls this through env_method)."""
        self.difficulty = int(np.clip(int(level), 0, MAX_DIFFICULTY))
        return self.difficulty

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        rng = self.np_random
        spec = CURRICULUM[self.difficulty]

        pos = np.array([0.0, 0.0, rng.uniform(*SPAWN_Z)])
        yaw = rng.uniform(-np.pi, np.pi)          # random heading at EVERY level
        tilt = math.radians(spec["tilt_deg"])
        roll = rng.uniform(-tilt, tilt) if tilt > 0.0 else 0.0
        pitch = rng.uniform(-tilt, tilt) if tilt > 0.0 else 0.0
        quat = quat_from_euler(roll, pitch, yaw)

        vel = np.zeros(3)
        if spec["speed"] > 0.0:
            direction = rng.normal(size=3)
            direction /= np.linalg.norm(direction)
            vel = direction * rng.uniform(0.0, spec["speed"])

        # Motor state starts at hover so the drone doesn't sag during the ~50 ms
        # it would otherwise take the first-order motor lag to spin up.
        self.state = make_state(pos, vel=vel, quat=quat,
                                motor=self.cfg.hover_thrust)
        self.target_pos = self._place_target(pos, quat)

        self._prev_action = np.zeros(ACT_DIM)
        self._steps = 0
        self._invisible = 0
        self._terms_sum = dict.fromkeys(WEIGHTS, 0.0)

        obs, bbox = self._build_obs()
        self._ever_seen = bool(bbox[3] > 0.5)

        info = {"distance": self._distance(), "terms": dict.fromkeys(WEIGHTS, 0.0),
                "event": None, "difficulty": self.difficulty}
        return obs, info

    def step(self, action):
        if self.state is None:
            raise RuntimeError("step() before reset()")

        action = np.clip(np.asarray(action, dtype=float).reshape(ACT_DIM), -1.0, 1.0)
        cmd = self._map_action(action)

        prev_state = self.state
        state = prev_state
        for _ in range(ACTION_REPEAT):     # 100 Hz physics under a 50 Hz policy
            state = self._physics_step(state, cmd, self.cfg)
        self.state = state
        self._steps += 1

        obs, bbox = self._build_obs()
        if bbox[3] > 0.5:
            self._ever_seen = True
            self._invisible = 0
        else:
            self._invisible += 1

        distance = self._distance()
        # obs[7:10] is gravity in body frame / G — a unit vector, so the clip
        # never touches it and obs[9] is the tumble test verbatim.
        event, terminated, truncated = self._check_events(distance, float(obs[9]))

        reward, terms = compute_reward(state, prev_state, action, self._prev_action,
                                       self.target_pos, self.cfg, event=event,
                                       bbox=bbox)
        self._prev_action = action
        for key, value in terms.items():
            self._terms_sum[key] += value

        info = {"distance": distance, "terms": terms, "event": event,
                "difficulty": self.difficulty}
        if terminated or truncated:
            info["success"] = event == "capture"
            info["terms_sum"] = dict(self._terms_sum)
        return obs, reward, terminated, truncated, info

    def _map_action(self, action):
        """Raw action in [-1, 1] -> (thrust_cmd [N], rate_cmd (3,) [rad/s]).

        a[0] == 0 is exact hover; +1 is max thrust. Public so tests (and
        baseline_p.py) can check the mapping without stepping physics.
        """
        a = np.clip(np.asarray(action, dtype=float).reshape(ACT_DIM), -1.0, 1.0)
        hover = self.cfg.hover_thrust
        thrust_cmd = float(np.clip(hover + a[0] * (self.cfg.max_thrust - hover),
                                   0.0, self.cfg.max_thrust))
        rate_cmd = a[1:4] * self.cfg.max_rate
        return thrust_cmd, rate_cmd

    # -------------------------------------------------------------- internals

    def _build_obs(self):
        """Returns (obs (12,) float32 clipped, raw bbox (4,)).

        The bbox comes back with the obs so step() can hand it straight to
        compute_reward instead of redoing the geometry.
        """
        state = self.state
        bbox = get_bbox(state[POS], state[QUAT], self.target_pos, TARGET_RADIUS,
                        self.cfg.cam_angle_deg, FOV_DEG)
        R_wb = _quat_to_rot(state[QUAT]).T          # world -> body, used twice
        vel_body = R_wb @ state[VEL]
        grav_body = R_wb @ GRAVITY_WORLD            # == stub_physics.gravity_body

        obs = np.empty(OBS_DIM)
        obs[0:4] = bbox
        obs[4:7] = state[RATES] / RATE_SCALE
        obs[7:10] = grav_body / G
        obs[10] = vel_body[0] / VEL_SCALE
        obs[11] = vel_body[2] / VEL_SCALE

        # Debug checklist #1: NaNs here mean the physics diverged. Clipping would
        # hide an inf, so check before it.
        if not np.isfinite(obs).all():
            raise FloatingPointError(
                f"non-finite observation: obs={obs} state={state} "
                f"target={self.target_pos} step={self._steps}")

        return np.clip(obs, -OBS_LIMIT, OBS_LIMIT).astype(np.float32), bbox

    def _distance(self):
        """Privileged true distance drone -> target [m]."""
        rel = self.target_pos - self.state[POS]
        return float(np.sqrt(rel @ rel))

    def _check_events(self, distance, grav_body_z):
        """Priority order matters: first hit wins and becomes `event`."""
        pos = self.state[POS]
        if distance < CAPTURE_RADIUS:
            return "capture", True, False
        if pos[2] < 0.0:
            return "crash", True, False
        if pos @ pos > ARENA_RADIUS * ARENA_RADIUS:
            return "oob", True, False
        if grav_body_z > 0.0:                      # gravity pointing up = inverted
            return "tumble", True, False
        grace = LOST_GRACE_STEPS if self._ever_seen else LOST_GRACE_INITIAL
        if self._invisible > grace:
            return "lost", True, False
        if self._steps >= MAX_EPISODE_STEPS:
            return "timeout", False, True          # the only truncation
        return None, False, False

    def _sample_image_coords(self):
        """Desired normalized image coords (bx, by) for this level."""
        rng = self.np_random
        level = self.difficulty
        if level == 0:
            return 0.0, 0.0
        if level >= 3 and rng.random() < OFFSCREEN_PROB:
            # Outside the frame -> the policy has to yaw-search to find it.
            sign = 1.0 if rng.random() < 0.5 else -1.0
            return sign * rng.uniform(*OFFSCREEN_BX), rng.uniform(-OFFSCREEN_BY,
                                                                  OFFSCREEN_BY)
        m = CURRICULUM[level]["img"]
        return rng.uniform(-m, m), rng.uniform(-m, m)

    def _place_target(self, drone_pos, quat):
        """Back-project sampled image coords through the camera axis.

        get_bbox() reads back exactly the (bx, by) we sampled, whatever the
        uptilt or spawn attitude — that is what makes the visibility contract
        hold. Camera basis mirrors camera.py lines 56-59.
        """
        rng = self.np_random
        d_lo, d_hi = CURRICULUM[self.difficulty]["d"]
        a = math.radians(self.cfg.cam_angle_deg)
        cam_fwd = np.array([math.cos(a), 0.0, math.sin(a)])
        cam_right = np.array([0.0, -1.0, 0.0])
        cam_up = np.array([-math.sin(a), 0.0, math.cos(a)])
        half_tan = math.tan(math.radians(FOV_DEG) / 2.0)
        R_bw = _quat_to_rot(quat)

        dir_world = R_bw @ cam_fwd  # fallback bearing, overwritten on the first try
        for _ in range(PLACEMENT_TRIES):
            bx, by = self._sample_image_coords()
            d = rng.uniform(d_lo, d_hi)
            dir_body = cam_fwd + bx * half_tan * cam_right + by * half_tan * cam_up
            dir_body /= np.linalg.norm(dir_body)
            dir_world = R_bw @ dir_body

            # Ground constraint: shorten d along the SAME ray (a z clamp would
            # move the target off the sampled bearing and break visibility).
            if dir_world[2] < 0.0 and drone_pos[2] + d * dir_world[2] < TARGET_MIN_Z:
                d = (TARGET_MIN_Z - drone_pos[2]) / dir_world[2]
                if d < MIN_TARGET_DIST:
                    continue
            return drone_pos + dir_world * d

        return drone_pos + dir_world * MIN_TARGET_DIST


if __name__ == "__main__":
    from dronegym.config_shim import load_config

    cfg = load_config("configs/freestyle_5inch.yaml")
    env = DroneTargetEnv(cfg, difficulty=0)

    obs, info = env.reset(seed=0)
    print(f"preset:          {cfg.name} (uptilt {cfg.cam_angle_deg:.0f} deg, "
          f"hover {cfg.hover_thrust:.2f} N, max {cfg.max_thrust:.2f} N)")
    print("level-0 obs:    ", np.round(obs, 4))
    print("  bbox x,y,size,vis =", np.round(obs[:4], 4))
    print("  distance          =", round(info["distance"], 4))
    # Debug checklist #2: if this fires, target placement is off the camera axis.
    assert obs[3] == 1.0, f"target not visible at level-0 spawn: {obs}"
    assert obs.shape == (OBS_DIM,) and obs.dtype == np.float32

    # a[0] = 0 must be exact hover, or a fresh policy rockets upward.
    thrust, rates = env._map_action(np.zeros(ACT_DIM))
    assert abs(thrust - cfg.hover_thrust) < 1e-9, thrust
    assert np.allclose(rates, 0.0)
    print(f"hover mapping:   a[0]=0 -> {thrust:.4f} N (hover {cfg.hover_thrust:.4f} N)")

    # Every level spawns the target inside the frame (level 3 sometimes not, by design).
    for level in (0, 1, 2):
        env.set_difficulty(level)
        for seed in range(50):
            o, _ = env.reset(seed=seed)
            assert o[3] == 1.0, (level, seed, o)
    print("visibility:      levels 0-2 visible at spawn over 50 seeds each")

    # 1000 random steps: finite obs throughout, inside the declared Box.
    env.set_difficulty(3)
    obs, _ = env.reset(seed=1)
    env.action_space.seed(1)
    actions = [env.action_space.sample() for _ in range(1000)]  # sampled outside the clock
    events = {}
    t0 = time.perf_counter()
    for i, action in enumerate(actions):
        obs, reward, terminated, truncated, info = env.step(action)
        assert np.isfinite(obs).all(), (i, obs)
        assert np.isfinite(reward), (i, reward)
        assert env.observation_space.contains(obs), (i, obs)
        if terminated or truncated:
            events[info["event"]] = events.get(info["event"], 0) + 1
            assert "success" in info and "terms_sum" in info
            obs, _ = env.reset()
    dt = time.perf_counter() - t0
    print(f"random rollout:  1000 steps, {1000 / dt:,.0f} steps/s "
          f"(incl. per-step asserts), events {events}")

    print("\nall env sanity checks passed")
