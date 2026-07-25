"""Tests for dronegym/env.py.

These lock the contracts everything downstream leans on:
  * the visibility contract (target in frame at spawn for levels 0-2, any
    preset uptilt, any seed) — if it breaks, reward is flat at the floor and
    no amount of PPO tuning helps;
  * hover-centred thrust (a[0] == 0 -> exactly hover);
  * terminated vs truncated (only the timeout truncates) — a silent value-
    bootstrapping bug otherwise;
  * obs bounds, so the Box(-5, 5) declaration stays honest.
"""

import math

import numpy as np
import pytest

from dronegym.camera import _quat_to_rot
from dronegym.presets import load_config
from dronegym.env import MAX_DIFFICULTY, TARGET_MIN_Z, DroneTargetEnv
from dronegym.rewards import WEIGHTS
from dronegym.physics import OMEGA, POS, QUAT, VEL
from dronegym.task import (ACT_DIM, CAPTURE_RADIUS, FOV_DEG, LOST_GRACE_STEPS,
                           MAX_EPISODE_STEPS, OBS_DIM)

# freestyle_5inch: 30 deg uptilt (the case that breaks naive body-x placement).
# tiny_whoop: 15 deg uptilt, 27 g, TWR 2.5 (the low-authority end).
PRESETS = ["freestyle_5inch", "tiny_whoop"]
N_SEEDS = 50

_CFG = {name: load_config(name) for name in PRESETS}


def _env(preset="freestyle_5inch", difficulty=0):
    return DroneTargetEnv(_CFG[preset], difficulty)


def _cam_forward_world(env):
    """Unit world vector along the camera axis for the current attitude."""
    a = math.radians(env.cfg.cam_angle_deg)
    return _quat_to_rot(env.state[QUAT]) @ np.array([math.cos(a), 0.0, math.sin(a)])


# --------------------------------------------------------------- reset / obs

@pytest.mark.parametrize("preset", PRESETS)
@pytest.mark.parametrize("level", [0, 1, 2])
def test_reset_obs_contract_and_visibility(preset, level):
    """Levels 0-2 must spawn the target inside the frame, every seed."""
    env = _env(preset, level)
    for seed in range(N_SEEDS):
        obs, info = env.reset(seed=seed)
        assert obs.shape == (OBS_DIM,)
        assert obs.dtype == np.float32
        assert np.isfinite(obs).all()
        assert env.observation_space.contains(obs)
        assert obs[3] == 1.0, f"{preset} level {level} seed {seed}: not visible {obs[:4]}"
        assert abs(obs[0]) <= 1.0 and abs(obs[1]) <= 1.0
        assert set(info) == {"distance", "terms", "event", "difficulty"}
        assert info["event"] is None and info["difficulty"] == level
        assert set(info["terms"]) == set(WEIGHTS)
        assert info["distance"] > CAPTURE_RADIUS      # never spawn inside capture
        assert env.target_pos[2] >= TARGET_MIN_Z - 1e-9   # never inside the ground


@pytest.mark.parametrize("preset", PRESETS)
def test_reset_level3_sometimes_spawns_off_frame(preset):
    """Level 3 must sometimes hide the target (that's the yaw-search drill)."""
    env = _env(preset, 3)
    visible = [env.reset(seed=s)[0][3] for s in range(N_SEEDS)]
    assert any(v == 0.0 for v in visible), "level 3 never spawns outside the frame"
    assert any(v == 1.0 for v in visible), "level 3 never spawns inside the frame"


@pytest.mark.parametrize("preset", PRESETS)
def test_ground_constraint_shortens_the_ray_without_losing_the_bearing(preset):
    """Targets pushed to the floor keep their sampled image coords.

    A z-clamp would be the obvious fix and would silently break visibility;
    shortening d along the same ray keeps the bearing exactly.
    """
    env = _env(preset, 2)
    shortened = 0
    for seed in range(200):
        obs, info = env.reset(seed=seed)
        assert env.target_pos[2] >= TARGET_MIN_Z - 1e-9
        if abs(env.target_pos[2] - TARGET_MIN_Z) < 1e-9:
            shortened += 1
            assert obs[3] == 1.0, f"ground-shortened target lost from frame: {obs[:4]}"
            assert info["distance"] > CAPTURE_RADIUS
    assert shortened > 0, "ground constraint never exercised — test is vacuous"


def test_reset_randomises_yaw_at_level_zero():
    """Yaw is randomised at EVERY level, so level 0 is not a fixed heading."""
    env = _env("freestyle_5inch", 0)
    headings = []
    for seed in range(20):
        env.reset(seed=seed)
        headings.append(_cam_forward_world(env)[:2].copy())
    spread = np.ptp(np.array(headings), axis=0)
    assert spread.min() > 0.5, headings


def test_obs_scaling_survives_extreme_state():
    """30 m/s, 20 rad/s and a point-blank target -> still inside Box(-5, 5)."""
    env = _env("freestyle_5inch", 0)
    env.reset(seed=0)
    env.state[VEL] = np.array([30.0, 30.0, -30.0])
    env.state[OMEGA] = np.array([20.0, -20.0, 20.0])
    env.target_pos = env.state[POS] + _cam_forward_world(env) * 0.05

    obs, bbox = env._build_obs()
    assert np.isfinite(obs).all()
    assert np.abs(obs).max() <= 5.0
    assert env.observation_space.contains(obs)
    assert obs[2] == 5.0, "bbox_size should be clipped, not passed through"
    assert bbox[2] > 5.0, "raw bbox is unscaled; only the obs copy is clipped"


# ------------------------------------------------------------ action mapping

@pytest.mark.parametrize("preset", PRESETS)
def test_map_action_is_hover_centred(preset):
    env = _env(preset)
    cfg = _CFG[preset]

    thrust, rates = env._map_action(np.zeros(ACT_DIM))
    assert thrust == pytest.approx(env.hover_thrust, abs=1e-9)
    assert np.allclose(rates, 0.0)

    thrust, rates = env._map_action(np.ones(ACT_DIM))
    assert thrust == pytest.approx(cfg.max_thrust_n, abs=1e-9)
    assert np.allclose(rates, cfg.max_body_rate)

    thrust, rates = env._map_action(-np.ones(ACT_DIM))
    assert thrust >= 0.0
    assert thrust == pytest.approx(max(0.0, 2 * env.hover_thrust - cfg.max_thrust_n),
                                   abs=1e-9)
    assert np.allclose(rates, -cfg.max_body_rate)

    # Out-of-range actions are clipped, not extrapolated.
    assert env._map_action(np.full(ACT_DIM, 9.0))[0] == pytest.approx(cfg.max_thrust_n)
    assert np.allclose(env._map_action(np.full(ACT_DIM, 9.0))[1], cfg.max_body_rate)


def test_zero_action_holds_a_perfect_hover():
    """a = 0 with a level spawn must be a fixed point of the dynamics."""
    env = _env("freestyle_5inch", 0)
    env.reset(seed=3)
    z0 = float(env.state[POS][2])
    for _ in range(100):
        env.step(np.zeros(ACT_DIM))
    assert abs(float(env.state[POS][2]) - z0) < 1e-6


# ------------------------------------------------------------- terminations

def test_ground_crash_terminates():
    env = _env("freestyle_5inch", 0)
    env.reset(seed=0)
    env.state[POS] = np.array([0.0, 0.0, -0.5])     # below ground, far from target
    obs, reward, terminated, truncated, info = env.step(np.zeros(ACT_DIM))
    assert terminated is True and truncated is False
    assert info["event"] == "crash"
    assert info["success"] is False
    assert info["terms"]["crash"] == WEIGHTS["crash"]


def test_zero_thrust_eventually_crashes():
    """The same event, reached by flying rather than by teleporting."""
    env = _env("freestyle_5inch", 0)
    env.reset(seed=1)
    for _ in range(MAX_EPISODE_STEPS):
        _, _, terminated, truncated, info = env.step(np.array([-1.0, 0.0, 0.0, 0.0]))
        if terminated or truncated:
            break
    assert terminated is True and truncated is False
    assert info["event"] == "crash"


def test_timeout_truncates_without_terminating():
    """Static hover at level 0 survives the full episode -> truncated only."""
    env = _env("freestyle_5inch", 0)
    env.reset(seed=2)
    running = dict.fromkeys(WEIGHTS, 0.0)
    for i in range(MAX_EPISODE_STEPS):
        obs, reward, terminated, truncated, info = env.step(np.zeros(ACT_DIM))
        assert reward == pytest.approx(sum(info["terms"].values()))
        for k, v in info["terms"].items():
            running[k] += v
        if i < MAX_EPISODE_STEPS - 1:
            assert not terminated and not truncated, (i, info["event"])
    assert truncated is True and terminated is False
    assert info["event"] == "timeout"
    assert info["success"] is False
    for k in WEIGHTS:
        assert info["terms_sum"][k] == pytest.approx(running[k])


def test_capture_sets_success():
    env = _env("freestyle_5inch", 0)
    env.reset(seed=4)
    # Just outside the capture sphere, moving straight at the target.
    env.state[POS] = env.target_pos + np.array([CAPTURE_RADIUS + 0.05, 0.0, 0.0])
    env.state[VEL] = np.array([-5.0, 0.0, 0.0])
    for _ in range(10):
        obs, reward, terminated, truncated, info = env.step(np.zeros(ACT_DIM))
        if terminated or truncated:
            break
    assert terminated is True and truncated is False
    assert info["event"] == "capture"
    assert info["success"] is True
    assert info["distance"] < CAPTURE_RADIUS
    assert info["terms"]["capture"] == WEIGHTS["capture"]


def test_capture_beats_crash_in_priority_order():
    """Below ground AND inside the capture sphere -> capture wins."""
    env = _env("freestyle_5inch", 0)
    env.reset(seed=5)
    env.target_pos = np.array([0.0, 0.0, 0.2])
    env.state[POS] = np.array([0.0, 0.0, -0.1])
    env.state[VEL] = np.zeros(3)
    _, _, terminated, truncated, info = env.step(np.zeros(ACT_DIM))
    assert terminated is True and truncated is False
    assert info["event"] == "capture"


def test_tumble_terminates_when_inverted():
    env = _env("freestyle_5inch", 0)
    env.reset(seed=6)
    env.state[QUAT] = np.array([0.0, 1.0, 0.0, 0.0])   # 180 deg roll: upside down
    _, _, terminated, truncated, info = env.step(np.zeros(ACT_DIM))
    assert terminated is True and truncated is False
    assert info["event"] == "tumble"


def test_out_of_bounds_terminates():
    env = _env("freestyle_5inch", 0)
    env.reset(seed=7)
    env.state[POS] = np.array([49.99, 0.0, 10.0])
    env.state[VEL] = np.array([20.0, 0.0, 0.0])
    _, _, terminated, truncated, info = env.step(np.zeros(ACT_DIM))
    assert terminated is True and truncated is False
    assert info["event"] == "oob"


def test_lost_after_grace_once_seen():
    """Yaw slowly away from a target seen at spawn -> `lost` after the grace."""
    env = _env("freestyle_5inch", 0)
    obs, _ = env.reset(seed=8)
    assert obs[3] == 1.0
    invisible_run = 0
    for i in range(MAX_EPISODE_STEPS):
        obs, _, terminated, truncated, info = env.step(np.array([0.0, 0.0, 0.0, 0.3]))
        invisible_run = 0 if obs[3] == 1.0 else invisible_run + 1
        if terminated or truncated:
            break
    assert terminated is True and truncated is False
    assert info["event"] == "lost"
    assert invisible_run == LOST_GRACE_STEPS + 1     # fires the step after the grace
    assert info["terms"]["lost"] == WEIGHTS["lost"]


def test_invisible_spawn_gets_the_long_grace():
    """Before the first sighting the budget is 100 steps, not 25."""
    env = _env("freestyle_5inch", 0)
    env.reset(seed=9)
    env.target_pos = env.state[POS] - _cam_forward_world(env) * 8.0   # behind
    env._ever_seen = False
    for i in range(LOST_GRACE_STEPS + 5):
        obs, _, terminated, truncated, info = env.step(np.zeros(ACT_DIM))
        assert obs[3] == 0.0
        assert not terminated, f"lost fired at step {i}, before the 100-step grace"


# ---------------------------------------------------------- misc contracts

def test_determinism_same_seed_same_rollout():
    a = _env("freestyle_5inch", 2)
    b = _env("freestyle_5inch", 2)
    obs_a, info_a = a.reset(seed=1234)
    obs_b, info_b = b.reset(seed=1234)
    assert np.array_equal(obs_a, obs_b)
    assert info_a["distance"] == info_b["distance"]
    assert np.array_equal(a.target_pos, b.target_pos)

    actions = np.random.default_rng(0).uniform(-1.0, 1.0, size=(10, ACT_DIM))
    for act in actions:
        ra = a.step(act)
        rb = b.step(act)
        assert np.array_equal(ra[0], rb[0])
        assert ra[1] == rb[1] and ra[2] == rb[2] and ra[3] == rb[3]
        assert ra[4]["event"] == rb[4]["event"]


def test_different_seeds_differ():
    env = _env("freestyle_5inch", 2)
    first = env.reset(seed=1)[0]
    other = env.reset(seed=2)[0]
    assert not np.array_equal(first, other)


def test_set_difficulty_clamps():
    env = _env("freestyle_5inch", 0)
    assert env.set_difficulty(2) == 2 and env.difficulty == 2
    assert env.set_difficulty(-5) == 0 and env.difficulty == 0
    assert env.set_difficulty(99) == MAX_DIFFICULTY and env.difficulty == MAX_DIFFICULTY
    assert env.reset(seed=0)[1]["difficulty"] == MAX_DIFFICULTY


def test_constructor_clamps_difficulty():
    assert DroneTargetEnv(_CFG["tiny_whoop"], difficulty=17).difficulty == MAX_DIFFICULTY
    assert DroneTargetEnv(_CFG["tiny_whoop"], difficulty=-1).difficulty == 0


@pytest.mark.parametrize("preset", PRESETS)
def test_random_rollouts_stay_finite_and_in_space(preset):
    env = _env(preset, 3)
    env.action_space.seed(0)
    obs, _ = env.reset(seed=0)
    for _ in range(600):
        obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
        assert np.isfinite(obs).all() and env.observation_space.contains(obs)
        assert np.isfinite(reward)
        assert set(info) >= {"distance", "terms", "event", "difficulty"}
        if terminated or truncated:
            assert "success" in info and "terms_sum" in info
            assert set(info["terms_sum"]) == set(WEIGHTS)
            obs, _ = env.reset()


def test_step_before_reset_raises():
    with pytest.raises(RuntimeError):
        _env().step(np.zeros(ACT_DIM))
