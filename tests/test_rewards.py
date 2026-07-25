"""Tests for dronegym/rewards.py.

Two kinds of test live here.

  * **Mechanics** — every term computes what EXECUTION_PLAN.md section 2.5 says
    it computes, the weighted terms always sum to the returned reward, and every
    key in WEIGHTS is present on every call (TensorBoard curves must never gap).
  * **Arithmetic regression** — RL_NOTES.md section 3 lists the reward-hacking
    failure modes to expect, and every one of them is an arithmetic failure of
    WEIGHTS rather than a code bug. They are encoded as tests so that a retune
    cannot silently reintroduce one. If editing WEIGHTS turns one of those red,
    the edit is wrong, not the test.

Everything here runs standalone against rewards.py using hand-built (14,) states
from state.make_state — no env.py, no physics rollout, no training.
"""

import math
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

# Standalone-friendly: `python -m pytest` already puts the repo root on the path,
# but a bare `pytest` invocation does not.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dronegym.camera import get_bbox                                    # noqa: E402
from dronegym.presets import load_config                            # noqa: E402
from dronegym.rewards import EVENT_TERMS, WEIGHTS, compute_reward       # noqa: E402
from dronegym.physics import POS, QUAT                                  # noqa: E402
from dronegym.state import make_state                 # noqa: E402
from dronegym.task import FOV_DEG, MAX_EPISODE_STEPS, TARGET_RADIUS     # noqa: E402

CFG = load_config("freestyle_5inch")

# Geometry-sensitive tests use a zero-uptilt stand-in. With cam_angle_deg = 0 a
# target dead ahead of a level drone lands at image (0, 0) *exactly*, so
# "perfectly centred" is an exact assertion rather than an approximate one. With
# the real 30 deg uptilt a level target sits BELOW centre (see camera.py's
# __main__), which test_uptilt_pushes_a_level_target_off_centre pins down.
CFG0 = replace(CFG, cam_angle_deg=0.0)

# Farthest the curriculum ever spawns a target (EXECUTION_PLAN.md section 3,
# WP1, level 3). Bounds the largest progress income an episode can earn.
MAX_SPAWN_DISTANCE = 30.0

HALF_TAN = math.tan(math.radians(FOV_DEG) / 2.0)
ZERO_ACT = np.zeros(4)
LEVEL_POS = np.array([0.0, 0.0, 10.0])


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #

def _target_at_image(drone_pos, bx, by, distance, cam_angle_deg=0.0):
    """World position of a target that lands at image coords (bx, by).

    Back-projects through camera.py's camera basis for a LEVEL drone (identity
    quaternion, so body frame == world frame). Because fwd/right/up are
    orthonormal, scaling out cancels and get_bbox returns (bx, by) exactly.
    This is the same construction env.py uses to guarantee framing at reset.
    """
    a = math.radians(cam_angle_deg)
    fwd = np.array([math.cos(a), 0.0, math.sin(a)])
    right = np.array([0.0, -1.0, 0.0])
    up = np.array([-math.sin(a), 0.0, math.cos(a)])
    d = fwd + bx * HALF_TAN * right + by * HALF_TAN * up
    return np.asarray(drone_pos, dtype=float) + d / np.linalg.norm(d) * distance


def _reward(state, prev_state=None, target_pos=None, *, action=ZERO_ACT,
            prev_action=ZERO_ACT, cfg=CFG0, event=None, bbox=None):
    """compute_reward with the uninteresting arguments defaulted."""
    if prev_state is None:
        prev_state = state
    if target_pos is None:
        target_pos = np.array([10.0, 0.0, 10.0])
    return compute_reward(state, prev_state, action, prev_action, target_pos,
                          cfg, event=event, bbox=bbox)


def _run_episode(states, actions, target, event):
    """Sum compute_reward over a hand-built trajectory.

    `event` fires on the final step only, as the env would. Returns
    (total_reward, per-term totals).
    """
    totals = dict.fromkeys(WEIGHTS, 0.0)
    total = 0.0
    last = len(states) - 1
    for i in range(1, len(states)):
        r, terms = _reward(states[i], states[i - 1], target,
                           action=actions[i], prev_action=actions[i - 1],
                           event=event if i == last else None)
        total += r
        for k, v in terms.items():
            totals[k] += v
    return total, totals


def _good_episode():
    """A plausible *good* run: closes 20 m in 4 s, framed and smooth, captures.

    Straight in along +x with a small lateral wobble, gentle yaw rate, tiny
    action deltas. Ends one capture-radius short of the target.
    """
    n = 200
    target = np.array([21.0, 0.0, 10.0])
    states, actions = [], []
    for i in range(n + 1):
        t = i / n
        phase = 6.0 * math.pi * t
        pos = np.array([20.0 * t, 0.3 * math.sin(phase), 10.0])
        rates = np.array([0.0, 0.0, 0.4 * math.cos(phase)])
        states.append(make_state(pos, rates=rates))
        actions.append(np.array([0.05, 0.0, 0.0, 0.04 * math.cos(phase)]))
    return states, actions, target


def _crash_episode():
    """A plausible *bad* run: dives into the ground 1.2 s in, target barely closer."""
    n = 60
    target = np.array([25.0, 0.0, 10.0])
    states, actions = [], []
    for i in range(n + 1):
        t = i / n
        pos = np.array([3.0 * t, 0.0, 10.0 - 10.0 * t])
        rates = np.array([2.0, 1.0, 0.0])
        states.append(make_state(pos, rates=rates))
        actions.append(np.array([-0.8 + 0.2 * math.sin(20.0 * t), 0.3, -0.4, 0.0]))
    return states, actions, target


# --------------------------------------------------------------------------- #
# 1. progress is potential-based
# --------------------------------------------------------------------------- #

def test_progress_telescopes_to_distance_closed():
    """Sum of progress over a trajectory == distance closed.

    This is the potential-based-shaping property (Ng et al. 1999, RL_NOTES
    section 0.2): the integral depends only on the endpoints, which is exactly
    why progress cannot be farmed by loitering or by orbiting the target.
    """
    start = np.array([0.0, 0.0, 10.0])
    end = np.array([20.0, 0.0, 10.0])
    target = np.array([25.0, 0.0, 10.0])
    n = 200

    d0 = float(np.linalg.norm(target - start))
    d1 = float(np.linalg.norm(target - end))
    assert d0 - d1 == pytest.approx(20.0)

    total = 0.0
    prev = make_state(start)
    for i in range(1, n + 1):
        cur = make_state(start + (end - start) * (i / n))
        _, terms = _reward(cur, prev, target)
        total += terms["progress"]
        prev = cur

    assert total == pytest.approx(WEIGHTS["progress"] * (d0 - d1), rel=1e-9)


def test_progress_cannot_be_farmed_by_loitering():
    """A closed loop back to the start earns exactly zero net progress."""
    target = np.array([30.0, 0.0, 10.0])
    n = 360

    def _on_circle(k):
        th = 2.0 * math.pi * k / n
        return make_state([5.0 * math.cos(th), 5.0 * math.sin(th), 10.0])

    total = 0.0
    prev = _on_circle(0)
    for k in range(1, n + 1):
        cur = _on_circle(k)
        _, terms = _reward(cur, prev, target)
        total += terms["progress"]
        prev = cur

    assert abs(total) < 1e-9


def test_progress_sign_and_scale():
    target = np.array([10.0, 0.0, 10.0])
    near = make_state([1.0, 0.0, 10.0])
    far = make_state([0.0, 0.0, 10.0])

    _, approaching = _reward(near, far, target)
    assert approaching["progress"] > 0.0
    assert approaching["progress"] == pytest.approx(WEIGHTS["progress"] * 1.0)

    _, receding = _reward(far, near, target)
    assert receding["progress"] < 0.0
    assert receding["progress"] == pytest.approx(-approaching["progress"])

    _, stationary = _reward(far, far, target)
    assert stationary["progress"] == 0.0


def test_progress_ignores_bbox_size():
    """Progress uses privileged true distance, never bbox_size (~1/depth).

    Same geometry, wildly different apparent size: a bbox handed in from far
    away must not change the progress term for a 1 m step.
    """
    target = np.array([10.0, 0.0, 10.0])
    near, far = make_state([1.0, 0.0, 10.0]), make_state([0.0, 0.0, 10.0])
    _, honest = _reward(near, far, target)
    _, spoofed = _reward(near, far, target, bbox=np.array([0.0, 0.0, 99.0, 1.0]))
    assert honest["progress"] == spoofed["progress"]


# --------------------------------------------------------------------------- #
# 2. terminal events
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("event", EVENT_TERMS)
def test_event_fires_exactly_its_weight(event):
    target = _target_at_image(LEVEL_POS, 0.0, 0.0, 8.0)
    s = make_state(LEVEL_POS)
    _, terms = _reward(s, s, target, event=event)

    assert terms[event] == WEIGHTS[event]
    for other in EVENT_TERMS:
        if other != event:
            assert terms[other] == 0.0, f"{other} fired alongside {event}"


def test_no_event_leaves_every_terminal_term_zero():
    target = _target_at_image(LEVEL_POS, 0.0, 0.0, 8.0)
    s = make_state(LEVEL_POS)
    _, terms = _reward(s, s, target, event=None)
    assert all(terms[e] == 0.0 for e in EVENT_TERMS)


# --------------------------------------------------------------------------- #
# 3. visibility / lost_step
# --------------------------------------------------------------------------- #

def test_centred_visible_target_pays_full_visibility():
    target = _target_at_image(LEVEL_POS, 0.0, 0.0, 8.0)
    s = make_state(LEVEL_POS)

    bbox = get_bbox(s[POS], s[QUAT], target, TARGET_RADIUS,
                    CFG0.cam_angle_deg, FOV_DEG)
    assert bbox[3] == 1.0 and np.allclose(bbox[:2], 0.0)

    _, terms = _reward(s, s, target)
    assert terms["visibility"] == WEIGHTS["visibility"]
    assert terms["lost_step"] == 0.0


def test_invisible_target_pays_lost_step_and_no_visibility():
    # Directly behind the drone -> camera.py returns all zeros.
    s = make_state(LEVEL_POS)
    target = np.array([-10.0, 0.0, 10.0])

    assert get_bbox(s[POS], s[QUAT], target, TARGET_RADIUS,
                    CFG0.cam_angle_deg, FOV_DEG)[3] == 0.0

    _, terms = _reward(s, s, target)
    assert terms["visibility"] == 0.0
    assert terms["lost_step"] == WEIGHTS["lost_step"]


def test_corner_bbox_is_worth_almost_nothing():
    """||(1, 1)|| / sqrt(2) == 1, so a target in the frame corner pays ~0."""
    eps = 1e-3
    target = _target_at_image(LEVEL_POS, 1.0 - eps, 1.0 - eps, 8.0)
    s = make_state(LEVEL_POS)

    _, terms = _reward(s, s, target)
    assert terms["lost_step"] == 0.0                      # still inside the frame
    assert 0.0 < terms["visibility"] < 0.01 * WEIGHTS["visibility"]


def test_visibility_decays_monotonically_from_centre():
    s = make_state(LEVEL_POS)
    last = float("inf")
    for off in (0.0, 0.2, 0.4, 0.6, 0.8, 0.95):
        target = _target_at_image(LEVEL_POS, off, 0.0, 8.0)
        _, terms = _reward(s, s, target)
        assert terms["visibility"] < last
        last = terms["visibility"]


def test_uptilt_pushes_a_level_target_off_centre():
    """RL_NOTES section 1.4: the camera is tilted UP, so a target level with the
    drone appears BELOW centre and pays less than a perfectly framed one."""
    assert CFG.cam_angle_deg > 0
    s = make_state(LEVEL_POS)
    target = np.array([10.0, 0.0, 10.0])                  # level with the drone

    bbox = get_bbox(s[POS], s[QUAT], target, TARGET_RADIUS,
                    CFG.cam_angle_deg, FOV_DEG)
    assert bbox[3] == 1.0 and bbox[1] < 0.0               # below centre

    _, terms = _reward(s, s, target, cfg=CFG)
    assert 0.0 < terms["visibility"] < WEIGHTS["visibility"]


# --------------------------------------------------------------------------- #
# 4. contract: terms sum to reward, every weight always present
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("event", [None, "capture", "crash", "oob", "tumble", "lost"])
def test_terms_sum_to_reward_and_cover_every_weight(event):
    target = _target_at_image(LEVEL_POS, 0.3, -0.2, 8.0)
    prev = make_state([-0.2, 0.0, 10.0], rates=[0.5, -1.2, 0.3])
    cur = make_state(LEVEL_POS, rates=[0.4, -1.0, 0.2])

    reward, terms = _reward(cur, prev, target, action=np.array([0.2, -0.1, 0.0, 0.5]),
                            prev_action=np.array([0.1, 0.0, 0.1, 0.4]), event=event)

    assert set(terms) == set(WEIGHTS)
    assert sum(terms.values()) == reward
    assert isinstance(reward, float)
    # `type(v) is float`, not isinstance: np.float64 subclasses float and would
    # slip through, but it is not JSON-serialisable when terms rides in info.
    assert all(type(v) is float for v in terms.values()), \
        {k: type(v).__name__ for k, v in terms.items()}


def test_every_weight_key_present_even_when_all_shaping_is_zero():
    """Zeros included — TensorBoard curves must never gap (RL_NOTES section 3)."""
    s = make_state(LEVEL_POS)
    _, terms = _reward(s, s, np.array([-10.0, 0.0, 10.0]))
    assert set(terms) == set(WEIGHTS)


# --------------------------------------------------------------------------- #
# 5. precomputed bbox == internally recomputed bbox
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("bx,by,visible", [(0.0, 0.0, True), (0.5, -0.4, True),
                                           (0.0, 0.0, False)])
def test_explicit_bbox_matches_internal_recompute(bx, by, visible):
    """env.py passes its own bbox to avoid doing the geometry twice (FPS)."""
    s = make_state(LEVEL_POS, rates=[0.3, 0.1, -0.2])
    prev = make_state([-0.1, 0.0, 10.0])
    target = (_target_at_image(LEVEL_POS, bx, by, 8.0) if visible
              else np.array([-8.0, 0.0, 10.0]))

    bbox = get_bbox(s[POS], s[QUAT], target, TARGET_RADIUS,
                    CFG0.cam_angle_deg, FOV_DEG)
    assert bool(bbox[3] == 1.0) is visible

    r_auto, t_auto = _reward(s, prev, target)
    r_given, t_given = _reward(s, prev, target, bbox=bbox)

    assert r_auto == r_given
    assert t_auto == t_given


# --------------------------------------------------------------------------- #
# 6. smoothness and rate regularisation
# --------------------------------------------------------------------------- #

def test_smoothness_penalises_only_action_changes():
    """CAPS-style ||a - a_prev||^2 (RL_NOTES section 0.3): holding an action,
    however large, is free; changing it is not."""
    s = make_state(LEVEL_POS)
    held = np.array([0.9, -0.7, 0.6, -0.5])

    _, same = _reward(s, s, action=held, prev_action=held)
    assert same["smoothness"] == 0.0

    _, jump = _reward(s, s, action=np.ones(4), prev_action=-np.ones(4))
    assert jump["smoothness"] < 0.0
    assert jump["smoothness"] == pytest.approx(WEIGHTS["smoothness"] * 16.0)

    _, nudge = _reward(s, s, action=np.array([0.1, 0.0, 0.0, 0.0]),
                       prev_action=ZERO_ACT)
    assert jump["smoothness"] < nudge["smoothness"] < 0.0


def test_rate_penalises_only_body_rates():
    _, still = _reward(make_state(LEVEL_POS, rates=[0.0, 0.0, 0.0]))
    assert still["rate"] == 0.0

    _, spinning = _reward(make_state(LEVEL_POS, rates=[10.0, 0.0, 0.0]))
    assert spinning["rate"] < 0.0
    assert spinning["rate"] == pytest.approx(WEIGHTS["rate"] * 1.0)

    _, tumbling = _reward(make_state(LEVEL_POS, rates=[10.0, 10.0, 10.0]))
    assert tumbling["rate"] < spinning["rate"] < 0.0


def test_time_penalty_fires_every_step():
    s = make_state(LEVEL_POS)
    for event in (None, "capture", "crash"):
        _, terms = _reward(s, s, event=event)
        assert terms["time"] == WEIGHTS["time"]


# --------------------------------------------------------------------------- #
# 7. anti-reward-hacking arithmetic (RL_NOTES section 3)
#
# These are regression tests on WEIGHTS itself. Each names the failure mode it
# prevents. A retune that breaks one of these reintroduces that failure mode.
# --------------------------------------------------------------------------- #

def test_no_hover_and_admire_the_target():
    """Failure mode 1: "hover and admire the target".

    Visibility pays per step and capture ends the income stream, so if a full
    episode of perfect framing ever approaches the capture bonus, parking at
    range and staring becomes competitive with flying in.
    """
    max_visibility_income = WEIGHTS["visibility"] * MAX_EPISODE_STEPS
    assert max_visibility_income < 0.5 * WEIGHTS["capture"]


def test_no_suicide_to_stop_the_bleeding():
    """Failure mode 2: "suicide to stop the bleeding".

    If the per-step bleed over a full episode exceeded the crash penalty, the
    fastest way to maximise return would be to hit the ground immediately.
    Symptom: rollout/ep_len_mean collapses to a few dozen steps.
    """
    assert MAX_EPISODE_STEPS * abs(WEIGHTS["time"]) < abs(WEIGHTS["crash"])


def test_no_cheapest_exit():
    """Failure mode 3: "fly away from the arena boundary".

    If leaving the arena were cheaper than crashing, the agent would find that
    exit instead of flying the task. Keep the two exits priced identically.
    """
    assert abs(WEIGHTS["oob"]) == abs(WEIGHTS["crash"])


def test_capture_dominates_all_achievable_shaping_income():
    """The sizing principle: the sparse terminal bonus must dominate.

    Progress telescopes, so an episode can never earn more than the spawn
    distance from it; visibility is capped at one weight per step. Every other
    shaping term is a penalty and can only reduce the total. So this is a hard
    upper bound on shaping income for any policy, and capture must beat it.
    """
    max_shaping_income = (WEIGHTS["progress"] * MAX_SPAWN_DISTANCE
                          + WEIGHTS["visibility"] * MAX_EPISODE_STEPS)
    assert WEIGHTS["capture"] > max_shaping_income
    assert all(WEIGHTS[k] <= 0.0
               for k in ("lost_step", "smoothness", "rate", "time"))


def test_giving_up_is_penalised_but_cheaper_than_crashing():
    """Failure mode 3, other half: "give up" must sit between flying and dying.

    Free `lost` would make bailing out the safe default; `lost` as bad as
    `crash` would make flying into the ground no worse than losing sight.
    """
    assert WEIGHTS["crash"] < WEIGHTS["lost"] < 0.0


def test_no_rate_penalty_paralysis():
    """Failure mode 4: "rate-penalty paralysis".

    An episode flown at half of the 5-inch preset's rate authority the whole way
    must still cost far less than the capture bonus, or the policy learns to
    freeze level and never turn.
    """
    half_authority = 0.5 * CFG.max_body_rate
    per_step = abs(WEIGHTS["rate"]) * 3.0 * (half_authority / 10.0) ** 2
    assert per_step * MAX_EPISODE_STEPS < 0.5 * WEIGHTS["capture"]


def test_shaping_terms_stay_within_an_order_of_magnitude():
    """RL_NOTES section 3 sizing principle: no shaping term should be able to
    swamp the others over an episode."""
    per_episode = {
        "progress": abs(WEIGHTS["progress"]) * 20.0,          # closing 20 m
        "visibility": abs(WEIGHTS["visibility"]) * MAX_EPISODE_STEPS,
        "time": abs(WEIGHTS["time"]) * MAX_EPISODE_STEPS,
    }
    assert max(per_episode.values()) < 10.0 * min(per_episode.values())


# --------------------------------------------------------------------------- #
# 8. end-to-end sanity: a good episode and a crash must separate cleanly
# --------------------------------------------------------------------------- #

def test_good_episode_scores_clearly_positive():
    states, actions, target = _good_episode()
    total, terms = _run_episode(states, actions, target, "capture")

    assert terms["capture"] == WEIGHTS["capture"]
    assert terms["progress"] == pytest.approx(20.0, abs=0.5)   # closed ~20 m
    assert terms["lost_step"] == 0.0                           # never lost sight
    assert terms["time"] == pytest.approx(-2.0)                # 200 steps
    # RL_NOTES section 3 sizes a good episode at about +70. Band is wide enough
    # to survive a retune but not a sign error anywhere in the sum.
    assert 40.0 < total < 100.0


def test_crash_episode_scores_clearly_negative():
    states, actions, target = _crash_episode()
    total, terms = _run_episode(states, actions, target, "crash")

    assert terms["crash"] == WEIGHTS["crash"]
    assert terms["capture"] == 0.0
    assert -60.0 < total < -10.0


def test_good_and_crash_episodes_are_cleanly_separated():
    good, _ = _run_episode(*_good_episode(), "capture")
    crash, _ = _run_episode(*_crash_episode(), "crash")
    assert good > 0.0 > crash
    assert good - crash > abs(WEIGHTS["crash"])
