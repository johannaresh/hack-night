"""Reward shaping, isolated so it can be retuned without touching env internals.

Design (see RL_NOTES.md section 3):
  * Progress is potential-based: r = d_prev - d_now. Summed over an episode it
    telescopes to "distance closed", so it cannot be farmed by loitering. A
    per-step -distance penalty would not have that property.
  * The terminal capture bonus dominates every shaping term, so parking at range
    and admiring the target never beats reaching it.
  * compute_reward ALWAYS returns every key in WEIGHTS, zeros included, so the
    TensorBoard curves never gap and a gamed term is visible in seconds.

WEIGHTS is the entire tuning surface. Retune here, and re-check the arithmetic
asserted in tests/test_rewards.py -- the classic failure modes are all arithmetic.
"""

import numpy as np

from dronegym.camera import get_bbox
from dronegym.stub_physics import POS, QUAT, RATES
from dronegym.task import FOV_DEG, TARGET_RADIUS

WEIGHTS = {
    # shaping
    "progress": 1.0,        # x metres closed per step  -> ~ +20 over a good episode
    "visibility": 0.02,     # per step, target centred   -> up to +10
    "lost_step": -0.10,     # per step, target not visible
    "smoothness": -0.02,    # x ||a - a_prev||^2         -> -1 to -3
    "rate": -0.01,          # x ||omega / 10||^2         -> -1 to -2
    "time": -0.01,          # per step                   -> -5
    # terminal
    "capture": 50.0,
    "crash": -25.0,
    "oob": -25.0,
    "tumble": -25.0,
    "lost": -10.0,
}

# Events that map onto a terminal weight of the same name.
EVENT_TERMS = ("capture", "crash", "oob", "tumble", "lost")

# Python float, not np.float64: every value in `terms` must be a plain float so
# the dict stays JSON-serialisable when it rides along in info.
_SQRT2 = float(np.sqrt(2.0))


def compute_reward(state, prev_state, action, prev_action, target_pos, cfg,
                   event=None, bbox=None):
    """Reward for one policy step.

    Args:
        state, prev_state: (14,) physics states, after and before the step
        action, prev_action: (4,) raw policy actions in [-1, 1]
        target_pos: (3,) target sphere centre, world frame
        cfg: DroneConfig (uses .cam_angle_deg)
        event: one of EVENT_TERMS, or None for a non-terminal step
        bbox: optional precomputed get_bbox output for `state`; recomputed if None
              (env passes its own to avoid doing the geometry twice)

    Returns:
        (reward: float, terms: dict[str, float]) -- terms holds the WEIGHTED
        contribution of every key in WEIGHTS, so sum(terms.values()) == reward.
    """
    state = np.asarray(state, dtype=float)
    prev_state = np.asarray(prev_state, dtype=float)
    target_pos = np.asarray(target_pos, dtype=float)

    terms = dict.fromkeys(WEIGHTS, 0.0)

    # Progress: potential-based, on privileged true distance (never bbox_size,
    # which goes as 1/depth and would explode exactly at capture range).
    d_now = float(np.linalg.norm(target_pos - state[POS]))
    d_prev = float(np.linalg.norm(target_pos - prev_state[POS]))
    terms["progress"] = WEIGHTS["progress"] * (d_prev - d_now)

    # Visibility: pay for keeping the target near the image centre.
    if bbox is None:
        bbox = get_bbox(state[POS], state[QUAT], target_pos, TARGET_RADIUS,
                        cfg.cam_angle_deg, FOV_DEG)
    if bbox[3] > 0.5:
        centring = 1.0 - float(np.linalg.norm(bbox[:2])) / _SQRT2
        terms["visibility"] = WEIGHTS["visibility"] * centring
    else:
        terms["lost_step"] = WEIGHTS["lost_step"]

    # Smoothness (CAPS-style action-difference regularisation) and rate penalty.
    da = np.asarray(action, dtype=float) - np.asarray(prev_action, dtype=float)
    terms["smoothness"] = WEIGHTS["smoothness"] * float(da @ da)
    w = state[RATES] / 10.0
    terms["rate"] = WEIGHTS["rate"] * float(w @ w)

    terms["time"] = WEIGHTS["time"]

    # EVENT_TERMS is the source of truth for the spelling of these five events.
    # An unrecognised event is tolerated (no terminal term fires) rather than
    # raised on, so that a truncation label like "timeout" cannot kill a run --
    # but it means a typo'd event silently forfeits its bonus. If
    # reward_terms/capture stays flat at zero while success rate is not, look
    # here first.
    if event in EVENT_TERMS:
        terms[event] = WEIGHTS[event]

    return float(sum(terms.values())), terms
