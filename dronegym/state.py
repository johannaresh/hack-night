"""State-construction helpers the RL layer needs on top of physics.py.

physics.py owns the state layout and the dynamics; it exposes `make_state(pos,
yaw)` and `initial_state(cfg, rng)`, neither of which can express "spawn tilted,
already moving, motors already at hover" — which is exactly what the curriculum
needs. These two helpers fill that gap without touching moterodiaz's file.

Slices are imported from physics.py rather than redefined, so there is one
source of truth for the layout.
"""

import numpy as np

from dronegym.physics import MOTOR, OMEGA, POS, QUAT, VEL

STATE_DIM = 14
LEVEL_QUAT = np.array([1.0, 0.0, 0.0, 0.0])


def make_state(pos, vel=None, quat=None, rates=None, motor=0.0):
    """Assemble a (14,) state array. Anything omitted is zero / level."""
    s = np.zeros(STATE_DIM)
    s[POS] = pos
    if vel is not None:
        s[VEL] = vel
    s[QUAT] = LEVEL_QUAT if quat is None else quat
    if rates is not None:
        s[OMEGA] = rates
    s[MOTOR] = motor
    return s


def quat_from_euler(roll, pitch, yaw):
    """Euler angles [rad] -> quaternion [w, x, y, z], body -> world.

    ZYX (aerospace) convention: R = Rz(yaw) @ Ry(pitch) @ Rx(roll). Used to spawn
    a drone with a sampled heading and tilt.
    """
    cr, sr = np.cos(roll / 2.0), np.sin(roll / 2.0)
    cp, sp = np.cos(pitch / 2.0), np.sin(pitch / 2.0)
    cy, sy = np.cos(yaw / 2.0), np.sin(yaw / 2.0)
    return np.array([
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    ])


def hover_thrust(cfg):
    """Collective thrust [N] that exactly cancels gravity for this config."""
    return cfg.mass_kg * cfg.gravity


if __name__ == "__main__":
    from dronegym.camera import _quat_to_rot

    assert np.allclose(quat_from_euler(0, 0, 0), LEVEL_QUAT)
    r, p, y = 0.3, -0.2, 2.1
    Rx = np.array([[1, 0, 0], [0, np.cos(r), -np.sin(r)], [0, np.sin(r), np.cos(r)]])
    Ry = np.array([[np.cos(p), 0, np.sin(p)], [0, 1, 0], [-np.sin(p), 0, np.cos(p)]])
    Rz = np.array([[np.cos(y), -np.sin(y), 0], [np.sin(y), np.cos(y), 0], [0, 0, 1]])
    assert np.allclose(_quat_to_rot(quat_from_euler(r, p, y)), Rz @ Ry @ Rx), "euler"
    assert np.allclose(_quat_to_rot(quat_from_euler(0, 0, np.pi / 2)) @ [1, 0, 0],
                       [0, 1, 0], atol=1e-12)

    s = make_state([1, 2, 3], vel=[4, 5, 6], rates=[0.1, 0.2, 0.3], motor=7.0)
    assert s.shape == (STATE_DIM,)
    assert np.allclose(s[POS], [1, 2, 3]) and np.allclose(s[VEL], [4, 5, 6])
    assert np.allclose(s[OMEGA], [0.1, 0.2, 0.3]) and s[MOTOR] == 7.0
    assert np.allclose(s[QUAT], LEVEL_QUAT)
    print("state helpers ok (ZYX euler -> quat, 14-dim assembly)")
