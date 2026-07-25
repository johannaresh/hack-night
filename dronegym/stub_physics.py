"""Stub quadrotor physics — stand-in for moterodiaz's dronegym/physics.py.

Simple but honest enough to train against: first-order motor lag, first-order
rate tracking (standing in for the inner P rate controller), quaternion attitude
integration, linear drag. Expect to retrain once the real dynamics land.

Conventions match camera.py exactly (world Z-up; body x forward, y left, z up;
quaternion [w, x, y, z] rotating body vectors into world). `_quat_to_rot` is
imported from camera.py rather than reimplemented, so there is one source of
truth for the rotation convention.

State: flat float64 array, shape (14,)
    [0:3]   position, world [m]
    [3:6]   velocity, world [m/s]
    [6:10]  quaternion [w, x, y, z], body -> world
    [10:13] body angular rates [rad/s]
    [13]    collective thrust [N] (motor lag state)
"""

import numpy as np

from dronegym.camera import _quat_to_rot

POS = slice(0, 3)
VEL = slice(3, 6)
QUAT = slice(6, 10)
RATES = slice(10, 13)
MOTOR = 13
STATE_DIM = 14

DT = 0.01         # physics timestep [s]
G = 9.81
TAU_MOTOR = 0.05  # motor spin-up time constant [s]
TAU_RATE = 0.08   # inner rate-loop time constant [s]
DRAG = 0.3        # linear drag [1/s]

LEVEL_QUAT = np.array([1.0, 0.0, 0.0, 0.0])


def make_state(pos, vel=None, quat=None, rates=None, motor=0.0):
    """Assemble a (14,) state array. Anything omitted is zero / level."""
    s = np.zeros(STATE_DIM)
    s[POS] = pos
    if vel is not None:
        s[VEL] = vel
    s[QUAT] = LEVEL_QUAT if quat is None else quat
    if rates is not None:
        s[RATES] = rates
    s[MOTOR] = motor
    return s


def quat_from_euler(roll, pitch, yaw):
    """Euler angles [rad] -> quaternion [w, x, y, z], body -> world.

    ZYX (aerospace) convention: R = Rz(yaw) @ Ry(pitch) @ Rx(roll), i.e. roll
    about body x, then pitch about body y, then yaw about world z. env.py uses
    this to spawn a drone with a sampled heading and tilt.
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
    """Collective thrust [N] that exactly cancels gravity."""
    return cfg.mass_kg * G


def _quat_deriv(q, w):
    """0.5 * q (x) [0, w] for a body-frame angular velocity w."""
    qw, qx, qy, qz = q
    wx, wy, wz = w
    return 0.5 * np.array([
        -qx * wx - qy * wy - qz * wz,
        qw * wx + qy * wz - qz * wy,
        qw * wy - qx * wz + qz * wx,
        qw * wz + qx * wy - qy * wx,
    ])


def step(state, cmd, cfg):
    """Advance one DT.

    Args:
        state: (14,) state array, not mutated
        cmd:   (thrust_cmd [N], rate_cmd (3,) [rad/s]) — body rate setpoints
        cfg:   anything exposing .mass_kg and .max_thrust

    Returns:
        new (14,) state array
    """
    thrust_cmd, rate_cmd = cmd
    s = np.asarray(state, dtype=float)

    thrust = s[MOTOR] + (float(thrust_cmd) - s[MOTOR]) * DT / TAU_MOTOR
    thrust = float(np.clip(thrust, 0.0, cfg.max_thrust))

    rates = s[RATES] + (np.asarray(rate_cmd, dtype=float) - s[RATES]) * DT / TAU_RATE

    quat = s[QUAT] + _quat_deriv(s[QUAT], rates) * DT
    quat = quat / np.linalg.norm(quat)

    # Thrust acts along body +z; gravity and linear drag act in world frame.
    acc = (_quat_to_rot(quat) @ np.array([0.0, 0.0, thrust]) / cfg.mass_kg
           + np.array([0.0, 0.0, -G])
           - DRAG * s[VEL])
    vel = s[VEL] + acc * DT
    pos = s[POS] + vel * DT

    out = np.empty(STATE_DIM)
    out[POS], out[VEL], out[QUAT], out[RATES] = pos, vel, quat, rates
    out[MOTOR] = thrust
    return out


def gravity_body(state):
    """Gravity vector expressed in the body frame [m/s^2]. Level -> [0, 0, -G]."""
    return _quat_to_rot(np.asarray(state, dtype=float)[QUAT]).T @ np.array([0.0, 0.0, -G])


if __name__ == "__main__":
    class _Cfg:
        mass_kg = 0.65
        max_thrust = 8.0 * 0.65 * G

    cfg = _Cfg()
    hover = hover_thrust(cfg)

    # Hovering start + hover command -> altitude holds
    s = make_state([0, 0, 10], motor=hover)
    for _ in range(500):  # 5 s
        s = step(s, (hover, np.zeros(3)), cfg)
    assert abs(s[POS][2] - 10.0) < 0.5, s[POS]
    print("hover 5 s:       z =", round(float(s[POS][2]), 6))

    # Zero thrust -> falls
    s = make_state([0, 0, 10])
    for _ in range(100):  # 1 s
        s = step(s, (0.0, np.zeros(3)), cfg)
    assert s[POS][2] < 9.0, s[POS]
    print("free fall 1 s:   z =", round(float(s[POS][2]), 4))

    # Pure yaw rate -> stays level (gravity stays on body -z)
    s = make_state([0, 0, 10], motor=hover)
    for _ in range(200):  # 2 s
        s = step(s, (hover, np.array([0.0, 0.0, 3.0])), cfg)
    assert abs(gravity_body(s)[2] / G + 1.0) < 1e-3, gravity_body(s)
    print("yaw 2 s:         gravity_body/G =", np.round(gravity_body(s) / G, 4))

    # Roll rate -> tilts, gravity leaks onto body y
    s = make_state([0, 0, 10], motor=hover)
    for _ in range(50):  # 0.5 s
        s = step(s, (hover, np.array([2.0, 0.0, 0.0])), cfg)
    assert abs(gravity_body(s)[1]) > 0.1, gravity_body(s)
    print("roll 0.5 s:      gravity_body/G =", np.round(gravity_body(s) / G, 4))

    # Quaternion stays unit, state stays finite
    assert abs(np.linalg.norm(s[QUAT]) - 1.0) < 1e-9
    assert np.isfinite(s).all()

    # Thrust is clipped to the config maximum
    s = step(make_state([0, 0, 10], motor=cfg.max_thrust), (1e6, np.zeros(3)), cfg)
    assert s[MOTOR] <= cfg.max_thrust + 1e-9, s[MOTOR]

    # quat_from_euler matches Rz(yaw) @ Ry(pitch) @ Rx(roll) exactly
    assert np.allclose(quat_from_euler(0, 0, 0), LEVEL_QUAT)
    r, p, y = 0.3, -0.2, 2.1
    Rx = np.array([[1, 0, 0], [0, np.cos(r), -np.sin(r)], [0, np.sin(r), np.cos(r)]])
    Ry = np.array([[np.cos(p), 0, np.sin(p)], [0, 1, 0], [-np.sin(p), 0, np.cos(p)]])
    Rz = np.array([[np.cos(y), -np.sin(y), 0], [np.sin(y), np.cos(y), 0], [0, 0, 1]])
    assert np.allclose(_quat_to_rot(quat_from_euler(r, p, y)), Rz @ Ry @ Rx), "euler convention"
    # Yaw-only rotation turns body +x into world +y at 90 deg
    assert np.allclose(_quat_to_rot(quat_from_euler(0, 0, np.pi / 2)) @ [1, 0, 0],
                       [0, 1, 0], atol=1e-12)
    print("euler->quat:     ok (ZYX, body->world)")

    print("all stub physics sanity checks passed")
