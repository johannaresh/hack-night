"""PLACEHOLDER physics so the GUI works end-to-end before physics.py lands.

moterodiaz: this is NOT your engine — dronegym/runner.py automatically
switches to `dronegym.physics` the moment it exists with the same three
functions (derive_params, make_state, step). Frame conventions are the ones
in camera.py / README: world Z-up, body x-fwd / y-left / z-up, quat [w,x,y,z].

Model: first-order lag on body rates (attitude agility) and total thrust
(motor spool-up), quaternion integration, gravity, linear drag. Parameters
are derived from DroneConfig with crude-but-directionally-right scaling:
    max_thrust = K_T * prop_d_in^4 * (kv * battery_v)^2      [prop disc law]
    rate lag   grows with mass * frame size                  [inertia]
    motor lag  grows with prop size                          [spool-up]
"""

import numpy as np

from dronegym.camera import _quat_to_rot

G = 9.81
K_T = 4.5e-11          # tuned so freestyle_5inch lands at ~8:1 thrust-to-weight
MAX_RATE = 10.0        # rad/s commanded at full stick


def derive_params(cfg):
    """DroneConfig dict -> physical parameters dict."""
    mass = cfg["mass_g"] / 1000.0
    max_thrust = K_T * cfg["prop_diameter_in"] ** 4 * (cfg["motor_kv"] * cfg["battery_v"]) ** 2
    return {
        "mass": mass,
        "max_thrust": max_thrust,
        "tau_rate": 0.02 + cfg["mass_g"] * cfg["frame_size_mm"] * 1e-7,
        "tau_motor": 0.01 + 0.008 * cfg["prop_diameter_in"],
        "max_rate": MAX_RATE,
        "drag": 0.1 + 30.0 / cfg["frame_size_mm"],
        "hover_frac": min(1.0, mass * G / max_thrust),
        "cam_angle_deg": cfg["cam_angle_deg"],
    }


def make_state(pos, yaw=0.0):
    """Fresh state: at rest, level, facing world yaw angle (0 = +x)."""
    return {
        "pos": np.asarray(pos, dtype=float),
        "vel": np.zeros(3),
        "quat": np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)]),
        "rates": np.zeros(3),
        "thrust": 0.0,
    }


def _quat_mul(a, b):
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ])


def step(state, action, p, dt):
    """Advance one timestep. action = [thrust, roll, pitch, yaw] in [-1, 1]."""
    action = np.clip(np.asarray(action, dtype=float), -1.0, 1.0)

    thrust_cmd = (action[0] + 1.0) / 2.0 * p["max_thrust"]
    thrust = state["thrust"] + (thrust_cmd - state["thrust"]) * dt / p["tau_motor"]

    rate_cmd = action[1:4] * p["max_rate"]
    rates = state["rates"] + (rate_cmd - state["rates"]) * dt / p["tau_rate"]

    q = state["quat"] + 0.5 * _quat_mul(state["quat"], np.concatenate([[0.0], rates])) * dt
    q /= np.linalg.norm(q)

    R = _quat_to_rot(q)                      # body -> world
    accel = R @ np.array([0.0, 0.0, thrust / p["mass"]])
    accel += np.array([0.0, 0.0, -G])
    accel -= p["drag"] * state["vel"]

    vel = state["vel"] + accel * dt
    return {
        "pos": state["pos"] + vel * dt,
        "vel": vel,
        "quat": q,
        "rates": rates,
        "thrust": thrust,
    }
