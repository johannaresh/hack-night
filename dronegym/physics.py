"""Minimal rigid-body-style quadrotor physics used by the RL environment."""

import numpy as np

POS = slice(0, 3)
VEL = slice(3, 6)
QUAT = slice(6, 10)
OMEGA = slice(10, 13)
MOTOR = 13


def pos(state):
    return np.asarray(state)[POS].copy()


def vel(state):
    return np.asarray(state)[VEL].copy()


def quat(state):
    return np.asarray(state)[QUAT].copy()


def omega(state):
    return np.asarray(state)[OMEGA].copy()


def motor(state):
    return np.asarray(state)[MOTOR].copy()


def quat_normalize(q):
    """Return q normalized under the scalar-first [w, x, y, z] convention."""
    q = np.asarray(q, dtype=float)
    norm = np.linalg.norm(q)
    if norm == 0.0:
        return np.array([1.0, 0.0, 0.0, 0.0])
    return q / norm


def quat_rotate(q, v):
    """Rotate vector v from body coordinates into world coordinates by q."""
    w, x, y, z = quat_normalize(q)
    v = np.asarray(v, dtype=float)
    # Equivalent to q * [0, v] * conjugate(q), expanded for clarity and speed.
    qvec = np.array([x, y, z])
    return v + 2.0 * (w * np.cross(qvec, v) + np.cross(qvec, np.cross(qvec, v)))


def quat_integrate(q, omega, dt):
    """Integrate body angular rate omega for dt and renormalize the attitude."""
    w, x, y, z = quat_normalize(q)
    p, r, s = np.asarray(omega, dtype=float)
    q_dot = 0.5 * np.array([
        -x * p - y * r - z * s,
        w * p + y * s - z * r,
        w * r - x * s + z * p,
        w * s + x * r - y * p,
    ])
    return quat_normalize(np.array([w, x, y, z]) + q_dot * dt)


def gravity_body(state):
    """Express standard world gravity in the vehicle body frame."""
    q = quat(state)
    q_inverse = q * np.array([1.0, -1.0, -1.0, -1.0])
    return quat_rotate(q_inverse, np.array([0.0, 0.0, -9.81]))


def velocity_body(state):
    """Express world-frame velocity in the vehicle body frame."""
    q = quat(state)
    q_inverse = q * np.array([1.0, -1.0, -1.0, -1.0])
    return quat_rotate(q_inverse, vel(state))


def initial_state(cfg):
    state = np.zeros(14, dtype=float)
    state[QUAT] = [1.0, 0.0, 0.0, 0.0]
    state[MOTOR] = cfg.mass_kg * cfg.gravity
    return state


def step(state, action, cfg, dt=1 / 250):
    """Advance the dynamics one timestep, without mutating state."""
    old = np.asarray(state, dtype=float)
    action = np.asarray(action, dtype=float)
    result = old.copy()

    position = old[POS].copy()
    velocity = old[VEL].copy()
    attitude = quat_normalize(old[QUAT])
    body_rate = old[OMEGA].copy()
    current_motor = old[MOTOR]

    thrust_cmd = (action[0] * 0.5 + 0.5) * cfg.max_thrust_n
    motor_new = current_motor + (thrust_cmd - current_motor) * dt / cfg.motor_tau_s
    rate_cmd = action[1:4] * cfg.max_body_rate
    angular_accel = cfg.kp_rate * (rate_cmd - body_rate)

    thrust_world = quat_rotate(attitude, np.array([0.0, 0.0, motor_new]))
    acceleration = (
        thrust_world
        + np.array([0.0, 0.0, -cfg.gravity * cfg.mass_kg])
        - cfg.lin_drag * velocity
    ) / cfg.mass_kg

    velocity += acceleration * dt
    position += velocity * dt
    body_rate += angular_accel * dt
    attitude = quat_integrate(attitude, body_rate, dt)

    result[POS] = position
    result[VEL] = velocity
    result[QUAT] = attitude
    result[OMEGA] = body_rate
    result[MOTOR] = motor_new
    return result


if __name__ == "__main__":
    from .config import DroneConfig

    cfg = DroneConfig(500, 5, 2300, 16, 0, 220)

    # Hover
    state = initial_state(cfg)
    hover_action = 2.0 * (cfg.mass_kg * cfg.gravity / cfg.max_thrust_n) - 1.0
    for _ in range(500):
        state = step(state, [hover_action, 0.0, 0.0, 0.0], cfg)
    assert abs(state[2]) < 1.0

    # Rate tracking
    state = initial_state(cfg)
    requested_rate = 0.5 * cfg.max_body_rate
    for _ in range(250):
        state = step(state, [hover_action, 0.5, 0.0, 0.0], cfg)
    assert abs(state[OMEGA][0] - requested_rate) < 0.1

    # Unit quaternion under arbitrary controls
    rng = np.random.default_rng(0)
    state = initial_state(cfg)
    for _ in range(500):
        state = step(state, rng.uniform(-1.0, 1.0, size=4), cfg)
    assert abs(np.linalg.norm(state[QUAT]) - 1.0) < 1e-6

    # Input is never changed by step.
    original = initial_state(cfg)
    before = original.copy()
    step(original, [hover_action, 0.0, 0.0, 0.0], cfg)
    assert np.array_equal(original, before)

    print("physics self-check OK")
