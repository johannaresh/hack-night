# DroneGym Physics Engine — Session Context

**Track:** moterodiaz — `dronegym/physics.py`, `dronegym/config.py`
**Session date:** 2026-07-24
**Status:** Physics engine complete + Tier 2a shipped

---

## What's on main (current head)

| Commit | Feature |
|---|---|
| `fb9dcfd` | Core: rigid-body quadrotor + inner P rate controller, flat (14,) state, motor lag, quaternion attitude |
| `7b27982` | Safety: action clamp to [-1,1] |
| `4f80726` | Domain randomization `randomize_config()` + randomized `initial_state(rng)` |
| `9180580` | **Tier 2a:** rigid-body rotational dynamics + motor thrust clamp |

---

## Physics engine architecture

### State vector — flat `np.ndarray` shape `(14,)` — DO NOT CHANGE without team sync

```
[pos(3), vel(3), quat(4, scalar-first w,x,y,z), omega(3), motor_thrust(1)]
```

Index constants exported from `physics.py`:
- `POS = slice(0,3)`, `VEL = slice(3,6)`, `QUAT = slice(6,10)`, `OMEGA = slice(10,13)`, `MOTOR = 13`

### Action space `(4,)` in `[-1, 1]`

`[thrust, roll_rate, pitch_rate, yaw_rate]` — collective thrust + body-rate commands.

### `DroneConfig` fields (inputs)

`mass_g, prop_diameter_in, motor_kv, battery_v, cam_angle_deg, frame_size_mm`

Knobs with defaults: `ct=0.1, inertia_scale=1.0, motor_tau_s=0.03, kp_rate=20.0, max_body_rate=14.0, lin_drag=0.1, gravity=9.81`

Derived in `__post_init__`: `mass_kg, arm_m, max_thrust_n, inertia (3x3 diag), inertia_inv`

---

## step() dynamics walkthrough

```python
# 1. Motor lag (first-order, tau~30ms)
thrust_cmd = (action[0]*0.5 + 0.5) * cfg.max_thrust_n
motor_new = clip(motor + (thrust_cmd - motor)*dt/tau, 0, max_thrust)

# 2. Rigid-body rotational dynamics (Tier 2a, Flightmare pattern)
rate_cmd = action[1:4] * cfg.max_body_rate
torque_des = cfg.inertia @ (cfg.kp_rate * (rate_cmd - omega))   # J-scaled so kp_rate stays rad/s^2/rad/s
gyro_torque = cross(omega, cfg.inertia @ omega)                  # gyroscopic coupling
angular_accel = cfg.inertia_inv @ (torque_des - gyro_torque)

# 3. Translational dynamics
thrust_world = quat_rotate(q, [0, 0, motor_new])
accel = (thrust_world + [0,0,-g*m] - lin_drag*vel) / m

# 4. Semi-implicit Euler integration (dt = 1/250 s)
vel += accel * dt;  pos += vel * dt
omega += angular_accel * dt
q = quat_integrate(q, omega, dt)  # renormalized
```

---

## Observation helpers consumed by env.py

| Function | Returns | Obs indices |
|---|---|---|
| `omega(state)` | body angular rates (3,) | 4-6 |
| `gravity_body(state, g)` | gravity in body frame (3,) | 7-9 |
| `velocity_body(state)` | velocity in body frame (3,) | 10-11 |

Obs 12-13 (bbox drift rate) computed in `env.py` — physics not involved.

---

## Domain randomization

```python
from dronegym.config import randomize_config
rcfg = randomize_config(cfg, rng, pct=0.1)
# jitters: ct, inertia_scale, motor_tau_s, kp_rate, lin_drag
# fixed: max_body_rate, gravity (action/world contract)
```

---

## Key decisions

| Decision | Why |
|---|---|
| Flat `(14,)` array | Vectorizes; Eschmann 2023 validates for RL |
| Body-rate + collective thrust | Better sim-to-real than RPM; Kaufmann 2022 |
| Quaternion scalar-first | No gimbal lock; Nekoo 2022 |
| `kp_rate` scaled by J before inversion | Preserves rad/s^2/rad/s semantics after Tier 2a |
| SKIP Tier 2b per-motor mixing | Expands state to (17,) — breaks contract |
| SKIP quadratic drag | Linear damping sufficient for training speeds |

---

## Pending optional

- **Rotational drag:** `angular_accel -= ang_drag * omega` (1 line). Add if training shows spin without input. Add `ang_drag: float = 0.1` knob to `DroneConfig`.

---

## Team ownership (never edit others' files)

| File | Owner |
|---|---|
| `dronegym/physics.py`, `dronegym/config.py` | moterodiaz |
| `dronegym/env.py`, `dronegym/rewards.py`, `train.py` | johannaresh |
| `dronegym/camera.py`, `viz/`, `evaluate.py`, `configs/` | zsun |
