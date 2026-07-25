# DroneGym 🚁

RL training gym for autonomous FPV drones. Configure a drone (prop size, motor KV, camera angle, weight), then train a PPO agent to fly at spherical targets using simulated camera input — a privileged-info bounding box standing in for real computer vision. Targets are either **static** (v1) or **moving on a straight constant-velocity path** for interception training (v2 — see Scenarios).

Built in 6 hours for a hackathon.

## Team

| Person | Track | Owns |
|---|---|---|
| **zsun** | Fake CV + visualization + demo | `dronegym/camera.py`, `viz/`, `evaluate.py`, `configs/` |
| **moterodiaz** | Physics engine | `dronegym/physics.py`, `dronegym/config.py` |
| **johannaresh** | RL environment + training | `dronegym/env.py`, `dronegym/rewards.py`, `train.py` |

## Project structure

```
DroneGym/
├── README.md
├── requirements.txt
├── configs/                      # zsun — drone presets for the demo
│   ├── tiny_whoop.yaml
│   ├── freestyle_5inch.yaml
│   ├── cinelifter.yaml
│   ├── longrange_7inch.yaml
│   └── longrange_10inch.yaml
├── dronegym/
│   ├── __init__.py
│   ├── config.py                 # moterodiaz — DroneConfig dataclass: mass, prop_diameter,
│   │                             #   motor_kv, cam_angle_deg, battery_v → derived max_thrust,
│   │                             #   inertia, motor time constant
│   ├── physics.py                # moterodiaz — rigid-body quadrotor dynamics + inner
│   │                             #   P rate-controller (acro mode), pure NumPy
│   ├── camera.py                 # zsun — geometry-only bbox (no rendering): drone pose +
│   │                             #   cam uptilt + target position → bbox (x, y, size, visible)
│   ├── env.py                    # johannaresh — gymnasium.Env: obs/action/reset/termination
│   └── rewards.py                # johannaresh — reward shaping, isolated for fast iteration
├── train.py                      # johannaresh — SB3 PPO, vectorized envs, checkpointing
├── evaluate.py                   # zsun — roll out a checkpoint, dump trajectory for replay
└── viz/
    └── replay.py                 # zsun — rerun viewer: drone pose, camera frustum, target
                                  #   sphere, 2D bbox inset ("what the drone sees")
```

## Interface contract — agree before splitting up, don't break without telling the team

**Action space** (4,) in [-1, 1]: `[thrust, roll_rate, pitch_rate, yaw_rate]`
(collective thrust + body rate commands; physics' inner loop tracks the rates)

**Observation space** (14,) — *widened from 12 on 2026-07-24 for the intercept scenario; no models had been trained yet, so there is no compatibility cost, but env.py and runner.py must both build the 14-dim version:*

| idx | value | source |
|---|---|---|
| 0–1 | bbox center x, y (normalized image coords, 0 = center) | camera.py |
| 2 | bbox size (proxy for distance) | camera.py |
| 3 | target visible flag (0/1) | camera.py |
| 4–6 | body angular rates (rad/s) | physics.py |
| 7–9 | gravity vector in body frame (attitude) | physics.py |
| 10–11 | velocity forward, vertical (body frame, m/s) | physics.py |
| 12–13 | bbox drift rate: d/dt of bbox x, y (units/s, finite difference of consecutive frames; 0 while target not visible) | env.py / runner.py |

**`DroneConfig` fields:** `mass_g`, `prop_diameter_in`, `motor_kv`, `battery_v`, `cam_angle_deg`, `frame_size_mm`

**Physics state** (what `physics.step(state, action, cfg)` takes/returns): position (3), velocity (3), quaternion (4), angular rates (3), motor thrust state (1). `camera.py` and `env.py` consume this, never mutate it.

**Frame conventions** (defined in `camera.py`, physics must match): world frame Z-up; body frame x forward, y left, z up; quaternion `[w, x, y, z]` rotating body vectors into world frame. Camera looks along body +x, tilted up by `cam_angle_deg`; image coords normalized to [-1, 1], (0,0) at center, +x right, +y up.

## Scenarios

### Static target (v1)
The original scope and still the default: a fixed sphere, drone spawns 7–10 m away facing roughly toward it. Everything above describes this mode.

### Intercept — constant-velocity target (v2, scope added 2026-07-24)
Train the drone to intercept a target crossing the arena in a straight line at constant speed — counter-UAS style, an interceptor chasing down a Shahed-type fixed-course drone. Shared spec so all three tracks (and their agents) build the same thing:

- **Target motion:** `target_pos(t) = pos0 + vel * t`, `vel` constant for the whole episode, level flight (`vel[2] = 0`).
- **Per-episode randomization:** speed 2–10 m/s; heading random but constrained so the path passes within ~6 m of the drone spawn; target spawns 12–20 m out.
- **Observation:** the 14-dim layout above, unchanged between modes. Indices 12–13 (bbox drift rate) exist exactly for this — they let the policy *lead* the intercept instead of tail-chasing.
- **Termination:** static's hit / crash / timeout, plus `escaped` — target range > 40 m and increasing.
- **Reward:** same shaping family (closing progress + centering + hit bonus); prefer a closing-velocity term over raw distance delta once the target moves fast.
- **Curriculum:** train static first, then 2 m/s → 5 m/s → 8–10 m/s, saving a separate checkpoint per stage so the demo can show the progression.
- **Run JSON format:** `"target"` becomes `{"pos0": [x,y,z], "vel": [vx,vy,0], "radius": r}`. Static runs write `vel: [0,0,0]`. Replays reconstruct the target's path from this.

**Who does what:** johannaresh — target motion + `escaped` termination in env.py, obs indices 12–13, curriculum stages. zsun — GUI scenario picker (static / intercept + target speed), moving-target rendering in both views, run-format update. moterodiaz — no physics changes required.

## Timeline

| Time | Milestone |
|---|---|
| 0:00–0:30 | All: agree interfaces, repo + venv setup, split |
| 0:30–2:00 | moterodiaz: dynamics working · johannaresh: env + PPO on stub physics · zsun: bbox math + rerun skeleton |
| 2:00–2:30 | **Integration #1:** real physics into env, real bbox into obs, first real training run |
| 2:30–4:30 | johannaresh (+moterodiaz): reward/curriculum iteration · zsun: replay polish, presets, demo flow |
| 4:30–5:30 | Train final policies on 2–3 presets, record best runs |
| 5:30–6:00 | Demo prep, README polish, freeze |

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

`requirements.txt`: `gymnasium`, `stable-baselines3`, `numpy`, `pyyaml`, `rerun-sdk`

## Workflow

Everyone commits straight to `main` — no branches, no PRs. `git pull --rebase` before every push. The three tracks live in different files, so conflicts should be rare.

## Demo fallback

If training misbehaves late: a hand-scripted proportional policy (steer to center the bbox, throttle toward it) runs through the same env and viewer. The demo still shows the full gym + configurability story.
