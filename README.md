# DroneGym 🚁

An RL training gym for autonomous FPV drones, built in one hackathon night.
Configure a drone (prop size, motor KV, battery, camera uptilt, weight, frame),
train a PPO policy to fly at a spherical target using only what an onboard
camera would see — a bounding box, standing in for real CV via privileged
geometry — then fly and replay it in a full desktop GUI. Targets are either
**static** or **moving on a constant-velocity path** (counter-UAS-style
interception).

Five minutes of CPU training is enough for a first model that captures the
static target 10/10 times.

## Quick start

```bash
pip install -r requirements.txt
py gui.py
```

1. Pick a preset (or Custom and drag the sliders).
2. TRAIN MODEL — the default 280k steps takes ~5 min on a laptop CPU.
3. START LIVE RUN when the checkpoint auto-selects.
4. Watch the isometric world view, the drone's FPV view, and the attitude
   panel; every flight is saved and ranked in the REPLAYS strip.

## The GUI

| Panel | What it does |
|---|---|
| DRONE | 5 presets or fully custom parameters (mass, prop, KV, battery, uptilt, frame) |
| SCENARIO | Static or Intercept (target speed 2–10 m/s); **custom setup** checkbox opens a scenario editor with drag-and-drop drone/target placement, attitude fields, and target heading |
| MODEL | checkpoint dropdown, live-run launcher, Import/Export (with load + obs-dim verification on import) |
| TRAINING | steps slider with wall-clock estimate, one-click training of the configured drone *and scenario*; finished checkpoints auto-install and pre-select |
| Views | isometric 3D world (grid, trail, target flight line, HIT flash) · FPV (bbox exactly as the policy sees it, crosshair, HUD) · attitude (centered 3D drone + roll/pitch/yaw readout) |
| Playback | play/pause, frame scrub, 0.25×–4× speed; stick-command bars (throttle %, rates in deg/s) synced to every frame |
| REPLAYS | every run recorded to `runs/*.json`, ranked by fastest hit or total reward |

## CLI training

```bash
# full curriculum run (levels 0-3 widen spawn distribution on success)
py train.py --config freestyle_5inch --steps 2000000 --run-name full1

# quick static bake at fixed difficulty (what the GUI button runs)
py train.py --config freestyle_5inch --steps 280000 --run-name quick --difficulty 0

# intercept: constant-velocity target; omit --target-speed for random 2-10 m/s
py train.py --config freestyle_5inch --scenario intercept --target-speed 5 \
    --steps 500000 --run-name int5

tensorboard --logdir runs/          # curriculum/level, reward_terms/*, fps
python -m pytest tests/             # env + reward test suites
```

## Architecture

```
DroneGym/
├── gui.py                    # DearPyGui app: config, training, live runs, replays
├── train.py                  # PPO (SB3): vec envs, curriculum, per-term reward logs
├── baseline_p.py             # hand-tuned P-controller baseline
├── configs/                  # 5 drone presets (tiny whoop ... 10" long range)
├── dronegym/
│   ├── config.py             # DroneConfig: real prop/KV/voltage -> thrust, inertia
│   ├── physics.py            # 250 Hz rigid-body quad + rate controller, 14-dim state
│   ├── state.py              # spawn helpers (tilted/moving starts for curriculum)
│   ├── camera.py             # geometry-only bbox: pose + target -> (x, y, size, vis)
│   ├── env.py                # DroneTargetEnv: obs/action mapping, curriculum, events
│   ├── rewards.py            # weighted terms, logged individually while training
│   ├── task.py               # shared constants (dims, rates, radii)
│   ├── presets.py            # YAML -> DroneConfig loader
│   └── runner.py             # GUI<->env bridge: frame stacking, recording,
│                             #   InterceptTargetEnv (moving target + escape)
├── tests/                    # env contract + reward suites
└── runs/, checkpoints/       # training output + models (gitignored; share models
                              #   with the GUI's Export/Import buttons)
```

`EXECUTION_PLAN.md`, `RL_NOTES.md`, and `RL_HANDOFF.md` record the RL design
reasoning; `MD CONTEXT/` holds the physics derivations.

## Contracts (change these only as a team)

**Frames.** World Z-up. Body: x forward, y left, z up. Quaternion `[w,x,y,z]`
rotates body → world. Camera looks along body +x, tilted up by
`cam_angle_deg`; image coords normalized to [-1, 1], (0,0) at center, +x
right, +y up. FOV 120°.

**Action** (4,) in [-1, 1]: `[thrust, roll_rate, pitch_rate, yaw_rate]`.
Thrust is **hover-centred**: 0 commands exact hover, +1 max thrust — a fresh
Gaussian policy must not rocket off the pad. Rates scale to
`cfg.max_body_rate` (14 rad/s default).

**Observation.** The env emits 12 dims, clipped to ±5:

| idx | value |
|---|---|
| 0–3 | bbox x, y, size, visible (raw from camera.py) |
| 4–6 | body rates / 10 |
| 7–9 | gravity in body frame / g (unit attitude vector) |
| 10–11 | forward, vertical body velocity / 15 |

Policies see **48 dims**: a 4-frame stack (`VecFrameStack`, newest last,
zero-padded at reset). The stack is not optional — a vanished bbox is
identical whether the target left the frame on the left or right, so a single
frame is not Markov. `train.make_vec_env` is the single source of truth for
the wrapper stack; `runner.EpisodeRunner` replicates it exactly for the GUI.

**Episodes.** 50 Hz policy over 250 Hz physics (action repeat 5), 10 s cap.
Termination: capture (< 1 m), crash, out-of-arena (50 m), tumble (inverted),
target lost (> 0.5 s after first sighting), timeout — plus **escaped**
(> 40 m and receding) in intercept.

## Scenarios

**Static** — target fixed in space, placed by back-projecting sampled image
coordinates through the camera so it is visible at spawn. Curriculum levels
0–3 widen distance (5 → 30 m), off-axis placement, spawn tilt, and initial
speed; level 3 spawns 30% of targets outside the frame to force yaw-search.

**Intercept** — target spawns 12–20 m out and crosses the arena on a level
constant-velocity line (2–10 m/s) passing near the drone. Same observation
contract; the frame stack encodes target motion so the policy can lead the
intercept rather than tail-chase.

## Drone presets

| | mass | prop | KV | battery | uptilt |
|---|---|---|---|---|---|
| tiny_whoop | 27 g | 1.2" | 22000 | 1S | 15° |
| freestyle_5inch | 650 g | 5.1" | 1850 | 6S | 30° |
| longrange_7inch | 950 g | 7.0" | 1300 | 6S | 20° |
| cinelifter | 2200 g | 8.0" | 1100 | 6S | 10° |
| longrange_10inch | 1500 g | 10.0" | 880 | 6S | 12° |

Thrust, inertia, and motor lag are derived from this geometry in
`DroneConfig`, so the same policy architecture flies very different aircraft.

## Team

| Person | Built |
|---|---|
| **zsun** | fake-CV camera, GUI (views, scenario editor, in-GUI training, import/export, replays), runner/env bridge, presets |
| **moterodiaz** | physics engine, DroneConfig derivations, intercept training flags |
| **johannaresh** | RL environment, reward design, curriculum, PPO pipeline, test suites |
