# RL track — handoff notes

What landed on `main` in commit `c467494`, what the rest of the team needs to know,
and what is still unverified. Written by johannaresh (RL env + training).

Design rationale lives in `RL_NOTES.md`; the build plan in `EXECUTION_PLAN.md`.
This file is only the parts that affect **other people's code**.

---

## 1. What landed

| file | what it is |
|---|---|
| `dronegym/env.py` | `DroneTargetEnv` — gymnasium env, 12-dim bbox obs, 4-dim CTBR action |
| `dronegym/rewards.py` | reward shaping, isolated so it can be retuned without touching the env |
| `dronegym/task.py` | task constants shared by env + rewards (capture radius, arena, episode cap) |
| `dronegym/state.py` | the two state helpers `physics.py` doesn't provide (tilted/moving spawn, euler→quat) |
| `dronegym/presets.py` | loads `configs/*.yaml` into the real `DroneConfig` |
| `train.py` | PPO, vectorised envs, curriculum + reward-term callbacks, checkpointing |
| `baseline_p.py` | hand-scripted P controller — env validator and demo fallback |
| `tests/` | 69 tests (31 env, 38 rewards) |

`dronegym/config_shim.py` was deleted — it was a placeholder for `config.py` and is no
longer needed. **Nothing owned by zsun or moterodiaz was modified**: `camera.py`,
`physics.py`, `config.py`, `runner.py`, `gui.py`, `stub_physics.py` and `configs/*` all
came through the merge untouched.

---

## 2. Landmines — read this section if nothing else

### 2.1 `evaluate.py` MUST build the env through `train.make_vec_env` — zsun

The policy sees **48 dims, not 12**. The env returns 12, and `VecFrameStack(n_stack=4)`
stacks four frames on top.

```python
from train import make_vec_env
venv = make_vec_env("freestyle_5inch", n_envs=1, difficulty=0, seed=0)
model = PPO.load("runs/<name>/best_model.zip")
```

Load a checkpoint without that identical wrapper stack and you get either a
48-vs-12 dimension error or — much worse — **a policy that loads fine and flies like
garbage with no error at all**. `make_vec_env` is deliberately the single definition of
the stack; call it rather than reproducing it.

Why the stack is required rather than an optimisation: `camera.py` returns `np.zeros(4)`
both when the target is behind the camera *and* when it leaves the frame. `obs[0:4]` is
therefore bit-identical whether the target slid off the left edge or the right edge, so a
single frame cannot tell the agent which way to yaw to reacquire — the state is not
Markov. Four frames encode the drift direction.

### 2.2 `env.py` inverts the thrust mapping in `physics.step` — moterodiaz

`physics.step` maps thrust as `(a[0] * 0.5 + 0.5) * max_thrust_n`, which puts **hover at
`a[0] = −0.80`** on the freestyle preset (TWR 10.1). A freshly initialised Gaussian
policy outputs ≈0 mean, so fed straight in it would command ~5× hover on step one and
leave the arena before learning anything.

So `env._physics_action()` keeps a hover-centred action space for the policy
(`a[0] == 0` → exact hover) and converts to your contract with `2T/max_thrust_n − 1`.
The inversion is exact and round-trip tested.

**If you change the thrust mapping in `physics.step`, `env._physics_action` needs the
matching change.** Nothing will crash — the drone will just fly wrong.

### 2.3 Timestep coupling

`physics.step` runs at `dt = 1/250`. The env applies each action for
`ACTION_REPEAT = 5` physics steps, giving a **50 Hz policy**. That rate is baked into
`gamma = 0.995` (effective horizon is `1/(1−γ)` *steps* = 200 steps = 4 s). If the
physics timestep changes, change `ACTION_REPEAT` in `dronegym/task.py` to keep the
policy at 50 Hz, or `gamma` becomes wrong and the agent goes structurally blind to
whether it ever reaches the target.

### 2.4 Event-name spellings

`rewards.EVENT_TERMS` is the source of truth for the five terminal events
(`capture`, `crash`, `oob`, `tumble`, `lost`). Import it rather than hardcoding strings.
An unrecognised event is *silently tolerated* — no terminal reward fires — deliberately,
so that a label like `"timeout"` cannot kill a run. The symptom of a typo is
`reward_terms/capture` flat at zero while success rate is not.

### 2.5 `*.zip` is in `.gitignore`

Trained checkpoints (`best_model.zip`, `final_model.zip`) **cannot be committed** without
`git add -f`. Hand them over directly, or we relax the ignore rule for `runs/`.

---

## 3. Interface contract as actually built

**Action** `(4,)` in `[-1, 1]`: `[thrust, roll_rate, pitch_rate, yaw_rate]`.
`thrust = 0` is exact hover; rates scale to `±cfg.max_body_rate`.

**Observation** `(12,)` float32, clipped to `[-5, 5]`:

| idx | value | scaling |
|---|---|---|
| 0–3 | bbox x, y, size, visible | none (already normalised) |
| 4–6 | body rates | ÷ 10 |
| 7–9 | gravity in body frame | ÷ 9.81 → unit vector (attitude) |
| 10 | forward velocity (body) | ÷ 15 |
| 11 | vertical velocity (body) | ÷ 15 |

No `VecNormalize`, by design — the scaling is fixed and lives in the env, so there is no
running-statistics file to save alongside a checkpoint and no way to silently load the
wrong one.

**`info` dict** — every step: `distance`, `terms`, `event`, `difficulty`. On episode end
additionally `success` (bool, true only on capture) and `terms_sum` (per-term totals).

---

## 4. Requests to other tracks

### 4.1 `camera.py` blanks the target at point-blank range — zsun

`get_bbox` returns all zeros when `|bbox_x| > 1 or |bbox_y| > 1`, testing the **sphere's
centre**. At close range the sphere can fill most of the frame with its centre just
outside the edge, so the target reads as invisible exactly at the moment of capture —
a discontinuity in both observation and reward at the worst possible place.

Any one of these fixes it, your call which:
- a margin, e.g. `> 1.0 + bbox_size / 2`
- a partial-visibility test using `bbox_size`
- clip `bbox_x/y` to `[-1, 1]` but keep `visible = 1` while the sphere overlaps the frame

Not blocking — training works today — but expect visibility to flicker on final approach
until it changes.

### 4.2 Observation has no lateral velocity

Indices 10–11 are forward and vertical body-frame velocity. Sideways drift is
unobservable, so the policy cannot directly damp lateral oscillation and may wobble on
approach. Frame stacking lets it infer drift from bbox motion, so this is **accepted, not
a bug** — flagging it so nobody is surprised by the wobble. A 13th dim would fix it if we
ever want it.

### 4.3 `longrange_10inch` derives to TWR 14.7 — moterodiaz

Derived thrust-to-weight per preset, from the current `config.py` calibration:

| preset | TWR |
|---|---|
| tiny_whoop | 2.9 |
| cinelifter | 6.4 |
| freestyle_5inch | 10.1 |
| longrange_7inch | 12.2 |
| longrange_10inch | **14.7** |

14.7:1 on a 10-inch long-range build looks high (those are usually gentle, ~4:1). Worth a
glance at `ct` / the prop-disc scaling. Not blocking — we train freestyle first.

---

## 5. Status: verified vs not

**Verified against the real physics** (post-merge):
- 69/69 tests pass
- `python -m dronegym.env` self-check: target visible at spawn across levels 0–2 over
  50 seeds each, hover mapping exact, 1000 random steps all finite
- all 5 presets load and can hover

**NOT yet revalidated after the physics swap** — both are re-runs, not rewrites:
- `baseline_p.py` gains. On the old stub physics it scored **100% / 85% / 57% / 36%**
  success at levels 0–3. The real physics has rigid-body inertia and gyroscopic torque
  where the stub had a first-order approximation, so the gains likely need a retune.
  Those numbers are also the baseline PPO has to beat.
- A training smoke run. 200k steps ran clean on the stub (no NaNs, episode length stable
  at ~75, artifacts landed correctly).

---

## 6. Running it

```bash
python -m pytest -q                       # 69 tests
python -m dronegym.env                    # env self-check + throughput
python baseline_p.py --episodes 100       # P-controller baseline, all levels
python train.py --config freestyle_5inch --steps 2000000 --run-name run1
tensorboard --logdir runs/
```

In TensorBoard watch `curriculum/level`, `curriculum/success_rate`, `reward_terms/*`,
`rollout/ep_len_mean`. **`reward_terms/*` is the important one** — when total reward
rises but the flight looks worse, those per-term curves name the gamed term in seconds.

`--difficulty N` pins the curriculum at one level and disables promotion, for debugging.

### Throughput: ~1,100 FPS, and it is the machine, not the code

PPO sustains roughly 1,100 steps/s on this laptop, so 2M steps ≈ 30 minutes. I checked
whether that was our code: an **identical pure-numpy loop** degrades 226k → 37k iters/s
across three consecutive runs with nothing else changing. It is the i7-12700H scheduler
migrating the process onto E-cores, not the env, the wrapper stack, GC, or torch.

Practical consequence: **running three presets in parallel will contend rather than
scale.** Prefer sequential runs, or pin to P-cores if someone wants to chase it.

---

## 7. Tuning notes

- `rewards.WEIGHTS` is the entire tuning surface. The binding constraint is
  capture-dominance: `capture (50) > 30 m of progress + 500 steps of visibility (40)`.
  Only 25% headroom. Raising `visibility` above 0.04, or widening level-3 spawn distance
  past ~40 m, reintroduces "hover and admire the target". A test asserts this.
- `tiny_whoop` will be the jitteriest preset (27 g at 22000 KV). Expect to raise
  `WEIGHTS["smoothness"]` 2–4× for it.
- Curriculum promotes on rolling success rate > 0.7 over 100 episodes, not on a reward
  plateau. If success oscillates at a promotion boundary, widen the window to 200 rather
  than lowering the threshold.
- If a level stalls, widen the **previous** level's spawn ranges rather than weakening the
  reward. The reward is meant to stay fixed; only the initial-state distribution moves.
