# Execution plan — RL layer, end to end

Build `dronegym/env.py`, `dronegym/rewards.py`, `train.py` plus the stubs, tests, and
baseline needed to train PPO policies that fly a quadrotor at a target from bbox
observations. This document is **self-contained**: an agent executing a work package
needs only this file plus the repo. `RL_NOTES.md` is the research rationale; where the
two disagree, this file wins (it resolves everything RL_NOTES left open — see §9).

Branch: `rl`. Commit per work package with a one-line message.

---

## 0. Ground truth

**Exists (do not modify):**
- `dronegym/camera.py` — zsun's bbox geometry. `get_bbox(drone_pos, drone_quat,
  target_pos, target_radius, cam_angle_deg, fov_deg=120.0) -> np.array([x, y, size,
  visible])`. Returns `np.zeros(4)` when target is behind camera OR center out of
  frame. Frame conventions are documented in its docstring and are authoritative.
- `configs/*.yaml` — 5 presets, fields: `name, mass_g, prop_diameter_in, motor_kv,
  battery_v, cam_angle_deg, frame_size_mm`.
- `README.md` — interface contract (obs 12-dim, action 4-dim in [-1,1], physics state
  14 numbers).

**Does not exist yet (owned by others — we stub, never block):**
- `dronegym/physics.py`, `dronegym/config.py` (moterodiaz)
- `evaluate.py`, `viz/replay.py` (zsun)

**We create:**
```
requirements.txt                  # README specs contents; file missing
dronegym/stub_physics.py          # our stand-in until moterodiaz delivers
dronegym/config_shim.py           # yaml -> config object, until config.py lands
dronegym/env.py
dronegym/rewards.py
train.py
baseline_p.py                     # hand-scripted fallback policy + env validator
tests/test_env.py
tests/test_rewards.py
```

**Coordinate systems (from camera.py, physics must match):** world Z-up; body x
forward, y left, z up; quaternion `[w, x, y, z]` rotating body→world; camera along
body +x tilted up by `cam_angle_deg`; image coords [-1,1], +x right, +y up.

---

## 1. Locked design decisions

These are settled — do not relitigate during implementation.

| decision | value | why (RL_NOTES §) |
|---|---|---|
| memory | `VecFrameStack(n_stack=4)` at the VecEnv level; env itself stays 12-dim | §1.1 — zeroed bbox is non-Markov |
| obs scaling | fixed manual divisors in env, then `clip(-5, 5)`; **no VecNormalize** | §2 |
| action mapping | hover-centered thrust; rates scaled by per-preset `max_rate` | §2 |
| control rate | physics dt 0.01 s, action repeat 2 → 50 Hz policy; 500-step cap (10 s) | §2 |
| termination | see table §3.4; only timeout is `truncated` | §2 |
| reward | potential-based progress + dominant capture bonus, per-term dict always returned | §3 |
| gamma | **0.995** | §4 |
| vec env | `DummyVecEnv`, 16 envs; measure FPS before considering Subproc | §4 |
| curriculum | fixed reward, widening initial-state distribution, 4 levels, promote on success rate > 0.7 over last 100 episodes | §4 |
| no-go list | no domain randomization, no RecurrentPPO, no VecNormalize, no training all 5 presets, no editing zsun's/moterodiaz's files | §0, §2, §4 |

---

## 2. Shared interfaces (WP0 — write these first, everything codes against them)

### 2.1 Physics state — flat `np.ndarray, shape (14,), float64`

```python
# dronegym/stub_physics.py (constants importable by env.py, rewards.py, tests)
POS   = slice(0, 3)    # world position [m]
VEL   = slice(3, 6)    # world velocity [m/s]
QUAT  = slice(6, 10)   # attitude [w, x, y, z], body->world
RATES = slice(10, 13)  # body angular rates [rad/s]
MOTOR = 13             # current collective thrust [N] (motor lag state)
DT    = 0.01           # physics timestep [s]
G     = 9.81
```

### 2.2 Stub physics

```python
def step(state, cmd, cfg):
    """One physics step. cmd = (thrust_cmd_N: float, rate_cmd: np.ndarray(3) rad/s).
    Returns new state array; never mutates input."""
```

Dynamics (simple but honest enough to train against):
- Motor lag: `T += (thrust_cmd - T) * DT / 0.05` (first-order, tau 50 ms), clip to
  `[0, cfg.max_thrust]`.
- Rate tracking (stands in for the inner P loop): `w += (rate_cmd - w) * DT / 0.08`.
- Quaternion integration: `q += 0.5 * q ⊗ [0, w] * DT`, renormalize every step.
- Accel: `a = R @ [0, 0, T] / cfg.mass_kg + [0, 0, -G] - 0.3 * v` (linear drag).
- Euler-integrate velocity then position.
- Reuse `_quat_to_rot` from `camera.py` (import it; single source of truth).

**Swap plan:** env calls `self._physics_step(state, cmd, cfg)`, set in `__init__` to
`stub_physics.step`. When moterodiaz lands `physics.py`, change that one assignment.
If their signature differs, adapt in env, not in their file.

### 2.3 Config shim

```python
# dronegym/config_shim.py
@dataclass
class DroneConfig:
    name: str; mass_g: float; prop_diameter_in: float; motor_kv: float
    battery_v: float; cam_angle_deg: float; frame_size_mm: float
    @property
    def mass_kg(self): return self.mass_g / 1000.0
    @property
    def max_thrust(self): return TWR[self.name] * self.mass_kg * G
    @property
    def max_rate(self): return MAX_RATE[self.name]

def load_config(path) -> DroneConfig   # yaml.safe_load -> DroneConfig(**d)
```

Stub tables (placeholder engineering guesses; moterodiaz's `config.py` will derive
real values — keep `.mass_kg`, `.max_thrust`, `.max_rate` as the only attributes env
touches so the swap is trivial):

| preset | TWR | max_rate rad/s |
|---|---|---|
| tiny_whoop | 2.5 | 12.0 |
| freestyle_5inch | 8.0 | 10.0 |
| cinelifter | 4.0 | 5.0 |
| longrange_7inch | 5.0 | 8.0 |
| longrange_10inch | 4.5 | 6.0 |

Unknown preset name → TWR 4.0, max_rate 8.0.

### 2.4 Env API

```python
class DroneTargetEnv(gymnasium.Env):
    metadata = {"render_modes": []}
    def __init__(self, cfg: DroneConfig, difficulty: int = 0): ...
    def reset(self, *, seed=None, options=None) -> (obs, info)
    def step(action) -> (obs, reward, terminated, truncated, info)
    def set_difficulty(self, level: int)     # clamps to [0, 3]
```

`info` every step: `{"distance": float, "terms": dict, "event": str|None,
"difficulty": int}`. On episode end additionally `{"success": bool}` (True only on
capture).

### 2.5 Rewards API

```python
# dronegym/rewards.py — pure function, no env imports beyond state constants
WEIGHTS = {  # single dict = the whole tuning surface
    "progress": 1.0, "capture": 50.0, "crash": -25.0, "oob": -25.0,
    "tumble": -25.0, "lost": -10.0, "visibility": 0.02, "lost_step": -0.10,
    "smoothness": -0.02, "rate": -0.01, "time": -0.01,
}

def compute_reward(state, prev_state, action, prev_action, target_pos, cfg, event):
    """event in {None,'capture','crash','oob','tumble','lost'}.
    Returns (reward: float, terms: dict[str, float]) — terms has EVERY key in
    WEIGHTS every call (zeros included) so TensorBoard curves never gap."""
```

---

## 3. Work packages

Dependency graph (A/B/C are parallelizable after WP0):

```
WP0 (interfaces: stub_physics.py, config_shim.py, requirements.txt)
 ├─ WP1 [Agent A] env.py + tests/test_env.py
 ├─ WP2 [Agent B] rewards.py + tests/test_rewards.py
 └─ WP3 [Agent C] train.py (against §2.4 spec)
      └─ WP4 integration: wire together, full pytest, gates 1–2
           ├─ WP5 baseline_p.py, gate 3
           └─ WP6 smoke run, gate 4 → iteration runbook §7
```

Sequential fallback (single agent): WP0 → WP1 → WP2 → WP4 → WP5 → WP3 → WP6.

### WP0 — scaffolding (~15 min)

1. `requirements.txt`: `numpy`, `gymnasium`, `stable-baselines3`, `tensorboard`,
   `pyyaml`, `pytest`, `rerun-sdk` (README specs it; zsun needs rerun).
2. `dronegym/stub_physics.py` per §2.1–2.2, with a `__main__` self-check in the style
   of `camera.py`: hover command holds altitude ±0.5 m over 5 s; zero thrust falls;
   pure yaw-rate command keeps gravity body-z component ≈ −1.
3. `dronegym/config_shim.py` per §2.3, `__main__` loads all 5 yamls and prints
   derived values; assert `max_thrust > mass_kg * G` for all.

**Gate 0:** `python -m dronegym.stub_physics` and `python -m dronegym.config_shim`
both pass their asserts.

### WP1 — `dronegym/env.py` (~60 min)

**Observation build** (12-dim, `float32`, `Box(-5, 5, (12,))`):

| idx | value | source | scale |
|---|---|---|---|
| 0–3 | bbox x, y, size, visible | `camera.get_bbox(pos, quat, target_pos, 0.5, cfg.cam_angle_deg)` | none |
| 4–6 | body rates | `state[RATES]` | / 10.0 |
| 7–9 | gravity in body frame | `R.T @ [0, 0, -G]` | / 9.81 → unit |
| 10 | forward velocity | `(R.T @ state[VEL])[0]` | / 15.0 |
| 11 | vertical velocity | `(R.T @ state[VEL])[2]` | / 15.0 |

Then `np.clip(obs, -5, 5)`. Assert `np.isfinite(obs).all()` every step — fail loudly
(debug checklist #1).

**Action mapping** (`Box(-1, 1, (4,))`, clip incoming action first):

```python
hover = cfg.mass_kg * G
thrust_cmd = np.clip(hover + a[0] * (cfg.max_thrust - hover), 0.0, cfg.max_thrust)
rate_cmd   = a[1:4] * cfg.max_rate
```

**Step:** run physics **2×** per env step (action repeat, same cmd), then build obs,
compute event, call `compute_reward`, return. Step counter caps at 500 → `truncated`.

**Termination** (evaluate in this priority order; first hit wins → `event`):

| event | condition | flag |
|---|---|---|
| `capture` | true distance < `0.5 + 0.5` = 1.0 m | terminated |
| `crash` | `pos[2] < 0` | terminated |
| `oob` | `norm(pos) > 50.0` | terminated |
| `tumble` | gravity body-z component > 0 (inverted) | terminated |
| `lost` | not visible for > 25 consecutive steps **after first sighting**; before any sighting the budget is 100 steps (lets level 3 spawn outside FOV and search) | terminated |
| timeout | step 500 | **truncated** |

Use **privileged true distance** for capture/reward; bbox only for obs.

**Reset / curriculum.** Drone spawns at `[0, 0, z]`, `z ~ U(8, 12)`; yaw uniform
`[-π, π)` at every level. Target is placed **along the camera axis** (not body x —
debug checklist #2: uptilt would otherwise push it out of frame at level 0). Sample
desired image coords `(bx, by)` per the table, back-project to a world direction:

```python
half_tan = tan(radians(120) / 2)
dir_body = normalize(cam_fwd + bx * half_tan * cam_right + by * half_tan * cam_up)
target_pos = drone_pos + R @ dir_body * d          # then clip target z to >= 1.0
```

where `cam_fwd/right/up` are the camera basis vectors from `camera.py` lines 56–59.

| level | distance d | (bx, by) sampled from | roll/pitch | initial speed |
|---|---|---|---|---|
| 0 | 5 | (0, 0) exactly | level | 0 |
| 1 | U(5, 15) | U(−0.5, 0.5)² | ±10° | 0 |
| 2 | U(5, 25) | U(−0.9, 0.9)² | ±20° | ≤ 3 m/s random dir |
| 3 | U(5, 30) | 70%: U(−0.9, 0.9)²; 30%: target placed at a uniform bearing within ±90° yaw of camera axis (may be outside FOV) | ±30° | ≤ 5 m/s |

Seeding: gymnasium's `self.np_random` throughout; `reset(seed=...)` plumbs through.

`__main__` self-check: construct on `freestyle_5inch`, print the level-0 reset obs
and **assert `obs[3] == 1` (visible at spawn)**; run 1,000 random steps asserting
finite obs and printing steps/sec.

**Tests (`tests/test_env.py`):**
- reset: obs shape (12,), dtype float32, finite, `obs[3] == 1.0` at every level 0–2
  over 50 seeded resets.
- action `a=0` → thrust_cmd ≈ hover (inspect via mapping helper, expose as method).
- ground crash → `terminated=True, truncated=False`, event `crash`; 500 no-op steps →
  `truncated=True`.
- teleport-style test: build env, force `state[POS]` next to target → step →
  `capture`, `info["success"] is True`.
- obs scaling: inject extreme state (30 m/s, 20 rad/s) → all |obs| ≤ 5.
- determinism: same seed → identical first obs and identical obs after 10 fixed
  actions.

### WP2 — `dronegym/rewards.py` (~30 min)

Implement §2.5 exactly. Term formulas:

| term | formula |
|---|---|
| progress | `d_prev − d_now` (true distances drone→target) |
| capture / crash / oob / tumble / lost | weight fires once, on that event |
| visibility | `1 − norm(bbox_xy) / √2` if visible else 0 |
| lost_step | 1 if not visible |
| smoothness | `‖a − a_prev‖²` |
| rate | `‖state[RATES] / 10‖²` |
| time | 1 per step |

`terms` dict returns **weighted** contributions (what actually entered the sum), all
keys always present.

**Tests (`tests/test_rewards.py`):**
- approaching target → progress term > 0; receding → < 0; sum of progress over a
  synthetic straight-line trajectory ≈ distance closed (potential-based property).
- each event fires exactly its bonus/penalty term.
- centered visible bbox → visibility term = weight; invisible → 0 and lost_step fires.
- **anti-hacking arithmetic** (encodes RL_NOTES §3 failure modes as regression
  tests): `WEIGHTS["visibility"] * 500 < 0.5 * WEIGHTS["capture"]` (no hover-and-
  stare); `500 * (|time| + |lost_step|·0.2 + typical smoothness+rate ≈ 0.01) <
  |WEIGHTS["crash"]|` is **not required** — instead assert
  `500 * |WEIGHTS["time"]| < |WEIGHTS["crash"]|` (no suicide incentive from time
  alone) and `|WEIGHTS["oob"]| == |WEIGHTS["crash"]|` (no cheap exit).
- all keys present in `terms` on a no-event step.

### WP3 — `train.py` (~60 min)

CLI: `python train.py --config configs/freestyle_5inch.yaml --steps 2_000_000
--run-name smoke1 [--difficulty N]` (fixed difficulty disables the curriculum
callback — needed for debugging).

**Env factory — the single source of truth for the wrapper stack** (zsun's
`evaluate.py` must import this; that's the handoff):

```python
def make_vec_env(cfg_path, n_envs=16, difficulty=0, seed=0):
    cfg = load_config(cfg_path)
    def thunk(i):
        def _f():
            env = DroneTargetEnv(cfg, difficulty)
            env.reset(seed=seed + i)
            return Monitor(env, info_keywords=("success",))
        return _f
    venv = DummyVecEnv([thunk(i) for i in range(n_envs)])
    return VecFrameStack(venv, n_stack=4)   # policy sees 48 dims
```

**PPO:**

```python
PPO("MlpPolicy", venv, n_steps=512, batch_size=2048, n_epochs=10,
    learning_rate=3e-4, gamma=0.995, gae_lambda=0.95, clip_range=0.2,
    ent_coef=0.0, vf_coef=0.5, max_grad_norm=0.5,
    policy_kwargs=dict(net_arch=[64, 64], activation_fn=torch.nn.Tanh),
    tensorboard_log="runs/", seed=0)
```

**CurriculumCallback** (`BaseCallback`): in `_on_step`, scan `self.locals["infos"]`
for completed episodes (`"episode"` key); push `info["success"]` into a
`deque(maxlen=100)`. When `len == 100` and mean > 0.7 and level < 3: level += 1,
`self.training_env.env_method("set_difficulty", level)` **and the same on the eval
env**, clear the deque. Every rollout: `self.logger.record("curriculum/level", level)`
and `curriculum/success_rate`.

**RewardTermsCallback:** on episode end, `info["terms"]` (env accumulates weighted
term sums over the episode and attaches at end); log mean of each as
`reward_terms/<name>`. This is the §3 debugging surface — non-negotiable.

**Checkpointing:** `CheckpointCallback(save_freq=100_000 // 16, save_path=
f"runs/{run_name}/ckpt")` + `EvalCallback` (separate `make_vec_env(..., n_envs=4,
seed=10_000)`, `eval_freq=50_000 // 16`, `n_eval_episodes=20`, `deterministic=True`,
`best_model_save_path=f"runs/{run_name}"`).

Print FPS every 10k steps (SB3 logs `time/fps` — surface it in stdout).

### WP4 — integration (~20 min)

Wire everything, run `pytest -q` clean, then the two gates:

- **Gate 1:** `python -m dronegym.env` — 1,000 random steps, finite, prints
  steps/sec (expect ≥ 10,000/s single env on stub physics).
- **Gate 2:** `pytest -q` fully green.

### WP5 — `baseline_p.py` (~20 min, do NOT skip — RL_NOTES §7)

Hand-scripted proportional policy through the same env (raw, unstacked):
`yaw_rate = -1.5 * bbox_x`, `pitch_rate = -1.0 * bbox_y` (drives target toward image
center; signs verified against camera.py conventions in a 3-line comment),
`thrust = 0.1 * bbox_y` correction around hover (`a[0] = 0.1 * bbox_y`), and if not
visible: yaw at 0.5 to search. Run 100 episodes at each difficulty, print success
rate per level.

- **Gate 3:** level 0 success ≥ 60%. If the P-controller can't reach a target 5 m
  dead ahead, **the env is broken, not the controller** — fix before any PPO run.
  This number is also the baseline PPO must beat.

### WP6 — smoke run + iteration (rest of budget)

- **Gate 4 — 200k-step smoke run** on `freestyle_5inch` (~10 min wall): pass =
  reward moving at all, no NaNs, `rollout/ep_len_mean` not collapsed to < 50,
  FPS ≥ 2,000. Not looking for competence.
- Then reward/curriculum iteration per runbook §7, `freestyle_5inch` only.
- Final runs: `freestyle_5inch` + `tiny_whoop` + `cinelifter` in parallel OS
  processes. For `tiny_whoop` raise `WEIGHTS["smoothness"]` 2–4× (27 g + 22000 KV =
  jitter machine).

---

## 4. Verification gates (summary)

| gate | command | pass |
|---|---|---|
| 0 | `python -m dronegym.stub_physics` / `config_shim` | asserts pass |
| 1 | `python -m dronegym.env` | 1k finite random steps, ≥ 10k steps/s |
| 2 | `pytest -q` | green |
| 3 | `python baseline_p.py` | level-0 success ≥ 60% |
| 4 | 200k smoke run | reward moves, no NaN, ep_len > 50, FPS ≥ 2k |
| 5 | full run | curriculum promotes past level 1; `best_model.zip` beats baseline success rate |

---

## 5. Environment setup (Windows)

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
pytest -q          # after WP4
```

Agents: activate the venv for every command; `torch` arrives with
stable-baselines3 (CPU build is fine — nets are [64,64]).

---

## 6. Coordination — messages to send, files not to touch

**Do not edit:** `camera.py`, `configs/*.yaml`, `physics.py`/`config.py` when they
appear. Propose diffs to owners instead.

**To zsun (send at integration #1):**
1. `evaluate.py` must build the env via `train.make_vec_env(...)` (or replicate
   `VecFrameStack(n_stack=4)` exactly) — a checkpoint loaded without the stack gets
   48-vs-12 dim mismatch or silent garbage. #1 integration landmine.
2. Requested `camera.py` change (owner's call): treat target visible while the
   sphere overlaps the frame — e.g. `abs(bbox_x) > 1.0 + bbox_size/2` — or clip
   center coords to [-1,1] keeping `visible=1`. Current center-only test blanks the
   bbox at point-blank range (worst moment for an obs discontinuity).

**To moterodiaz:**
1. Confirm `physics.step(state, cmd, cfg)` signature and that `cmd` is
   `(thrust_N, rate_cmd_rad_s)` — env adapts if not, just tell us.
2. Ask that `config.py`'s `DroneConfig` expose `mass_kg`, `max_thrust`, `max_rate`
   (or tell us the real names) so the shim swap is mechanical.
3. FYI: stub assumes motor tau 50 ms, rate-loop tau 80 ms, linear drag 0.3 — if the
   real dynamics are wildly different, retraining is expected at integration #1.

---

## 7. Training iteration runbook (post-Gate-4)

One variable per run. Read `reward_terms/*` curves before touching anything.

| symptom | diagnosis | fix |
|---|---|---|
| NaN in first 1k steps | physics divergence / quat normalization | assert-finite already in env; check stub integration step |
| reward flat at floor | target not visible at spawn | run env `__main__`; check level-0 bbox print |
| ep_len collapses to ~20 | suicide-to-stop-bleeding | per-step penalty sum vs crash penalty arithmetic |
| visibility term ↑, progress flat, success 0 | hover-and-stare | raise capture bonus or time penalty; check capture radius reachable |
| reward ↑ but flight looks worse | a term is being gamed | per-term curves point at it in 30 s |
| violent jitter | smoothness under-weighted | raise `WEIGHTS["smoothness"]` (esp. tiny_whoop) |
| learns L0–1, dies at L2–3 | curriculum step too big | widen the *previous* level's ranges instead of weakening reward |
| success oscillates at promotion boundary | window too small | widen deque to 200, threshold stays 0.7 |

TensorBoard: `tensorboard --logdir runs/`. Watch `curriculum/level`,
`curriculum/success_rate`, `reward_terms/*`, `rollout/ep_len_mean`, `time/fps`.

---

## 8. Deliverables checklist (hand to zsun)

- [ ] `runs/<preset>/best_model.zip` for freestyle_5inch (+ tiny_whoop, cinelifter
      if time)
- [ ] Wrapper stack: `make_vec_env` importable from `train.py`, `n_stack=4`
      documented in its docstring
- [ ] No VecNormalize stats files — by design, nothing else to ship
- [ ] `baseline_p.py` success rates per level (fallback demo + bragging number)

---

## 9. Decisions made where RL_NOTES was silent (audit me)

| decision | value | rationale |
|---|---|---|
| state layout | flat (14,) array with named slices | cheapest thing physics/env/tests can share |
| lost-target terminal penalty | −10 | worse than nothing, better than crashing (−25); keeps "give up" < "fly into ground" |
| never-seen grace | 100 steps (2 s) before first sighting | level 3 spawns target outside FOV; 25-step budget would kill the episode before a 180° yaw completes |
| arena radius | 50 m | max spawn distance 30 m + room to overshoot |
| drone spawn | [0, 0, U(8, 12)] m | room below for dives, far from OOB sphere |
| target z clamp | ≥ 1.0 m | capture radius 1.0 — target on the ground would force ground contact |
| target placement | back-projected from sampled image coords along camera axis | guarantees the level-0/1/2 visibility contract under any uptilt |
| capture radius | 1.0 m (= target_radius 0.5 + 0.5) | RL_NOTES formula, target_radius pinned to camera.py's example value |
| stub TWR / max_rate tables | §2.3 | placeholder until config.py; only 3 attrs touched so swap is one file |
| eval env difficulty | synced on promotion by the curriculum callback | eval at stale difficulty makes best_model selection meaningless |
| Monitor placement | per-env, inside DummyVecEnv, before VecFrameStack | Monitor is an env-level wrapper; also needed for success-rate infos |
