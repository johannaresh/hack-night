# RL layer — design + execution plan (johannaresh)

Research-grounded plan for `dronegym/env.py`, `dronegym/rewards.py`, `train.py`.
Numbered refs at the bottom.

---

## 0. TL;DR of the research

The task — fly a quadrotor at a target using a bbox as the only exteroceptive input,
commanding collective thrust + body rates — sits on top of a well-established
literature. Five findings change what I'd write:

1. **CTBR is the right action space.** The team already picked
   `[thrust, roll_rate, pitch_rate, yaw_rate]`. Kaufmann's benchmark found policies
   commanding body-rates + thrust transfer far more robustly than policies commanding
   individual rotor thrusts, and hit >45 km/h on real hardware [2]. Same action space
   later carried Swift to champion-level drone racing [3]. No change needed — but know
   that this is the validated choice, not an arbitrary one.
2. **Progress reward, not distance penalty.** Potential-based shaping
   (`r = Φ(s') − Φ(s)` with `Φ = −distance`) provably leaves the optimal policy
   unchanged [4]. A per-step `−distance` penalty does *not* have that guarantee and
   is where most "my drone learned to hover and stare" bugs come from.
3. **Action-difference regularization is not optional.** Penalizing `‖a_t − a_{t−1}‖²`
   (CAPS) eliminated high-frequency oscillation on a real quad and cut power draw ~80%
   [5]; SimpleFlight independently lists it as one of five factors that matter for
   quadrotor PPO [1]. Cheap to add, visibly better demo footage.
4. **Curriculum by expanding the initial-state distribution.** Keep the reward function
   fixed and widen where the drone/target spawn [23]. Reverse-curriculum theory backs
   starting near the goal and receding [14]; continuous/decaying schedules beat static
   ones on reaching tasks [15].
5. **Skip domain randomization.** Every sim-to-real paper pushes it [1][19][20], but
   there is no real drone here — the sim *is* the target. DR would only slow
   convergence. Ignore that whole branch of the literature.

---

## 1. Contract issues to raise with the team NOW (before 0:30)

These are cheap to fix at 0:30 and expensive at 3:00.

### 1.1 A lost target produces an all-zero observation → the state is not Markov

`camera.py` returns `np.zeros(4)` when the target is behind the camera *or* its center
leaves the frame. So obs[0:4] is identical whether the target just drifted off the left
edge or the right edge. **From a single frame the agent cannot know which way to turn
to recover.** Single-frame policies will never learn search-and-reacquire.

Fix, cheapest first:

- **`VecFrameStack(n_stack=4)`** — the wrapper sees the bbox drifting toward an edge
  before it vanished, which encodes the recovery direction. Env still returns 12 dims,
  so the interface contract is untouched. **This is my recommendation.**
- Latch last-seen bbox + an "age" counter in the env (changes the obs contract).
- `RecurrentPPO` from sb3-contrib (slower to train, more knobs — not in 6 hours).

Memory under partial observability is exactly what the DR theory work flags as
necessary rather than optional [19], and the privileged-information UAV navigation work
makes the same architectural point [12].

> **Tell zsun:** `evaluate.py` must apply the *same* frame-stacking wrapper, or the
> checkpoint will load and fly like garbage. This is the #1 integration landmine.

### 1.2 Obs has forward + vertical velocity but no lateral velocity

Indices 10–11 are forward and vertical body-frame velocity. Sideways drift is
unobservable, so the agent cannot damp lateral oscillation — it will wobble side to
side on approach. Either ask for a 13th dim (lateral velocity), or accept it and rely
on frame stacking to let the policy infer drift from bbox motion. Frame stacking
covers it; flag it anyway so nobody is surprised by the wobble.

### 1.3 `camera.py` blanks the target exactly when it matters most

`get_bbox` returns all zeros when `|bbox_x| > 1 or |bbox_y| > 1` — testing the
**sphere's center**. At close range the sphere can fill most of the frame with its
center just outside the edge → target reads as invisible right at the moment of
capture. Expect visibility to flicker on final approach, which is the worst possible
place for a discontinuity in both obs and reward.

Ask zsun for one of: a margin (`> 1.2`), a partial-visibility test using
`bbox_size`, or clipping `bbox_x/y` to `[-1, 1]` while keeping `visible = 1` as long as
the sphere overlaps the frame. Maintaining features inside the FOV is a whole
sub-literature for aggressive quadrotor flight [17][18] — worth 5 minutes of their time.

### 1.4 Camera uptilt couples "go forward" to "target rises in frame"

With `cam_angle_deg` of 30° on the 5", pitching down to accelerate forward pushes the
target *up* in the image. The agent must learn this coupling, and it differs per preset
(10° cinelifter vs 30° freestyle). Don't be alarmed when a policy trained on one preset
looks incompetent on another — train per preset.

### 1.5 I need a physics stub at 0:30, not at 2:00

Per the timeline I train on stub physics for the first 90 minutes. I'll write
`_stub_physics()` inside my own test file matching the agreed state format —
position(3), velocity(3), quat(4), rates(3), motor(1) — so I never block on
moterodiaz, and swapping in the real `physics.step` is a one-line change.

---

## 2. `dronegym/env.py`

```
class DroneTargetEnv(gymnasium.Env)
    __init__(config: DroneConfig, difficulty: int = 0, seed=None)
    reset(seed, options) -> (obs, info)
    step(action) -> (obs, reward, terminated, truncated, info)
    set_difficulty(level: int)      # curriculum hook, called by callback
```

### Control rate: 50 Hz policy, 100 Hz physics

Run physics at `dt = 0.01` but let the policy act every **2** physics steps
(action repeat). Two reasons: real FPV flight controllers run rate loops far above the
outer control rate, and — more importantly — a 100 Hz policy makes the discount horizon
problem twice as bad (see §4). Episode cap 10 s = **500 policy steps**.

### Observation scaling — do it in the env, not with `VecNormalize`

The 12 dims have wildly different scales: bbox terms are already `[-1, 1]`, body rates
reach ±20 rad/s, velocities ±30 m/s. Unscaled, the network sees the bbox as noise.

Use **fixed manual divisors in `env.py`**, not `VecNormalize`:

| idx | raw | scaling |
|---|---|---|
| 0–1 | bbox x, y | none (already ±1) |
| 2 | bbox size | none (already ~±1) |
| 3 | visible | none (0/1) |
| 4–6 | body rates rad/s | `/ 10.0` |
| 7–9 | gravity in body frame | `/ 9.81` (→ unit vector) |
| 10–11 | vel fwd, vert m/s | `/ 15.0` |

Then `clip(obs, -5, 5)`.

**Why not `VecNormalize`:** its running statistics are a separate artifact that must be
saved alongside the checkpoint and reloaded identically in `evaluate.py`. With five
presets that's five stat files for zsun to keep straight, and silently loading the wrong
one produces a policy that looks broken with no error. Fixed scaling is deterministic,
config-independent, and needs nothing from anyone else. Normalized observations are
standard practice in the drone-gym ecosystem [26]; this just pins the constants.

Note the gravity vector (7–9) is a good attitude encoding — no quaternion sign
discontinuity, no gimbal singularity. It does drop absolute yaw, which is fine because
the task is entirely bbox-relative. SimpleFlight found the full rotation matrix in the
actor input mattered [1]; the gravity vector is the 2-DOF subset that the task needs.

### Action mapping — center the hover point

Naive `thrust = ((a₀+1)/2) · max_thrust` puts hover at `a₀ = 2/TWR − 1`, i.e. −0.5 for
a TWR-4 build and −0.75 for a TWR-8 one. A freshly initialized Gaussian policy outputs
≈0 mean, so it starts by commanding 2–4× hover thrust and rockets upward for the first
several thousand steps.

Instead make `a₀ = 0` mean hover:

```python
hover = cfg.mass_kg * 9.81
thrust = np.clip(hover + a[0] * (cfg.max_thrust - hover), 0.0, cfg.max_thrust)
rates  = a[1:4] * MAX_RATE          # MAX_RATE ~ 10 rad/s, tune per preset
```

Rate limits should probably scale with the preset — a tiny whoop at 22000 KV has
enormous rate authority; a 2.2 kg cinelifter does not. Coordinate with moterodiaz on
whether `MAX_RATE` derives from `DroneConfig`.

### Termination — `terminated` vs `truncated` matters

Get this wrong and value bootstrapping is wrong at every episode end.

| condition | flag | detail |
|---|---|---|
| capture | `terminated` | true distance < `target_radius + 0.5` |
| ground | `terminated` | `z < 0` |
| out of arena | `terminated` | `‖pos‖ > arena_radius` |
| tumbled | `terminated` | body-frame gravity z-component > 0 (inverted) |
| target lost too long | `terminated` | `visible == 0` for > 25 consecutive steps (0.5 s) |
| time limit | **`truncated`** | 500 steps |

Only the time limit is `truncated`. SB3 bootstraps the value function correctly across
truncation but not termination, so mislabeling silently biases learning.

The lost-target grace period is a deliberate middle ground: terminating instantly on
loss gives a sharp signal but forbids ever learning reacquisition; never terminating
lets the agent waste whole episodes flying blind. A recent study specifically on
quadrotor RL found termination-condition choice materially changes convergence and
final performance, and that badly chosen ones cause premature saturation [24] — so this
is a knob worth one deliberate experiment, not a guess left untouched.

Use **privileged true distance** for termination and reward, never `bbox_size`.
`bbox_size ∝ 1/depth`, so reward built on it has exploding gradients exactly at
capture range. Bbox is for the *observation* (it's the CV stand-in); ground truth is
fine for reward and termination since both are training-time only. This actor-sees-
pixels / critic-sees-truth asymmetry is standard [12].

---

## 3. `dronegym/rewards.py`

Isolated file, per the README, so it can be iterated without touching env internals.

```python
def compute_reward(state, prev_state, action, prev_action, target_pos, cfg, event):
    """-> (reward: float, terms: dict)   # terms for tensorboard, always"""
```

**Always return the per-term breakdown and log every term separately to TensorBoard.**
When reward goes up but flight looks worse, the per-term curves tell you which term got
gamed in about thirty seconds. This is the single highest-leverage thing in the file.

### Terms and starting weights

| term | formula | weight | ≈ per-episode contribution |
|---|---|---|---|
| progress | `d_prev − d_now` | `1.0` | +20 (closing 20 m) |
| capture bonus | on capture, terminate | `+50` | +50 |
| crash / OOB | on crash, terminate | `−25` | −25 |
| visibility | `1 − ‖bbox_xy‖₂/√2` if visible | `+0.02/step` | up to +10 |
| lost penalty | `1` if not visible | `−0.10/step` | 0 when flown well |
| action smoothness | `‖a_t − a_{t−1}‖²` | `−0.02` | −1 to −3 |
| rate penalty | `‖ω/10‖²` | `−0.01` | −1 to −2 |
| time | `1` per step | `−0.01/step` | −5 |

Good episode ≈ **+70**. Crash ≈ **−25**. Clean separation, and no single shaping term
can outweigh the sparse capture bonus.

**The sizing principle matters more than these exact numbers:** keep every shaping
term's per-episode contribution within about one order of magnitude of the others, and
keep the terminal bonus dominant. Re-check the arithmetic whenever you retune, because
the failure modes below are all arithmetic failures.

### Reward hacking to expect — in likelihood order

1. **Hover and admire the target.** Visibility pays per step; capture ends the episode
   and stops the income. If `0.02 × 500 = +10` ever approaches the capture bonus, the
   optimal policy is to park at range and stare. Defenses: dominant capture bonus,
   the time penalty, and progress-as-potential [4]. *Symptom:* visibility term climbs,
   progress term flat, success rate stuck near zero.
2. **Suicide to stop the bleeding.** If per-step penalties sum below the crash penalty,
   crashing early beats flying. Keep `Σ` step penalties (≈ −8) well under `−25`.
   Xue et al. used an explicit per-step alive bonus for exactly this [9]. *Symptom:*
   episode length collapses to a few dozen steps.
3. **Fly away from the arena boundary.** If OOB is penalized less than crashing, the
   agent finds the cheapest exit. Keep them comparable.
4. **Rate-penalty paralysis.** Over-weight the rate term and the policy refuses to
   maneuver, freezing level and never turning. *Symptom:* rate penalty near zero,
   progress near zero.

The reward-mechanism study behind PPO's success on drone traversal makes the same
point about continuous vs sparse return design [8], and comparative work on reward
choice for hover shows performance swings substantially with these weights [10].

### Yaw

Nothing in the reward mentions yaw. The task is bbox-relative and yaw is unobservable
from the gravity vector, so leave it unconstrained — but expect slow yaw drift in the
demo footage. If it looks bad, a tiny `−‖ω_yaw‖²` term fixes it; don't add it
pre-emptively.

---

## 4. `train.py`

### `gamma` is the one hyperparameter people get wrong here

Effective horizon ≈ `1/(1−γ)` **steps**, not seconds. At the SB3 default `γ = 0.99`
and 50 Hz control that's 100 steps = **2 seconds** — shorter than the flight itself, so
the agent is structurally blind to whether it ever reaches the target.

Use **`γ = 0.995`** (200 steps = 4 s) at 50 Hz. If you move to 100 Hz control, `0.997`.
This alone can be the difference between "PPO doesn't work on this" and a policy that
converges.

### Starting hyperparameters

```python
PPO("MlpPolicy", venv,
    n_steps=512,           # × 16 envs = 8192-step rollout
    batch_size=2048,       # large batches: SimpleFlight factor #5 [1]
    n_epochs=10,
    learning_rate=3e-4,
    gamma=0.995,           # see above
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.0,
    vf_coef=0.5,
    max_grad_norm=0.5,
    policy_kwargs=dict(net_arch=[64, 64], activation_fn=nn.Tanh),
    tensorboard_log="runs/")
```

`[64, 64]` + tanh is what the quadrotor RL literature converges on [22], and matching
published architecture removes one variable when debugging. Large batch size is a
deliberate deviation from the SB3 default of 64 — SimpleFlight found it one of five
things that actually mattered [1]. Published quadrotor PPO setups cluster around
`lr` 2–3e-4, `n_steps` 2048–4096, `batch` 128, `n_epochs` 10–12 [22][25].

### `DummyVecEnv`, not `SubprocVecEnv`

Physics is pure NumPy and a step costs microseconds. `SubprocVecEnv`'s
pickle-and-pipe overhead per step will likely exceed the physics cost outright. Start
`DummyVecEnv` with 16 envs, **measure FPS in the first 30 seconds of the smoke run**,
and only try Subproc if FPS is disappointing. Don't spend hackathon time on this —
just don't assume subprocesses are free.

### Curriculum

Fixed reward, widening initial-state distribution [23]. Implement as
`env.set_difficulty(level)` plus a callback that promotes on **success rate**, not on
reward plateau — the published criterion of "cumulative reward stable for 1M steps"
[22] is far too slow for a 6-hour budget.

| level | target distance | target bearing | initial attitude | initial velocity |
|---|---|---|---|---|
| 0 | 5 m | dead ahead, centered | level | zero |
| 1 | 5–15 m | within inner 50% of FOV | roll/pitch ±10° | zero |
| 2 | 5–25 m | anywhere in FOV | roll/pitch ±20° | ±3 m/s |
| 3 | 5–30 m | may start **outside** FOV | roll/pitch ±30° | ±5 m/s |

Promote when rolling success rate over the last 100 episodes > **0.7**. Log the current
level to TensorBoard so you can see promotions against the reward curve. Level 3 is
what forces search-and-reacquire, and it only works if §1.1 is fixed.

If a level stalls, widen the *previous* level rather than weakening the reward — that's
the reverse-curriculum insight [14], and a decaying/continuous schedule beats a static
one on reaching tasks [15]. Staged transfer also failed in the curriculum-navigation
study when stages were too far apart [16], so prefer more, smaller steps.

### Checkpointing

`CheckpointCallback` every 100k steps + `EvalCallback` on a separate deterministic env
tracking `best_model`. **Hand zsun `best_model.zip` per preset, plus the exact wrapper
stack**, since `evaluate.py` and `viz/replay.py` are theirs. No `VecNormalize` file to
pass along, by design (§2).

### Budget

Pure-NumPy physics × 16 envs should give thousands of steps/s, so ~2 h of wall clock is
several million steps per preset. Plan:

- **2:00–2:15** — 200k-step smoke run. Not looking for competence, looking for: reward
  moving at all, no NaNs, episode length not collapsing, FPS acceptable.
- **2:15–4:30** — reward/curriculum iteration on `freestyle_5inch` only. One variable
  per run. This is where the whole task is won or lost.
- **4:30–5:30** — final runs on 2–3 presets in parallel processes.

Do not train all five presets. `freestyle_5inch` is the money shot; `tiny_whoop` and
`cinelifter` are the "look how configurable it is" contrast. The 22000 KV whoop has
extreme control authority relative to its 27 g mass and will be the jitteriest — it
needs the smoothness weight raised, possibly 2–4×, and is the most likely to look bad
on stage.

---

## 5. Debug checklist, in the order symptoms actually appear

1. **NaNs in obs within the first 1k steps** → physics divergence or a bad quaternion
   normalization. Assert finite in `step()` from the very first version; fail loudly.
2. **Reward flat at the floor** → check `visible` is ever 1 at reset. If level 0 spawns
   the target outside the FOV because of `cam_angle_deg` uptilt (§1.4), the agent never
   sees anything. Print the level-0 reset bbox before training anything.
3. **Episode length collapses to ~20 steps** → hack #2 above.
4. **Reward rises, flight looks worse** → read per-term TensorBoard curves; something
   is being gamed.
5. **Policy jitters violently** → raise smoothness weight; that's what it's for
   [5][6][7].
6. **Learns level 0–1, dies at level 2–3** → curriculum step too large, or §1.1 not
   fixed.

## 6. Stretch, only if genuinely ahead

- **Asymmetric actor-critic:** critic gets true relative target position, actor keeps
  only the bbox. Directly the privileged-information approach [12], and a clean fit
  since the "fake CV" already *is* privileged info. Needs a custom SB3 policy — real
  work, hence stretch.
- **Moving targets.** The vision-based tracking work handles trajectory-following
  targets and occlusion [11][13]. Great demo, big scope increase.
- **Multiple sequential targets** → a racing gate course [3][21].

## 7. Fallback

The README's hand-scripted proportional policy (steer to center the bbox, throttle
toward it) is the right insurance. **Write it at 2:00, not at 5:00** — it takes ten
minutes, it validates the env independently of PPO, and if the env is subtly broken the
P-controller failing is a much faster diagnosis than a PPO run that just doesn't
improve. It also gives a baseline number to beat.

---

## References

[1] [What Matters in Learning a Zero-Shot Sim-to-Real RL Policy for Quadrotor Control? A Comprehensive Study](https://consensus.app/papers/details/80765d16b2345d59ba91efb85e7ae0ca/?utm_source=claude_code) (Jiayu Chen et al., 2024, IEEE RA-L, 17 citations)
[2] [A Benchmark Comparison of Learned Control Policies for Agile Quadrotor Flight](https://consensus.app/papers/details/c81fd3dfd3be5a1781018d6a4129dec9/?utm_source=claude_code) (Elia Kaufmann et al., 2022, ICRA, 116 citations)
[3] [Champion-level drone racing using deep reinforcement learning](https://consensus.app/papers/details/bc1d6597c2f25a59a04dab2b480cb9e8/?utm_source=claude_code) (Elia Kaufmann et al., 2023, Nature, 829 citations)
[4] [Policy Invariance Under Reward Transformations: Theory and Application to Reward Shaping](https://consensus.app/papers/details/b926892c823552b8bbcfac89c5940204/?utm_source=claude_code) (A. Ng et al., 1999, 2947 citations)
[5] [Regularizing Action Policies for Smooth Control with Reinforcement Learning](https://consensus.app/papers/details/749cde5a768358db85d5a6120bd7ae7d/?utm_source=claude_code) (Siddharth Mysore et al., 2020, ICRA, 118 citations)
[6] [Benchmarking Smoothness and Reducing High-Frequency Oscillations in Continuous Control Policies](https://consensus.app/papers/details/12c2e47cdf6f5942bb9772c12010dbf6/?utm_source=claude_code) (Guilherme Christmann et al., 2024, IROS, 8 citations)
[7] [Gradient-based Regularization for Action Smoothness in Robotic Control with Reinforcement Learning](https://consensus.app/papers/details/a6126062423353a68c4e9058e873ebfc/?utm_source=claude_code) (I. Lee et al., 2024, IROS, 9 citations)
[8] [Application of Reinforcement Learning in Controlling Quadrotor UAV Flight Actions](https://consensus.app/papers/details/0bd9c50f6bf05eff8c4216b09e603fdd/?utm_source=claude_code) (Shang-En Shen et al., 2024, Drones, 13 citations)
[9] [Robust Wind-Resistant Hovering Control of Quadrotor UAVs Using Deep Reinforcement Learning](https://consensus.app/papers/details/82f250d790d75519947a71c8588253c4/?utm_source=claude_code) (J. Xue et al., 2023, IEEE T-IV, 19 citations)
[10] [Quadrotor Motion Control Using Deep Reinforcement Learning](https://consensus.app/papers/details/c1197eaac5905dc798a4fad20f518b3c/?utm_source=claude_code) (Zifei Jiang et al., 2021, J. Unmanned Vehicle Systems, 20 citations)
[11] [A Vision-Based End-to-End Reinforcement Learning Framework for Drone Target Tracking](https://consensus.app/papers/details/4d2e13622f245deb95a9f4050c03ab6a/?utm_source=claude_code) (Xunyi Zhao et al., 2024, Drones, 9 citations)
[12] [Vision-Based Deep Reinforcement Learning of UAV Autonomous Navigation Using Privileged Information](https://consensus.app/papers/details/0ff0d05b1c3f571f8f5c4c6b9682dc02/?utm_source=claude_code) (Junqiao Wang et al., 2024, Drones, 26 citations)
[13] [Coarse-to-Fine UAV Target Tracking With Deep Reinforcement Learning](https://consensus.app/papers/details/20c2d0edddea5114b6dee21b5d4aabb2/?utm_source=claude_code) (Wei Zhang et al., 2019, IEEE T-ASE, 129 citations)
[14] [Reverse Curriculum Generation for Reinforcement Learning](https://consensus.app/papers/details/b6cee1a535665f659a1f77f64a32cfef/?utm_source=claude_code) (Carlos Florensa et al., 2017, ArXiv, 527 citations)
[15] [Accelerating Reinforcement Learning for Reaching Using Continuous Curriculum Learning](https://consensus.app/papers/details/08ba366d316254f893464628abfa3b8a/?utm_source=claude_code) (Shan Luo et al., 2020, IJCNN, 61 citations)
[16] [Curriculum Reinforcement Learning From Avoiding Collisions to Navigating Among Movable Obstacles in Diverse Environments](https://consensus.app/papers/details/4ecce6b24df0570aadd53b88fb006845/?utm_source=claude_code) (Hsueh-Cheng Wang et al., 2023, IEEE RA-L, 37 citations)
[17] [Perception-Aware Image-Based Visual Servoing of Aggressive Quadrotor UAVs](https://consensus.app/papers/details/5788e933a1645b41b01991e5dfe404ed/?utm_source=claude_code) (Chao Qin et al., 2023, IEEE/ASME T-Mech, 25 citations)
[18] [SVPTO: Safe Visibility-Guided Perception-Aware Trajectory Optimization for Aerial Tracking](https://consensus.app/papers/details/2c9c4e65e05c53e392e3ae0f04603aa6/?utm_source=claude_code) (Hanzhang Wang et al., 2024, IEEE T-IE, 20 citations)
[19] [Understanding Domain Randomization for Sim-to-real Transfer](https://consensus.app/papers/details/fd1efdbc97bf5580aed2050eb7aec37d/?utm_source=claude_code) (Xiaoyu Chen et al., 2021, ArXiv, 168 citations)
[20] [Sim-to-Real Transfer in Deep Reinforcement Learning for Robotics: a Survey](https://consensus.app/papers/details/ee38f69cd99c55fbaa419c07d0bb316b/?utm_source=claude_code) (Wenshuai Zhao et al., 2020, IEEE SSCI, 1021 citations)
[21] [Dashing for the Golden Snitch: Multi-Drone Time-Optimal Motion Planning with Multi-Agent Reinforcement Learning](https://consensus.app/papers/details/bd26480f7b45539b956bfe42d8abd3f5/?utm_source=claude_code) (Xian Wang et al., 2024, ICRA, 9 citations)

Non-Consensus sources:

[22] [Curriculum-based Sample Efficient Reinforcement Learning for Robust Stabilization of a Quadrotor](https://arxiv.org/html/2501.18490v1) — concrete 3-stage curriculum, reward `R = 25 − 20Tₑ − 100E + 20S − 18wₑ`, ±40° tumble termination, 5 s episodes, `[64,64]` tanh
[23] [Learning to Fly in Seconds](https://arxiv.org/abs/2311.13081) — curriculum that holds the reward fixed and expands the initialization domain
[24] [A Heuristic Approach for Performance Tuning in RL-based Quadrotor Control via Reward Design and Termination Conditions](https://arxiv.org/pdf/2605.19166) — termination conditions materially affect convergence; poor ones cause premature saturation
[25] [PPO — Stable Baselines3 documentation](https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html) — defaults: `lr=3e-4, n_steps=2048, batch_size=64, n_epochs=10, gamma=0.99, gae_lambda=0.95, clip_range=0.2`
[26] [gym-pybullet-drones](https://github.com/utiasDSL/gym-pybullet-drones) — normalized-observation / normalized-action-space conventions for SB3 drone envs
