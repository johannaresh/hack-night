"""PPO training for DroneTargetEnv — WP3.

Run:
    python train.py --config configs/freestyle_5inch.yaml --steps 2000000 \
        --run-name smoke1
    python train.py --config freestyle_5inch --steps 200000 --run-name dbg \
        --difficulty 0            # pins the curriculum, disables its callback

The four choices in here that are load-bearing (see RL_NOTES.md sections 1.1 and 4,
EXECUTION_PLAN.md section 1):

  1. `make_vec_env` is the ONE definition of the wrapper stack, and zsun's
     evaluate.py imports it. The VecFrameStack is part of the observation space,
     not an optimisation — details in that function's docstring.
  2. gamma = 0.995. The effective horizon is 1/(1-gamma) *steps*, not seconds. At
     50 Hz control the SB3 default 0.99 buys 100 steps = 2 s, which is shorter
     than the 10 s episode — the agent would be structurally blind to whether it
     ever reaches the target. 0.995 gives 200 steps = 4 s.
  3. The curriculum widens the initial-state distribution and leaves the reward
     alone, promoting on rolling success rate rather than a reward plateau.
  4. Every reward term is logged separately. When reward rises but the flight
     looks worse, `reward_terms/*` names the gamed term in about thirty seconds.
     That is the primary debugging surface for the whole project.

TensorBoard: `tensorboard --logdir runs/`. Watch curriculum/level,
curriculum/success_rate, reward_terms/*, rollout/ep_len_mean, time/fps.
"""

import argparse
import time
from collections import defaultdict, deque
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CheckpointCallback,
    EvalCallback,
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack

from dronegym.config_shim import load_config
from dronegym.env import DroneTargetEnv
from dronegym.task import MAX_EPISODE_STEPS, N_STACK, OBS_DIM

# --- PPO hyperparameters (RL_NOTES.md section 4) -----------------------------
N_STEPS = 512               # x 16 envs = 8192-step rollout
BATCH_SIZE = 2048           # deliberate deviation from SB3's default 64
N_EPOCHS = 10
LEARNING_RATE = 3e-4
GAMMA = 0.995               # see module docstring — the one people get wrong
GAE_LAMBDA = 0.95
CLIP_RANGE = 0.2
ENT_COEF = 0.0
VF_COEF = 0.5
MAX_GRAD_NORM = 0.5
NET_ARCH = [64, 64]         # what the quadrotor RL literature converges on

# --- curriculum --------------------------------------------------------------
MAX_LEVEL = 3
PROMOTE_WINDOW = 100        # episodes; widen to 200 if success oscillates at a boundary
PROMOTE_THRESHOLD = 0.7     # rolling success rate

# --- run bookkeeping ---------------------------------------------------------
RUNS_DIR = Path("runs")
CHECKPOINT_EVERY = 100_000  # env steps
EVAL_EVERY = 50_000         # env steps
EVAL_EPISODES = 20
EVAL_N_ENVS = 4
EVAL_SEED = 10_000          # far from the training seeds, so eval is a real held-out draw
FPS_REPORT_EVERY = 10_000   # env steps
FPS_GATE = 2_000            # EXECUTION_PLAN.md gate 4


def make_vec_env(cfg_path, n_envs=16, difficulty=0, seed=0):
    """Build the training/eval env stack. THE SINGLE SOURCE OF TRUTH FOR WRAPPERS.

    evaluate.py and viz/replay.py must build their env by calling this function
    (or by replicating it exactly). A checkpoint loaded without an identical
    stack either raises a 48-vs-12 observation-dimension mismatch or, worse,
    silently flies like garbage. This is integration landmine #1.

    Why the frame stack is required rather than optional: camera.py returns
    np.zeros(4) both when the target is behind the camera AND when it leaves the
    frame, so obs[0:4] is bit-identical whether the target just slid off the left
    edge or the right edge. A single frame therefore cannot tell the agent which
    way to yaw to reacquire — the state is not Markov. Stacking 4 frames lets the
    policy see the bbox drifting toward an edge before it vanished, which encodes
    the recovery direction. The env itself stays 12-dim; the policy sees
    OBS_DIM * N_STACK = 12 * 4 = 48.

    Why DummyVecEnv and not SubprocVecEnv: the physics is pure NumPy and a step
    costs microseconds, so per-step pickle-and-pipe overhead would likely exceed
    the physics cost outright. Measure FPS before reaching for subprocesses.

    Args:
        cfg_path: path to a preset YAML, or a bare preset name
        n_envs: parallel copies inside the DummyVecEnv
        difficulty: curriculum level 0-3 every sub-env starts at
        seed: env i is seeded with `seed + i`

    Returns:
        VecFrameStack wrapping DummyVecEnv of Monitor-wrapped DroneTargetEnv.
    """
    cfg = load_config(cfg_path)

    def thunk(i):
        def _f():
            env = DroneTargetEnv(cfg, difficulty)
            env.reset(seed=seed + i)
            # Monitor sits per-env inside the DummyVecEnv (it is an env-level
            # wrapper) and lifts "success" out of the final info dict, which is
            # what CurriculumCallback promotes on.
            return Monitor(env, info_keywords=("success",))
        return _f

    venv = DummyVecEnv([thunk(i) for i in range(n_envs)])
    return VecFrameStack(venv, n_stack=N_STACK)


class CurriculumCallback(BaseCallback):
    """Promote the initial-state distribution on rolling success rate.

    The reward function stays fixed; only where the drone and target spawn
    widens (Learning to Fly in Seconds, RL_NOTES.md section 4). Promotion is
    keyed on success rate rather than a reward plateau because "reward stable for
    1M steps" is far too slow for this budget.

    The eval env is promoted in lockstep. Evaluating at a stale difficulty makes
    best_model selection meaningless — the returns would not be comparable
    across the promotion boundary.
    """

    def __init__(self, eval_env=None, window=PROMOTE_WINDOW,
                 threshold=PROMOTE_THRESHOLD, max_level=MAX_LEVEL, verbose=0):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.threshold = threshold
        self.max_level = max_level
        self.level = 0
        self.window = deque(maxlen=window)
        # Held across a promotion (which clears the window) so the TensorBoard
        # curve does not drop to zero every time the level advances.
        self.success_rate = 0.0

    def _on_step(self):
        for info in self.locals.get("infos", ()):
            # Monitor attaches "episode" only on the step an episode finishes.
            if "episode" not in info:
                continue
            self.window.append(bool(info.get("success", False)))

        if self.window:
            self.success_rate = float(np.mean(self.window))

        if (len(self.window) == self.window.maxlen
                and self.success_rate > self.threshold
                and self.level < self.max_level):
            self._promote()
        return True

    def _promote(self):
        self.level += 1
        self.training_env.env_method("set_difficulty", self.level)
        if self.eval_env is not None:
            self.eval_env.env_method("set_difficulty", self.level)
        self.window.clear()
        print(f"[curriculum] step {self.num_timesteps:,}: promoted to level "
              f"{self.level} (success rate {self.success_rate:.2f})", flush=True)

    def _on_rollout_end(self):
        self.logger.record("curriculum/level", self.level)
        self.logger.record("curriculum/success_rate", self.success_rate)


class RewardTermsCallback(BaseCallback):
    """Log the per-episode sum of every reward term to reward_terms/<name>.

    Non-negotiable (RL_NOTES.md section 3): when total reward rises but the
    flight looks worse, these curves identify which term is being gamed in
    seconds. The classic signatures are visibility climbing while progress stays
    flat (hover-and-stare) and the rate penalty pinned at zero while progress is
    also zero (rate-penalty paralysis).

    The env attaches info["terms_sum"] — the per-term weighted sums accumulated
    over the whole episode — on the final step of each episode.
    """

    def __init__(self, window=PROMOTE_WINDOW, verbose=0):
        super().__init__(verbose)
        self.window = window
        self.recent = defaultdict(lambda: deque(maxlen=window))
        self._warned = False

    def _on_step(self):
        for info in self.locals.get("infos", ()):
            if "episode" not in info:
                continue
            terms = info.get("terms_sum")
            if not terms:
                # Defensive: never take down a multi-hour run over a log field.
                if not self._warned:
                    print("[reward_terms] WARNING: finished episode carried no "
                          "'terms_sum'; per-term curves will be empty.", flush=True)
                    self._warned = True
                continue
            for name, value in terms.items():
                self.recent[name].append(float(value))
        return True

    def _on_rollout_end(self):
        for name, values in self.recent.items():
            if values:
                self.logger.record(f"reward_terms/{name}", float(np.mean(values)))


class ThroughputCallback(BaseCallback):
    """Print steps/s to stdout periodically.

    SB3 logs time/fps to TensorBoard, but the operator needs to see throughput in
    the terminal within the first minute: gate 4 is FPS >= 2000, and finding out
    two hours in that the run is crawling costs the whole budget.
    """

    def __init__(self, every=FPS_REPORT_EVERY, gate=FPS_GATE, verbose=0):
        super().__init__(verbose)
        self.every = every
        self.gate = gate
        self._t_start = 0.0
        self._t_last = 0.0
        self._steps_last = 0
        self._next_report = 0
        self._warned = False

    def _on_training_start(self):
        self._t_start = self._t_last = time.perf_counter()
        self._steps_last = self.model.num_timesteps
        self._next_report = self.model.num_timesteps + self.every

    def _on_step(self):
        if self.num_timesteps < self._next_report:
            return True
        now = time.perf_counter()
        window = max(now - self._t_last, 1e-9)
        fps = (self.num_timesteps - self._steps_last) / window
        print(f"[throughput] {self.num_timesteps:>12,} steps  {fps:8.0f} FPS  "
              f"{(now - self._t_start) / 60.0:6.1f} min elapsed", flush=True)
        if fps < self.gate and not self._warned:
            print(f"[throughput] WARNING: {fps:.0f} FPS is below the gate of "
                  f"{self.gate}; check n_envs and the physics step cost.", flush=True)
            self._warned = True
        self._t_last, self._steps_last = now, self.num_timesteps
        self._next_report = self.num_timesteps + self.every
        return True


def build_model(venv, n_envs, seed, tensorboard_log):
    """PPO with the RL_NOTES.md section 4 hyperparameters."""
    # 8192-step rollout / 2048 batch = 4 minibatches at the default 16 envs.
    # Shrink the batch rather than let SB3 silently truncate on small --n-envs.
    batch_size = min(BATCH_SIZE, N_STEPS * n_envs)
    if batch_size != BATCH_SIZE:
        print(f"[ppo] batch_size lowered to {batch_size} to fit a "
              f"{N_STEPS * n_envs}-step rollout", flush=True)

    return PPO(
        "MlpPolicy",
        venv,
        n_steps=N_STEPS,
        batch_size=batch_size,
        n_epochs=N_EPOCHS,
        learning_rate=LEARNING_RATE,
        gamma=GAMMA,
        gae_lambda=GAE_LAMBDA,
        clip_range=CLIP_RANGE,
        ent_coef=ENT_COEF,
        vf_coef=VF_COEF,
        max_grad_norm=MAX_GRAD_NORM,
        policy_kwargs=dict(net_arch=NET_ARCH, activation_fn=torch.nn.Tanh),
        tensorboard_log=tensorboard_log,
        seed=seed,
        verbose=1,
    )


def parse_args():
    p = argparse.ArgumentParser(
        description="Train PPO on DroneTargetEnv.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--config", required=True,
                   help="preset YAML path, or a bare preset name e.g. freestyle_5inch")
    p.add_argument("--run-name", required=True,
                   help="artifacts land in runs/<run-name>/")
    p.add_argument("--steps", type=int, default=2_000_000,
                   help="total env steps")
    p.add_argument("--n-envs", type=int, default=16,
                   help="DummyVecEnv width")
    p.add_argument("--difficulty", type=int, default=None, choices=range(MAX_LEVEL + 1),
                   help="pin the curriculum at this level and disable the "
                        "curriculum callback (debugging); omit for the normal run")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--torch-threads", type=int, default=1,
                   help="torch intra-op threads. 1 is faster here (~+12%% measured) "
                        "because the [64,64] net is far too small to pay for "
                        "thread dispatch, and it stops the final parallel runs "
                        "from fighting each other for cores. 0 = leave as-is")
    return p.parse_args()


def main():
    args = parse_args()

    if args.torch_threads > 0:
        torch.set_num_threads(args.torch_threads)

    run_dir = RUNS_DIR / args.run_name
    (run_dir / "ckpt").mkdir(parents=True, exist_ok=True)

    curriculum = args.difficulty is None
    start_level = 0 if curriculum else args.difficulty

    venv = make_vec_env(args.config, n_envs=args.n_envs,
                        difficulty=start_level, seed=args.seed)
    # Separate deterministic env for best_model selection. Its seed is far from
    # the training seeds so eval episodes are genuinely held out.
    eval_venv = make_vec_env(args.config, n_envs=EVAL_N_ENVS,
                             difficulty=start_level, seed=EVAL_SEED)

    print(f"config      : {args.config}")
    print(f"run dir     : {run_dir}")
    print(f"envs        : {args.n_envs} (DummyVecEnv)  eval {EVAL_N_ENVS}")
    print(f"obs         : {OBS_DIM} x {N_STACK} stacked = {OBS_DIM * N_STACK} dims")
    print(f"episode cap : {MAX_EPISODE_STEPS} steps @ 50 Hz = "
          f"{MAX_EPISODE_STEPS / 50.0:.0f} s; gamma {GAMMA} -> "
          f"{1.0 / (1.0 - GAMMA):.0f}-step horizon")
    mode = ("on, starting at level 0" if curriculum
            else f"OFF, pinned at level {start_level}")
    print(f"curriculum  : {mode}")
    print(f"total steps : {args.steps:,}", flush=True)

    model = build_model(venv, args.n_envs, args.seed, tensorboard_log=str(RUNS_DIR))

    callbacks = []
    if curriculum:
        # First in the list so a promotion reaches the eval env before
        # EvalCallback can run on the same step.
        callbacks.append(CurriculumCallback(eval_env=eval_venv))
    callbacks += [
        RewardTermsCallback(),
        ThroughputCallback(),
        # save_freq / eval_freq count calls to the VecEnv, i.e. env steps / n_envs.
        CheckpointCallback(
            save_freq=max(1, CHECKPOINT_EVERY // args.n_envs),
            save_path=str(run_dir / "ckpt"),
            name_prefix="model",
        ),
        EvalCallback(
            eval_venv,
            eval_freq=max(1, EVAL_EVERY // args.n_envs),
            n_eval_episodes=EVAL_EPISODES,
            deterministic=True,
            best_model_save_path=str(run_dir),
            log_path=str(run_dir),
        ),
    ]

    try:
        model.learn(total_timesteps=args.steps, callback=callbacks,
                    tb_log_name=args.run_name)
    finally:
        model.save(run_dir / "final_model")
        venv.close()
        eval_venv.close()

    print()
    print(f"final model : {run_dir / 'final_model.zip'}")
    print(f"best model  : {run_dir / 'best_model.zip'}  "
          f"(EvalCallback, deterministic, {EVAL_EPISODES} episodes)")
    print(f"tensorboard : tensorboard --logdir {RUNS_DIR}/")
    # ASCII only in printed strings: Windows consoles default to cp1252 and
    # mangle anything else.
    print("REMINDER: evaluate.py must build its env with train.make_vec_env(...) - "
          f"the policy expects {OBS_DIM * N_STACK} dims, i.e. VecFrameStack(n_stack="
          f"{N_STACK}). Loading a checkpoint without it fails or flies like garbage.")


if __name__ == "__main__":
    main()
