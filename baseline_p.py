"""Hand-scripted proportional policy — env validator and demo fallback.

Two jobs (RL_NOTES.md section 7):
  1. VALIDATE THE ENV independently of PPO. If this controller cannot reach a
     target 5 m dead ahead, the environment is broken, not the controller. That
     is a far faster diagnosis than a PPO run that merely fails to improve.
  2. Be the demo fallback, and the baseline number PPO has to beat.

It flies on the same 12-dim observation the policy sees — no privileged state —
so it exercises the real obs pipeline.

Sign conventions, derived from camera.py / physics.py rather than guessed:
  * body x forward, y LEFT, z up; image +x right, +y up.
  * bbox_x > 0 => target is right of centre. Positive yaw rate is about body +z
    (up), which turns LEFT. So yaw_rate = -k * bbox_x.
  * Positive pitch rate is about body +y (left), which by the right-hand rule
    pitches the NOSE DOWN.
  * With g_body = R^T [0,0,-G], a nose-down pitch theta gives
    g_body = [G sin(theta), 0, -G cos(theta)], so obs[7] IS sin(pitch) with
    positive meaning nose-down. Likewise obs[8] = -sin(roll).

The control problem the uptilt creates: the camera looks cam_angle_deg ABOVE
body forward, so centring the target in frame and accelerating toward it pull
the pitch axis in opposite directions. This controller resolves that by using
PITCH for forward speed and THRUST for the vertical axis, reconstructing the
target's true elevation from bbox_y and the measured pitch:

    target_elevation ~= (cam_angle - pitch) + atan(bbox_y * tan(fov/2))

which is pitch-compensated and therefore stable while manoeuvring.
"""

import argparse
from math import asin, atan, cos, radians, tan

import numpy as np

from dronegym.env import DroneTargetEnv
from dronegym.presets import load_config
from dronegym.state import hover_thrust
from dronegym.task import FOV_DEG

GAINS = dict(
    yaw=1.5,        # bbox_x        -> yaw rate
    pitch=2.0,      # pitch error   -> pitch rate
    roll=2.0,       # roll error    -> roll rate
    alt=0.6,        # elevation rad -> thrust action
    vz=0.05,        # vertical velocity damping
    lean=0.45,      # sin(pitch) commanded when lined up ~ 27 deg nose down
    align=0.4,      # |bbox_x| beyond which we stop leaning in and just turn
    search_yaw=0.5,  # yaw rate command while hunting for a lost target
)

HALF_TAN = tan(radians(FOV_DEG) / 2.0)


def p_policy(obs, cfg, g=GAINS):
    """12-dim observation -> action in [-1, 1]^4."""
    bbox_x, bbox_y, _size, visible = obs[0], obs[1], obs[2], obs[3]
    sin_pitch = float(np.clip(obs[7], -1.0, 1.0))   # +ve = nose down
    v_vert = obs[11] * 15.0                          # undo the obs scaling

    pitch_angle = asin(sin_pitch)
    # Hover needs more thrust once tilted: T = hover / cos(pitch). Expressed in
    # action units, where a[0] = (T - hover) / (max_thrust - hover).
    hover = hover_thrust(cfg)
    hover_frac = hover / (cfg.max_thrust_n - hover)
    lean_ff = hover_frac * (1.0 / max(cos(pitch_angle), 0.3) - 1.0)

    roll_cmd = g["roll"] * obs[8]   # obs[8] = -sin(roll); drives roll back to level

    if visible < 0.5:
        # Lost: hold level and altitude, yaw steadily until something appears.
        thrust = lean_ff - g["vz"] * v_vert
        pitch_cmd = g["pitch"] * (0.0 - sin_pitch)
        return np.clip([thrust, roll_cmd, pitch_cmd, g["search_yaw"]],
                       -1.0, 1.0).astype(np.float32)

    # Turn toward the target.
    yaw_cmd = -g["yaw"] * bbox_x

    # Lean in to accelerate forward, but back off while still badly misaligned —
    # turning first and accelerating second beats spiralling around the target.
    align = max(0.0, 1.0 - abs(bbox_x) / g["align"])
    pitch_cmd = g["pitch"] * (g["lean"] * align - sin_pitch)

    # Vertical: reconstruct true target elevation, then damp on vertical speed.
    elevation = (radians(cfg.cam_angle_deg) - pitch_angle) + atan(bbox_y * HALF_TAN)
    thrust = lean_ff + g["alt"] * elevation - g["vz"] * v_vert

    return np.clip([thrust, roll_cmd, pitch_cmd, yaw_cmd], -1.0, 1.0).astype(np.float32)


def run(cfg, difficulty, episodes, seed=0, verbose=False):
    """Roll out the P controller. Returns (success_rate, mean_steps, event_counts)."""
    env = DroneTargetEnv(cfg, difficulty)
    successes, steps_used, events = 0, [], {}
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + ep)
        done, steps = False, 0
        while not done:
            obs, _reward, terminated, truncated, info = env.step(p_policy(obs, cfg))
            steps += 1
            done = terminated or truncated
        event = info.get("event") or "timeout"
        events[event] = events.get(event, 0) + 1
        successes += bool(info.get("success"))
        steps_used.append(steps)
        if verbose:
            print(f"  ep {ep:3d}  {event:<8} {steps:3d} steps  "
                  f"final dist {info['distance']:.2f} m")
    return successes / episodes, float(np.mean(steps_used)), events


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default="freestyle_5inch",
                    help="preset name or path to a config YAML")
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--difficulty", type=int, default=None,
                    help="single level to run; default runs 0-3")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    levels = [args.difficulty] if args.difficulty is not None else [0, 1, 2, 3]

    print(f"P-controller baseline — {cfg.name} (uptilt {cfg.cam_angle_deg:.0f} deg, "
          f"TWR {cfg.max_thrust_n / hover_thrust(cfg):.1f})")
    print(f"{args.episodes} episodes per level\n")
    print(f"{'level':>5} {'success':>8} {'mean steps':>11}  outcomes")
    results = {}
    for level in levels:
        rate, mean_steps, events = run(cfg, level, args.episodes, args.seed, args.verbose)
        results[level] = rate
        breakdown = "  ".join(f"{k}={v}" for k, v in sorted(events.items()))
        print(f"{level:>5} {rate:>7.0%} {mean_steps:>11.1f}  {breakdown}")

    # Gate 3: if the controller cannot solve level 0, the env is broken.
    if 0 in results:
        print()
        if results[0] >= 0.60:
            print(f"GATE 3 PASS — level 0 success {results[0]:.0%} >= 60%")
        else:
            print(f"GATE 3 FAIL — level 0 success {results[0]:.0%} < 60%. "
                  "Suspect the env, not the controller: check the level-0 reset "
                  "bbox, the action mapping, and the termination ordering.")


if __name__ == "__main__":
    main()
