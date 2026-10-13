"""Hand-written schedule-following baseline: no learning, no yielding.

Each car steers at its own goal and asks for the speed that would land it on the
deadline. It never yields, so when two cars meet the low level brakes both of
them and they deadlock. That deadlock is the gap a learned policy has to close,
which is why this is a LOWER BOUND and not a ceiling.

    python -m arrival_v2.baseline --controller mpc --episodes 50

Use it to sanity-check the environment (a healthy env gives roughly 35-45% here)
and as the "no learning" row of the results table.

Measured with the MPC low level, 25-30 episodes, default configuration:
    gain 0.85 -> 33% success      gain 1.05 -> 33%
    gain 0.95 -> 37% success      gain 1.15 -> 23%   (arrives early)
    gain 1.00 -> 33% success      gain 1.35 ->  7%   (arrives very early)
Timeout stays near 27% at every gain: those are deadlocks, not pacing errors.
"""
import argparse
import json
import time

import numpy as np

from .config import EnvConfig
from .env import ArrivalEnv

PACING_GAIN = 0.95      # swept; >1 systematically arrives early, <0.85 starts arriving late


def schedule_action(env, gain=PACING_GAIN):
    """Aim at the goal and request distance / remaining-time, scaled by `gain`."""
    cfg = env.config
    actions = np.zeros((env.n, 3))
    for i, (x, y, yaw, speed, steer) in enumerate(env.states):
        rotation = np.array([[np.cos(yaw), np.sin(yaw)], [-np.sin(yaw), np.cos(yaw)]])
        goal = rotation @ (env.goals[i] - env.states[i, :2])
        distance = float(np.linalg.norm(goal))
        forward = np.clip(goal[0], cfg.waypoint_forward_min, cfg.waypoint_forward_max)
        lateral = np.clip(goal[1], -cfg.waypoint_lateral_max, cfg.waypoint_lateral_max)
        required = gain * distance / max(env.deadline - env.time, cfg.policy_dt)
        span = cfg.waypoint_forward_max - cfg.waypoint_forward_min
        actions[i] = [2 * (forward - cfg.waypoint_forward_min) / span - 1,
                      lateral / cfg.waypoint_lateral_max,
                      2 * np.clip(required, 0.0, cfg.max_speed) / cfg.max_speed - 1]
    return actions


def run(config, episodes=50, seed_base=900_000, gain=PACING_GAIN, progress=True):
    """Deterministic baseline sweep over `episodes` fixed seeds.

    MPC episodes take a few seconds each, so progress is printed as it goes
    rather than leaving the terminal silent for minutes.
    """
    if episodes < 1:
        raise ValueError("episodes must be positive")
    env = ArrivalEnv(config)
    rows = []
    started = time.perf_counter()
    for episode in range(episodes):
        observation = env.reset(seed=seed_base + episode)
        done = False
        while not done:
            observation, reward, done, info = env.step(schedule_action(env, gain))
        rows.append(info)
        if progress:
            outcome = ("success" if info["success"] else "collision" if info["collision"]
                       else "timeout" if info["timeout"] else "off-deadline")
            done_n = episode + 1
            rate = done_n / max(time.perf_counter() - started, 1e-9)
            print(f"  episode {done_n:3d}/{episodes}  {outcome:12s} "
                  f"running success {np.mean([r['success'] for r in rows]):5.1%}  "
                  f"eta {(episodes - done_n) / max(rate, 1e-9):5.0f}s", flush=True)
    arrived = [row for row in rows if row["all_arrived"]]
    return dict(
        episodes=episodes,
        gain=gain,
        controller=config.controller,
        success_rate=float(np.mean([row["success"] for row in rows])),
        all_arrived_rate=float(np.mean([row["all_arrived"] for row in rows])),
        collision_rate=float(np.mean([row["collision"] for row in rows])),
        timeout_rate=float(np.mean([row["timeout"] for row in rows])),
        arrival_spread_mean=float(np.mean([row["arrival_spread"] for row in arrived])) if arrived else None,
        max_deadline_error_mean=float(np.mean([row["max_deadline_error"] for row in arrived])) if arrived else None,
        episode_reward_mean=float(np.mean([row["episode_reward"] for row in rows])),
        early_arrivals=int(sum(row["early_arrival_count"] for row in rows)),
        on_time_arrivals=int(sum(row["on_time_arrival_count"] for row in rows)),
        late_arrivals=int(sum(row["late_arrival_count"] for row in rows)),
    )


def main():
    parser = argparse.ArgumentParser(description="Schedule-following baseline (no learning).")
    parser.add_argument("--controller", choices=("pp", "mpc"), default="mpc")
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--n-cars", type=int, default=3)
    parser.add_argument("--family", default="mixed")
    parser.add_argument("--gain", type=float, default=PACING_GAIN)
    parser.add_argument("--seed-base", type=int, default=900_000)
    parser.add_argument("--quiet", action="store_true", help="suppress the per-episode progress lines")
    args = parser.parse_args()
    config = EnvConfig(controller=args.controller, n_cars=args.n_cars, family=args.family)
    result = run(config, args.episodes, args.seed_base, args.gain, progress=not args.quiet)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
