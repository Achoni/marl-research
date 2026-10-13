"""Watch a trained policy drive real MuJoCo cars, zero-shot, in a native window.

    mjpython -m arrival_final.view_mujoco --checkpoint runs/mpc_s0/best.pt
    mjpython -m arrival_final.view_mujoco --checkpoint runs/mpc_s0/best.pt --cars 6
    mjpython -m arrival_final.view_mujoco --checkpoint runs/mpc_s0/best.pt --family crossing

macOS needs ``mjpython`` for the viewer window; plain ``python3`` works with
--headless, which prints the same per-episode lines without a window.

The policy is frozen and was trained in the kinematic bicycle model. Here it is
driving torque-controlled motors and servo-driven steering racks with real tyre
contacts, with nothing retrained. Targets are drawn as the coloured spheres.
"""
import argparse
import math
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

from .config import EnvConfig
from .env import ArrivalEnv
from .mappo import MAPPOAgent
from .plants import MujocoPlant
from .scenarios import FAMILIES


def outcome(info):
    if info["success"]:
        return "SUCCESS"
    if info["collision"]:
        return "COLLISION"
    if info["timeout"]:
        return "TIMEOUT"
    return "off-deadline"


def episode_line(index, env, info):
    times = [None if t is None else round(t, 2) for t in info["arrival_times"]]
    spread = info["arrival_spread"]
    gap = info["min_separation"]
    gap_text = "n/a" if gap is None else f"{gap:.2f}m"
    return (f"episode {index:3d} [{info['family']:>10}]  {outcome(info):12s} "
            f"T={info['deadline']:5.2f}s  arrivals={times}  "
            f"spread={'n/a' if spread is None else f'{spread:.2f}s'}  "
            f"min sep={gap_text}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--cars", type=int, help="zero-shot team size override")
    parser.add_argument("--collision-model", choices=("discs", "single"),
                        help="discs = body-aligned circles (correct); single = legacy rear-axle circle")
    parser.add_argument("--footprint-discs", type=int, help="number of body discs when using discs")
    parser.add_argument("--neighbors", type=int, help="actor neighbour count override")
    parser.add_argument("--family", choices=FAMILIES, help="restrict the scenario distribution")
    parser.add_argument("--speed", type=float, default=1.0, help="playback multiplier, 0.125 to 8")
    parser.add_argument("--episodes", type=int, default=20, help="headless only")
    parser.add_argument("--friction-scale", type=float, default=1.0,
                        help="scale the MuJoCo tyre/floor friction, e.g. 0.3 for a slippery floor")
    parser.add_argument("--headless", action="store_true", help="metrics only, no window")
    parser.add_argument("--seed", type=int, help="fix the scenario seed")
    arguments = parser.parse_args()
    if not 0.125 <= arguments.speed <= 8:
        parser.error("--speed must be between 0.125 and 8")

    loaded = MAPPOAgent.from_checkpoint(arguments.checkpoint)
    agent, metadata = loaded if isinstance(loaded, tuple) else (loaded, {})
    agent.actor.eval()
    config = EnvConfig(**metadata["env_config"])
    changes = {key: value for key, value in (("n_cars", arguments.cars),
                                             ("max_neighbors", arguments.neighbors),
                                             ("family", arguments.family),
                                             ("collision_model", arguments.collision_model),
                                             ("footprint_discs", arguments.footprint_discs))
               if value is not None}
    if changes:
        config = replace(config, **changes).validate()

    plant = MujocoPlant(config, render=not arguments.headless,
                        friction_scale=arguments.friction_scale)
    env = ArrivalEnv(config, plant=plant)
    print(f"checkpoint : {arguments.checkpoint}")
    print(f"controller : {config.controller}   cars: {config.n_cars}   family: {config.family}")
    print(f"plant      : MuJoCo (friction x{arguments.friction_scale})   "
          f"{'headless' if arguments.headless else 'window'}\n")

    stats = dict(success=0, collision=0, timeout=0, episodes=0, spreads=[])
    index = 0
    try:
        while True:
            if arguments.headless and index >= arguments.episodes:
                break
            if not arguments.headless and not plant.viewer.is_running():
                break
            index += 1
            seed = arguments.seed if arguments.seed is not None else None
            observation = env.reset(seed=seed if seed is None else seed + index - 1)
            done = False
            while not done:
                if not arguments.headless and not plant.viewer.is_running():
                    break
                started = time.perf_counter()
                observation, reward, done, info = env.step(agent.predict(observation))
                if not arguments.headless:
                    time.sleep(max(0.0, config.policy_dt / arguments.speed
                                   - (time.perf_counter() - started)))
            if not done:
                break
            stats["episodes"] += 1
            stats["success"] += info["success"]; stats["collision"] += info["collision"]
            stats["timeout"] += info["timeout"]
            if info["all_arrived"]:
                stats["spreads"].append(info["arrival_spread"])
            print(episode_line(index, env, info), flush=True)
            if not arguments.headless:
                time.sleep(1.5)
    finally:
        plant.close()

    n = max(stats["episodes"], 1)
    spread = np.mean(stats["spreads"]) if stats["spreads"] else float("nan")
    print(f"\n=== MuJoCo, {config.n_cars} cars, {stats['episodes']} episodes ===")
    print(f"  success {stats['success'] / n:.0%}   collision {stats['collision'] / n:.0%}   "
          f"timeout {stats['timeout'] / n:.0%}   arrival spread {spread:.2f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
