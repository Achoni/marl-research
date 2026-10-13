"""Zero-shot sim2sim transfer: one frozen policy, several physics back-ends.

    python -m arrival_v2.transfer --checkpoint runs/mpc_s0/best.pt --episodes 40
    python -m arrival_v2.transfer --checkpoint runs/mpc_s0/best.pt --extra

Nothing is retrained and nothing above the plant changes. The policy, the
reference generator, the MPC and the reward are identical in every row; only the
integrator underneath differs. Each row uses the SAME episode seeds, so the
comparison is paired.

Read the table two ways. Rows that match the kinematic row say the mismatch is
too small to matter at this speed. Rows that fall away name the specific physical
effect the pipeline is sensitive to, which is far more useful than a single
"it transfers" claim.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from .config import EnvConfig
from .env import ArrivalEnv
from .mappo import MAPPOAgent
from .plants import DynamicPlant, MujocoPlant

BASE_ROWS = [
    ("kinematic (training model)", lambda cfg: None),
    ("dynamic  mu=1.0, lag 0.15/0.20", lambda cfg: DynamicPlant(cfg)),
    ("dynamic  mu=0.6", lambda cfg: DynamicPlant(cfg, friction=0.6)),
    ("dynamic  mu=0.3", lambda cfg: DynamicPlant(cfg, friction=0.3)),
    ("dynamic  lag 0.30/0.40", lambda cfg: DynamicPlant(cfg, steer_lag=0.30, drive_lag=0.40)),
    ("MuJoCo   F1TENTH", lambda cfg: MujocoPlant(cfg)),
]
EXTRA_ROWS = [
    ("dynamic  mu=0.15 (ice)", lambda cfg: DynamicPlant(cfg, friction=0.15)),
    ("dynamic  lag 0.25/0.35", lambda cfg: DynamicPlant(cfg, steer_lag=0.25, drive_lag=0.35)),
    ("dynamic  lag 0.40/0.50", lambda cfg: DynamicPlant(cfg, steer_lag=0.40, drive_lag=0.50)),
    ("MuJoCo   tyre friction x0.3", lambda cfg: MujocoPlant(cfg, friction_scale=0.3)),
    ("MuJoCo   tyre friction x0.1", lambda cfg: MujocoPlant(cfg, friction_scale=0.1)),
    ("MuJoCo   soft drive gain 8", lambda cfg: MujocoPlant(cfg, torque_gain=8.0)),
]


def evaluate_plant(agent, cfg, build_plant, episodes, seed_base):
    env = ArrivalEnv(cfg, plant=build_plant(cfg))
    success = collision = timeout = arrived = 0
    spreads, errors, separations = [], [], []
    for episode in range(episodes):
        observation = env.reset(seed=seed_base + episode)
        done = False
        while not done:
            observation, reward, done, info = env.step(agent.predict(observation))
        success += info["success"]; collision += info["collision"]
        timeout += info["timeout"]; arrived += info["all_arrived"]
        if info["min_separation"] is not None:
            separations.append(info["min_separation"])
        if info["all_arrived"]:
            spreads.append(info["arrival_spread"]); errors.append(info["max_deadline_error"])
    n = episodes
    return dict(success=success / n, collision=collision / n, timeout=timeout / n,
                all_arrived=arrived / n,
                spread=float(np.mean(spreads)) if spreads else None,
                deadline_error=float(np.mean(errors)) if errors else None,
                worst_separation=float(np.min(separations)) if separations else None)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--episodes", type=int, default=40)
    parser.add_argument("--cars", type=int, help="override team size for every row")
    parser.add_argument("--family", help="restrict to one scenario family")
    parser.add_argument("--seed-base", type=int, default=3_000_000)
    parser.add_argument("--extra", action="store_true", help="add the harder friction/lag rows")
    parser.add_argument("--output", type=Path, help="write the table as JSON")
    arguments = parser.parse_args()

    loaded = MAPPOAgent.from_checkpoint(arguments.checkpoint)
    agent, metadata = loaded if isinstance(loaded, tuple) else (loaded, {})
    agent.actor.eval()
    config = EnvConfig(**metadata["env_config"])
    changes = {}
    if arguments.cars:
        changes["n_cars"] = arguments.cars
    if arguments.family:
        changes["family"] = arguments.family
    if changes:
        from dataclasses import replace
        config = replace(config, **changes).validate()

    print(f"checkpoint : {arguments.checkpoint}")
    print(f"controller : {config.controller}    cars: {config.n_cars}    "
          f"family: {config.family}    episodes/row: {arguments.episodes}")
    print("nothing is retrained; only the integrator under the controller changes\n")
    header = (f"{'plant':>34} {'success':>8} {'arrived':>8} {'collide':>8} "
              f"{'timeout':>8} {'spread':>8} {'worst sep':>10}")
    print(header + "\n" + "-" * len(header))

    rows = BASE_ROWS + (EXTRA_ROWS if arguments.extra else [])
    results = {}
    for label, build in rows:
        stats = evaluate_plant(agent, config, build, arguments.episodes, arguments.seed_base)
        results[label] = stats
        spread = "n/a" if stats["spread"] is None else f"{stats['spread']:.2f}s"
        worst = "n/a" if stats["worst_separation"] is None else f"{stats['worst_separation']:.2f}m"
        print(f"{label:>34} {stats['success']:8.0%} {stats['all_arrived']:8.0%} "
              f"{stats['collision']:8.0%} {stats['timeout']:8.0%} {spread:>8} {worst:>10}", flush=True)

    if arguments.output:
        arguments.output.write_text(json.dumps(
            dict(checkpoint=str(arguments.checkpoint), episodes=arguments.episodes,
                 env_config=config.to_dict(), rows=results), indent=2))
        print(f"\nsaved {arguments.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
