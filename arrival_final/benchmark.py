"""Paired, resumable evaluation of frozen policies and local non-learning baselines."""
import argparse
from dataclasses import asdict, replace
import os
from pathlib import Path
import platform
from importlib.metadata import version
import time

import numpy as np
import torch

from .artifacts import assert_frozen, digest, file_digest, read_json, write_json
from .baselines import GoalTimePolicy, LatticeConfig, LocalLatticePolicy
from .config import EnvConfig
from .env import ArrivalEnv
from .mappo import MAPPOAgent
from .missions import load_manifest, scenario_from_record
from .plants import make_plant
from .train import _aggregate_rows, _json_safe, source_hashes

METHODS = {
    "attention_pp": ("attention", "pp", "pp"),
    "attention_pp_mpc": ("attention", "pp", "mpc"),
    "attention_mpc": ("attention", "mpc", "mpc"),
    "mean_mpc": ("mean", "mpc", "mpc"),
    "goal_mpc": (None, None, "mpc"),
    "lattice_mpc": (None, None, "mpc"),
    "goal_pp": (None, None, "pp"),
}
PLANTS = {
    "kinematic": ("kinematic", {}),
    "dynamic": ("dynamic", {}),
    "dynamic_low_friction": ("dynamic", {"friction": 0.6}),
    "dynamic_lag": ("dynamic", {"steer_lag": 0.30, "drive_lag": 0.40}),
    "mujoco": ("mujoco", {}),
    "mujoco_low_friction": ("mujoco", {"friction_scale": 0.3}),
}


def timing_summary(samples, period):
    samples = np.asarray(samples, dtype=float)
    if len(samples) == 0:
        return dict(count=0, mean=None, median=None, p95=None, maximum=None, overrun_rate=None)
    return dict(count=len(samples), mean=float(samples.mean()), median=float(np.median(samples)),
                p95=float(np.quantile(samples, .95)), maximum=float(samples.max()),
                overrun_rate=float(np.mean(samples > period)))


def aggregate(episodes, config, family_weights):
    families = {family: _aggregate_rows([r for r in episodes if r["family"] == family],
                                        config.deadline_window) for family in family_weights}
    result = dict(per_family=families, episodes=len(episodes),
                  raw_successes=sum(r["success"] for r in episodes),
                  raw_success_rate=float(np.mean([r["success"] for r in episodes])))
    for field in ("success_rate", "collision_rate", "timeout_rate", "all_arrived_rate",
                  "arrival_fraction_mean", "fallback_rate_mean"):
        result[field] = sum(family_weights[f] * families[f][field] for f in family_weights)
    for field in ("arrival_spread_mean_all_arrived", "max_deadline_error_mean_all_arrived"):
        mass = result["all_arrived_rate"]
        result[field] = (sum(family_weights[f] * families[f]["all_arrived_rate"] * families[f][field]
                            for f in families if families[f][field] is not None) / mass if mass else None)
    result["physical_collision_rate"] = sum(family_weights[f] * float(np.mean([
        r["physical_collision"] for r in episodes if r["family"] == f])) for f in family_weights)
    periods = {"solver": config.control_dt, "team_control": config.control_dt,
               "policy": config.policy_dt, "simulation_step": config.policy_dt}
    result["timing_seconds"] = {name: timing_summary(
        [value for r in episodes for value in r["timing_samples"][name]], period)
        for name, period in periods.items()}
    result["minimum_footprint_gap"] = min((r["min_separation"] for r in episodes
                                           if r["min_separation"] is not None), default=None)
    return result


def run_benchmark(manifest_path, method, output, checkpoint=None, plant="kinematic",
                  baseline_parameters=None, resume=False, device="cpu", progress=True):
    manifest = load_manifest(manifest_path)
    assert_frozen(manifest["source_hashes"])
    pooling, trained_controller, controller = METHODS[method]
    config = replace(EnvConfig(**manifest["env_config"]), controller=controller).validate()
    parameters = (baseline_parameters or LatticeConfig()).validate()
    metadata = {}
    if pooling is not None:
        if checkpoint is None:
            raise ValueError("A learned method requires --checkpoint")
        policy, metadata = MAPPOAgent.from_checkpoint(checkpoint, device=device)
        assert_frozen(metadata["source_hashes"])
        if (metadata["algorithm"]["pooling"] != pooling or
                metadata["env_config"]["controller"] != trained_controller):
            raise ValueError("Checkpoint architecture/training controller does not match method label")
        permitted = {"controller", "n_cars", "max_neighbors", "family", "family_weights",
                     "arena_half_min", "arena_half_max"}
        mismatches = [k for k, v in config.to_dict().items()
                      if k not in permitted and metadata["env_config"].get(k) != v]
        if mismatches:
            raise ValueError(f"Checkpoint and benchmark disagree on fixed task/controller fields: {mismatches}")
        policy.actor.eval()
    else:
        if checkpoint is not None:
            raise ValueError("Non-learning methods must not receive a checkpoint")
        policy = LocalLatticePolicy(config, parameters) if method == "lattice_mpc" else GoalTimePolicy(config)
    identity = dict(manifest_id=manifest["manifest_id"], method=method, plant=plant,
                    checkpoint_sha256=file_digest(checkpoint) if checkpoint else None,
                    baseline_parameters=asdict(parameters) if method == "lattice_mpc" else None,
                    source_hashes=source_hashes(), device=device)
    job_id = digest(identity)
    output = Path(output)
    if output.exists():
        previous = read_json(output)
        if resume and previous.get("job_id") == job_id:
            return previous
        raise FileExistsError(f"Result exists or belongs to a different job: {output}")
    parts = output.with_name(output.stem + ".episodes")
    if parts.exists():
        if not resume or read_json(parts / "identity.json") != identity:
            raise ValueError("Partial benchmark exists; --resume requires the identical evaluation job")
    else:
        parts.mkdir(parents=True)
        write_json(parts / "identity.json", identity)
    name, kwargs = PLANTS[plant]
    physics = make_plant(name, config, **kwargs)
    env = ArrivalEnv(config, plant=physics)
    rows = []
    try:
        for index, mission in enumerate(manifest["missions"]):
            path = parts / f"{index:06d}.json"
            if path.exists():
                row = read_json(path)
                if row["mission_id"] != mission["mission_id"]:
                    raise ValueError("Partial episode belongs to a different mission")
                rows.append(row)
                continue
            observation = env.reset(scenario=scenario_from_record(mission))
            policy_times, step_times = [], []
            trajectory_times, trajectory_states, trajectory_actions = [env.time], [env.states.copy()], []
            while not env.done:
                started = time.perf_counter()
                action = policy.predict(observation)
                policy_times.append(time.perf_counter() - started)
                observation, _, _, _ = env.step(action)
                step_times.append(time.perf_counter() - started)
                trajectory_times.append(env.time)
                trajectory_states.append(env.states.copy())
                trajectory_actions.append(action.copy())
            trace = parts / f"{index:06d}.trajectory.npz"
            temporary = trace.with_suffix(".tmp")
            with temporary.open("wb") as stream:
                np.savez_compressed(stream, time=trajectory_times, states=trajectory_states,
                                    actions=trajectory_actions, goals=env.goals, deadline=env.deadline)
            temporary.replace(trace)
            row = _json_safe(dict(env.info(), trajectory_file=str(trace.relative_to(output.parent)),
                mission_id=mission["mission_id"],
                scenario_seed=mission["seed"], timing_samples=dict(solver=env.solve_times,
                team_control=env.control_interval_times, policy=policy_times, simulation_step=step_times)))
            write_json(path, row)
            rows.append(row)
            if progress:
                print(f"{method} {plant} {manifest['name']} {index+1}/{len(manifest['missions'])} "
                      f"success={row['success']} collision={row['collision']} timeout={row['timeout']}", flush=True)
    finally:
        if hasattr(physics, "close"):
            physics.close()
    result = dict(kind="arrival_final_result", schema_version=1, job_id=job_id, **identity,
                  case=manifest["name"], split=manifest["split"], env_config=config.to_dict(),
                  family_weights=manifest["family_weights"], training_seed=metadata.get("train_config", {}).get("seed"),
                  trained_steps=metadata.get("global_step"),
                  actor_parameters=sum(p.numel() for p in policy.actor.parameters()) if pooling else 0,
                  checkpoint=str(Path(checkpoint).resolve()) if checkpoint else None,
                  runtime=dict(python=platform.python_version(), system=platform.platform(),
                               processor=platform.processor(), cpu_count=os.cpu_count(),
                               torch_threads=torch.get_num_threads(), numpy=np.__version__,
                               torch=str(torch.__version__), scipy=version("scipy"), casadi=version("casadi"),
                               mujoco=getattr(getattr(physics, "mj", None), "__version__", None)),
                  metrics=aggregate(rows, config, manifest["family_weights"]), episodes=rows)
    write_json(output, result)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--method", choices=METHODS, required=True)
    p.add_argument("--checkpoint", type=Path)
    p.add_argument("--plant", choices=PLANTS, default="kinematic")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--baseline-config", type=Path)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--device", default="cpu")
    args = p.parse_args()
    torch.set_num_threads(1)
    params = LatticeConfig(**read_json(args.baseline_config)["parameters"]) if args.baseline_config else None
    result = run_benchmark(args.manifest, args.method, args.output, args.checkpoint,
                           args.plant, params, args.resume, args.device)
    print(result["metrics"])


if __name__ == "__main__":
    main()
