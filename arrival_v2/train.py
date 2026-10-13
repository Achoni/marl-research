"""Train the shared attention policy from scratch; no simulator transfer here.

``--steps`` counts joint environment decisions, not per-car actions, and is the
TOTAL target on resume. Resuming restores networks, optimizer and Torch RNG but
starts a fresh episode (it is not a bit-for-bit replay of uninterrupted training).
"""
from __future__ import annotations

import argparse
from collections import deque
from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path
import platform
import sys
import time

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    __package__ = Path(__file__).resolve().parent.name

import numpy as np
import torch

from .config import EnvConfig
from .curriculum import next_boundary, training_config
from .env import ArrivalEnv
from .mappo import MAPPOAgent

VALIDATION_SEED_BASE = 1_000_000_000
FINAL_TEST_SEED_BASE = 2_000_000_000
DEFAULT_FAMILY_WEIGHTS = dict(random=0.45, crossing=0.20, converging=0.15,
                              unequal=0.15, parallel=0.05)


def _json_safe(value):
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _write_json(path, record):
    path = Path(path)
    temporary = path.with_name("." + path.name + ".tmp")
    temporary.write_text(json.dumps(_json_safe(record), indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def _append_json(path, record):
    with Path(path).open("a") as stream:
        stream.write(json.dumps(_json_safe(record), allow_nan=False) + "\n")


def source_hashes():
    directory = Path(__file__).resolve().parent
    return {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(directory.glob("*.py"))}


def _conditional_mean(rows, key):
    values = [float(row[key]) for row in rows
              if row["all_arrived"] and row.get(key) is not None]
    return float(np.mean(values)) if values else None


def validation_families(config):
    if config.family != "mixed":
        return {config.family: 1.0}
    weights = config.family_weights or DEFAULT_FAMILY_WEIGHTS
    total = sum(weights.values())
    return {family: value / total for family, value in weights.items() if value > 0}


def stratified_plan(config, episodes, seed_base):
    """Fixed family strata, at least one each when the episode budget permits.

    Aggregate metrics weight family means by the target mixture, so rounding
    counts and oversampling a rare family do not alter the selection objective.
    A final test may choose any episode budget; the trainer requires all strata.
    """
    weights = validation_families(config)
    families = sorted(weights)
    counts = {family: int(episodes >= len(families)) for family in families}
    for _ in range(episodes - sum(counts.values())):
        family = max(families, key=lambda name: (episodes * weights[name] - counts[name], name))
        counts[family] += 1
    return [(family, seed_base + index * 10_000_000 + episode)
            for index, family in enumerate(families)
            for episode in range(counts[family])]


def _aggregate_rows(rows, deadline_window):
    metrics = {key + "_rate": float(np.mean([bool(row[key]) for row in rows]))
               for key in ("success", "collision", "timeout", "all_arrived")}
    arrived_times = [(time, row["deadline"]) for row in rows
                     for time in row["arrival_times"] if time is not None]
    metrics.update(
        episodes=len(rows), all_arrived_count=sum(bool(row["all_arrived"]) for row in rows),
        arrival_fraction_mean=float(np.mean([row["arrival_fraction"] for row in rows])),
        arrival_spread_mean_all_arrived=_conditional_mean(rows, "arrival_spread"),
        max_deadline_error_mean_all_arrived=_conditional_mean(rows, "max_deadline_error"),
        episode_reward_mean=float(np.mean([row["episode_reward"] for row in rows])),
        fallback_rate_mean=float(np.mean([row["fallback_rate"] or 0.0 for row in rows])),
        early_arrival_fraction_of_arrivals=(float(np.mean([t < deadline - deadline_window
                                            for t, deadline in arrived_times])) if arrived_times else None),
        late_arrival_fraction_of_arrivals=(float(np.mean([t > deadline + deadline_window
                                           for t, deadline in arrived_times])) if arrived_times else None),
    )
    for name in ("early", "late", "on_time"):
        metrics[name + "_arrival_fraction_of_all_cars"] = float(np.mean([
            row.get(name + "_arrival_count", 0) / len(row["arrival_times"]) for row in rows]))
    return metrics


def evaluate(agent, config, episodes, seed_base=VALIDATION_SEED_BASE, stratified=True):
    """Deterministic validation; timing means include ALL all-arrived episodes.

    The returned per-episode rows retain failures rather than silently dropping
    them. Collision and success rates always use the full episode denominator.
    This is a validation set used for model selection, not a final test set.
    """
    if episodes < 1:
        raise ValueError("episodes must be positive")
    rows = []
    plan = (stratified_plan(config, episodes, seed_base) if stratified
            else [(config.family, seed_base + episode) for episode in range(episodes)])
    environments = {}
    for family, seed in plan:
        if family not in environments:
            environments[family] = ArrivalEnv(replace(config, family=family))
        env = environments[family]
        observation = env.reset(seed=seed)
        done, reward_sum = False, 0.0
        while not done:
            observation, rewards, done, info = env.step(agent.predict(observation))
            reward_sum += float(np.mean(rewards))
        row = {key: info.get(key) for key in (
            "success", "collision", "timeout", "all_arrived", "arrival_spread",
            "max_deadline_error", "fallback_rate", "family", "deadline", "time",
            "arrival_times", "arrival_fraction", "min_separation", "arena_half",
            "mpc_solve_count", "mpc_solve_mean_seconds", "mpc_solve_max_seconds",
            "arrived", "final_goal_distances", "initial_distances", "early_arrival_count",
            "late_arrival_count", "on_time_arrival_count")}
        row.update(seed=seed, episode_reward=reward_sum)
        rows.append(_json_safe(row))
    metrics = _aggregate_rows(rows, config.deadline_window)
    metrics.update(validation_seed_base=seed_base, stratified=bool(stratified), per_family={})
    separations = [row["min_separation"] for row in rows if row["min_separation"] is not None]
    solve_count = sum(row["mpc_solve_count"] or 0 for row in rows)
    total_solve_time = sum((row["mpc_solve_count"] or 0) * (row["mpc_solve_mean_seconds"] or 0.0)
                           for row in rows)
    metrics.update(min_separation_worst=min(separations) if separations else None,
                   mpc_solve_count=solve_count,
                   mpc_solve_mean_seconds=total_solve_time / max(solve_count, 1),
                   mpc_solve_max_seconds=max(row["mpc_solve_max_seconds"] or 0.0 for row in rows))
    for family in sorted({row["family"] for row in rows if row["family"] is not None}):
        subset = [row for row in rows if row["family"] == family]
        metrics["per_family"][family] = _aggregate_rows(subset, config.deadline_window)
    expected = validation_families(config)
    covered = set(expected).issubset(metrics["per_family"])
    metrics["all_target_families_covered"] = covered
    metrics["target_family_weights"] = expected
    metrics["unweighted_episode_rates"] = {key: metrics[key] for key in
        ("success_rate", "collision_rate", "timeout_rate", "all_arrived_rate", "arrival_fraction_mean")}
    if stratified and covered:
        for key in ("success_rate", "collision_rate", "timeout_rate", "all_arrived_rate",
                    "arrival_fraction_mean", "episode_reward_mean", "fallback_rate_mean",
                    "early_arrival_fraction_of_all_cars", "late_arrival_fraction_of_all_cars",
                    "on_time_arrival_fraction_of_all_cars"):
            metrics[key] = sum(weight * metrics["per_family"][family][key]
                               for family, weight in expected.items())
        # Timing is conditional on completion, with its correct mixture mass.
        completion_mass = metrics["all_arrived_rate"]
        for key in ("arrival_spread_mean_all_arrived", "max_deadline_error_mean_all_arrived"):
            metrics[key] = (sum(expected[family] * sub["all_arrived_rate"] * sub[key]
                               for family, sub in metrics["per_family"].items()
                               if sub[key] is not None) / completion_mass
                            if completion_mass > 0 else None)
    return metrics, rows


def selection_score(metrics):
    """Success, collision avoidance, completion, then conditional timing.

    Missing timing is worst, never a zero error. Completing no episodes cannot
    beat a policy with equal success/collision rates that actually reaches goals.
    """
    spread = metrics["arrival_spread_mean_all_arrived"]
    error = metrics["max_deadline_error_mean_all_arrived"]
    return (metrics["success_rate"], -metrics["collision_rate"],
            metrics["all_arrived_rate"], metrics.get("arrival_fraction_mean", 0.0),
            -(spread if spread is not None else 1e9),
            -(error if error is not None else 1e9))


def parser():
    argument_parser = argparse.ArgumentParser(description=__doc__)
    argument_parser.add_argument("--steps", type=int, default=1_000_000,
                                 help="total joint environment steps, including steps before resume")
    argument_parser.add_argument("--run-dir", type=Path, required=True)
    argument_parser.add_argument("--resume", type=Path,
                                 help="restore a checkpoint and start a fresh episode")
    argument_parser.add_argument("--device", default="cpu", help="cpu, cuda, cuda:0, or mps")
    argument_parser.add_argument("--torch-threads", type=int, default=1)
    argument_parser.add_argument("--controller", choices=("pp", "mpc"))
    argument_parser.add_argument("--curriculum", choices=("staged", "none"))
    argument_parser.add_argument("--curriculum-steps", type=int,
                                 help="saved stage schedule budget; default 1000000, independent of --steps")
    argument_parser.add_argument("--checkpoint-every", type=int,
                                 help="archive a periodic checkpoint; default 100000, 0 disables")
    # None means the documented new-run default, or the saved value on resume.
    for flag, value_type in (("seed", int), ("cars", int), ("neighbors", int),
                             ("rollout", int), ("eval-every", int), ("eval-episodes", int),
                             ("arena-half-min", float), ("arena-half-max", float),
                             ("heading-jitter-deg", float), ("initial-speed-max", float),
                             ("deadline-slack-min", float), ("deadline-slack-max", float),
                             ("lr", float), ("epochs", int), ("batch", int), ("hidden", int)):
        argument_parser.add_argument("--" + flag, type=value_type)
    argument_parser.add_argument("--family", help="scenario family; default mixed")
    argument_parser.add_argument("--family-weights", type=str,
                                 help='mixed-family mix, e.g. "random=0.5,crossing=0.3,converging=0.2". '
                                      'Omit a family or give it 0 to drop it (the curriculum drops it too).')
    argument_parser.add_argument("--unequal-short-max", type=float,
                                 help="longest SHORT trip in the unequal family (default 2.0 m; "
                                      "raise it to soften the loitering requirement)")
    return argument_parser


def _configuration(arguments, checkpoint=None):
    checkpoint = checkpoint or {}
    config = EnvConfig(**checkpoint["env_config"]) if checkpoint else EnvConfig()
    mapping = dict(controller="controller", cars="n_cars", neighbors="max_neighbors", family="family",
                   arena_half_min="arena_half_min", arena_half_max="arena_half_max",
                   initial_speed_max="initial_speed_max",
                   deadline_slack_min="deadline_slack_min", deadline_slack_max="deadline_slack_max")
    changes = {field: getattr(arguments, flag) for flag, field in mapping.items()
               if getattr(arguments, flag) is not None}
    if arguments.heading_jitter_deg is not None:
        changes["heading_jitter"] = math.radians(arguments.heading_jitter_deg)
    if getattr(arguments, "unequal_short_max", None) is not None:
        changes["unequal_short_max"] = arguments.unequal_short_max
    if getattr(arguments, "family_weights", None):
        weights = {}
        for item in arguments.family_weights.split(","):
            if not item.strip():
                continue
            name, _, value = item.partition("=")
            if not _:
                raise SystemExit(f"--family-weights entry {item!r} must look like name=value")
            weights[name.strip()] = float(value)
        if not weights or sum(weights.values()) <= 0:
            raise SystemExit("--family-weights must have positive total mass")
        changes["family_weights"] = weights
    if checkpoint:
        mismatch = {key: (getattr(config, key), value) for key, value in changes.items()
                    if getattr(config, key) != value}
        if mismatch:
            raise ValueError(f"Resume must preserve the environment configuration: {mismatch}")
    config = replace(config, **changes).validate()
    settings = dict(seed=0, rollout=1024, eval_every=20_000, eval_episodes=50,
                    curriculum="staged", curriculum_steps=1_000_000, checkpoint_every=100_000)
    settings.update(checkpoint.get("train_config", {}))
    for key in ("seed", "rollout", "eval_every", "eval_episodes", "curriculum",
                "curriculum_steps", "checkpoint_every"):
        supplied = getattr(arguments, key)
        if supplied is not None:
            if checkpoint and supplied != settings[key]:
                raise ValueError(f"Resume must preserve --{key.replace('_', '-')}")
            settings[key] = supplied
    for key in ("rollout", "eval_episodes", "curriculum_steps"):
        if settings[key] < 1:
            raise ValueError(f"{key} must be positive")
    if settings["eval_every"] < 0:
        raise ValueError("eval_every must be nonnegative (0 disables intermediate validation)")
    if settings["checkpoint_every"] < 0:
        raise ValueError("checkpoint_every must be nonnegative")
    if settings["eval_episodes"] < len(validation_families(config)):
        raise ValueError("eval-episodes must cover every target family (at least 5 for default mixed)")
    if not 0 <= settings["seed"] < VALIDATION_SEED_BASE:
        raise ValueError("seed must be between 0 and 999999999")
    algorithm = dict(checkpoint.get("algorithm", {}))
    for key in ("lr", "epochs", "batch", "hidden"):
        supplied = getattr(arguments, key)
        if supplied is not None:
            if supplied <= 0:
                raise ValueError(f"{key} must be positive")
            if checkpoint and supplied != algorithm[key]:
                raise ValueError(f"Resume must preserve --{key}")
            algorithm[key] = supplied
    return config, settings, algorithm


def _episode_seed(seed, serial):
    # All training seeds are below the reserved validation and final-test ranges.
    generated = np.random.SeedSequence([seed, serial, 731]).generate_state(1)[0]
    return int(generated) % VALIDATION_SEED_BASE


def collect_transition(agent, env, observation, state, *, truncate=False):
    """Collect one joint step; the action that achieves arrival stays active.

    An artificial rollout/curriculum cut bootstraps from the final state while
    stopping the GAE trace. An actual task ending receives no bootstrap.
    """
    decision = agent.choose(observation, state)
    active = (~env.arrived).copy()
    next_observation, rewards, done, info = env.step(decision["actions"])
    next_state = env.critic_state()
    next_values = (np.zeros(len(active), dtype=np.float32) if done
                   else agent.estimate_values(next_state))
    agent.remember(observation, state, decision["latent"], decision["log_probs"],
                   decision["values"], rewards, next_values, done,
                   bool(truncate and not done), active)
    return next_observation, next_state, done, info


def rollout_length(global_step, total_steps, rollout, validation_step,
                   checkpoint_step, curriculum_boundary=None):
    """Never overshoot the total budget or an update/distribution boundary."""
    stops = [total_steps, validation_step, checkpoint_step]
    if curriculum_boundary is not None:
        stops.append(curriculum_boundary)
    length = min(rollout, min(stops) - global_step)
    if length < 1:
        raise ValueError("the next rollout boundary must be after the current step")
    return length


def _next_period(step, period, total_steps):
    return ((step // period + 1) * period) if period else total_steps


def _numpy_rng_state():
    name, keys, position, has_gauss, cached_gaussian = np.random.get_state()
    return dict(name=name, keys=keys.tolist(), position=position,
                has_gauss=has_gauss, cached_gaussian=cached_gaussian)


def _restore_numpy_rng(state):
    np.random.set_state((state["name"], np.asarray(state["keys"], dtype=np.uint32),
                         state["position"], state["has_gauss"], state["cached_gaussian"]))


def main(argv=None):
    arguments = parser().parse_args(argv)
    if arguments.steps < 1 or arguments.torch_threads < 1:
        raise ValueError("steps and torch-threads must be positive")
    torch.set_num_threads(arguments.torch_threads)
    checkpoint = None
    if arguments.resume:
        checkpoint = torch.load(arguments.resume, map_location="cpu", weights_only=True)
        if checkpoint.get("schema_version") != 2:
            raise ValueError("Unsupported checkpoint format")
    config, settings, algorithm = _configuration(arguments, checkpoint)
    hashes = source_hashes()
    if checkpoint and checkpoint.get("source_hashes") != hashes:
        raise ValueError("Source files changed since this checkpoint; resume from the saved source snapshot")
    run_dir = arguments.run_dir.resolve()
    if run_dir.exists() and any(run_dir.iterdir()):
        if not arguments.resume or arguments.resume.resolve().parent != run_dir:
            raise ValueError(f"Run directory is not empty: {run_dir}; select a fresh directory")
        existing = run_dir / "latest.pt"
        if existing.exists() and existing.resolve() != arguments.resume.resolve():
            raise ValueError("Resume latest.pt in an existing run; use a fresh run directory to branch from best.pt")
    run_dir.mkdir(parents=True, exist_ok=True)
    np.random.seed(settings["seed"])
    torch.manual_seed(settings["seed"])
    if checkpoint:
        agent, checkpoint = MAPPOAgent.from_checkpoint(arguments.resume, arguments.device,
                                                       load_optimizer=True)
        torch.set_rng_state(checkpoint["torch_rng_state"].cpu())
        if checkpoint.get("numpy_rng_state"):
            _restore_numpy_rng(checkpoint["numpy_rng_state"])
        if torch.cuda.is_available() and checkpoint.get("cuda_rng_states"):
            saved = checkpoint["cuda_rng_states"]
            if len(saved) == torch.cuda.device_count():
                torch.cuda.set_rng_state_all(saved)
    else:
        agent = MAPPOAgent(device=arguments.device, **algorithm)
    provenance = dict(env_config=config.to_dict(), train_config=settings,
                      algorithm=agent.algorithm, source_hashes=hashes,
                      versions=dict(python=platform.python_version(), numpy=np.__version__,
                                    torch=str(torch.__version__)),
                      step_definition="one joint 0.20 s policy/environment decision",
                      source_folder=Path(__file__).resolve().parent.name,
                      resume_mode="fresh episode; not bit-exact uninterrupted continuation",
                      validation_config=config.to_dict(),
                      validation_definition="fixed stratified target distribution; target-weighted rates",
                      seed_ranges=dict(training=[0, VALIDATION_SEED_BASE - 1],
                                       validation_base=VALIDATION_SEED_BASE,
                                       final_test_base=FINAL_TEST_SEED_BASE))
    if not (run_dir / "configuration.json").exists():
        _write_json(run_dir / "configuration.json", provenance)
        snapshot = run_dir / "source"
        snapshot.mkdir(exist_ok=True)
        for name in hashes:
            (snapshot / name).write_bytes((Path(__file__).resolve().parent / name).read_bytes())
    global_step = int(checkpoint.get("global_step", 0)) if checkpoint else 0
    if arguments.steps <= global_step:
        raise ValueError(f"--steps is the total target and must exceed saved step {global_step}")
    completed_episodes = int(checkpoint.get("completed_episodes", 0)) if checkpoint else 0
    episode_serial = int(checkpoint.get("episode_serial", -1)) + 1 if checkpoint else 0
    best = tuple(checkpoint["best_score"]) if checkpoint and checkpoint.get("best_score") else None
    last_validation_step = int(checkpoint.get("last_validation_step", -1)) if checkpoint else -1
    next_validation_step = _next_period(global_step, settings["eval_every"], arguments.steps)
    next_checkpoint_step = _next_period(global_step, settings["checkpoint_every"], arguments.steps)
    training_start_step, start_time = global_step, time.monotonic()
    last_progress_time = start_time
    recent = deque(maxlen=100)
    stage, active_config = training_config(config, global_step, settings["curriculum_steps"],
                                          settings["curriculum"])
    env = ArrivalEnv(active_config)
    observation = env.reset(seed=_episode_seed(settings["seed"], episode_serial))
    state = env.critic_state()
    interrupted = False
    failed = False
    validation_score = checkpoint.get("validation_score") if checkpoint else None

    def metadata():
        return dict(provenance, global_step=global_step, target_steps=arguments.steps,
                    completed_episodes=completed_episodes, episode_serial=episode_serial,
                    best_score=list(best) if best is not None else None,
                    last_validation_step=last_validation_step,
                    checkpoint_validated=(last_validation_step == global_step),
                    validation_score=validation_score if last_validation_step == global_step else None,
                    active_env_config=active_config.to_dict(), curriculum_stage=stage,
                    numpy_rng_state=_numpy_rng_state(),
                    interrupted=interrupted, failed=failed, device=str(agent.device))

    print(f"controller={config.controller} cars={config.n_cars} neighbors={config.max_neighbors} "
          f"target_family={config.family} curriculum={settings['curriculum']} stage={stage} "
          f"steps={global_step}->{arguments.steps} run={run_dir}", flush=True)
    _append_json(run_dir / "training.jsonl", dict(event="resume" if checkpoint else "start",
                                                 **metadata()))
    try:
        while global_step < arguments.steps:
            next_stage, next_config = training_config(config, global_step,
                settings["curriculum_steps"], settings["curriculum"])
            if next_stage != stage:
                discarded = env.time > 0 and not env.done
                _append_json(run_dir / "training.jsonl", dict(
                    event="curriculum_transition", global_step=global_step,
                    previous_stage=stage, curriculum_stage=next_stage,
                    discarded_incomplete_episode=discarded,
                    discarded_episode_time=env.time if discarded else 0.0,
                    previous_rollout_truncated=True,
                    active_env_config=next_config.to_dict()))
                if discarded:
                    episode_serial += 1
                stage, active_config = next_stage, next_config
                env = ArrivalEnv(active_config)
                observation = env.reset(seed=_episode_seed(settings["seed"], episode_serial))
                state = env.critic_state()
                recent.clear()
                print(f"curriculum step={global_step} stage={stage}; "
                      f"discarded_incomplete_episode={discarded}", flush=True)
            boundary = next_boundary(global_step, settings["curriculum_steps"], settings["curriculum"])
            rollout_steps = rollout_length(global_step, arguments.steps, settings["rollout"],
                                            next_validation_step, next_checkpoint_step, boundary)
            for index in range(rollout_steps):
                observation, state, done, info = collect_transition(agent, env, observation, state,
                    truncate=(index == rollout_steps - 1))
                global_step += 1
                now = time.monotonic()
                if now - last_progress_time >= 30.0:
                    print(f"collecting step={global_step}/{arguments.steps} "
                          f"episodes={completed_episodes} "
                          f"sps={(global_step - training_start_step) / (now - start_time):.1f}", flush=True)
                    last_progress_time = now
                if done:
                    completed_episodes += 1
                    row = {key: info.get(key) for key in (
                        "success", "collision", "timeout", "all_arrived", "arrival_spread",
                        "max_deadline_error", "fallback_rate", "episode_reward", "family",
                        "arrival_fraction", "early_arrival_count", "late_arrival_count")}
                    recent.append(row)
                    _append_json(run_dir / "training.jsonl", dict(
                        event="episode", global_step=global_step, episode=completed_episodes,
                        scenario_seed=_episode_seed(settings["seed"], episode_serial),
                        curriculum_stage=stage, **row))
                    episode_serial += 1
                    observation = env.reset(seed=_episode_seed(settings["seed"], episode_serial))
                    state = env.critic_state()
            learning = agent.learn()
            elapsed = time.monotonic() - start_time
            summary = dict(event="update", global_step=global_step,
                           curriculum_stage=stage,
                           completed_episodes=completed_episodes,
                           environment_steps_per_second=(global_step - training_start_step) / max(elapsed, 1e-9),
                           **learning)
            for key in ("success", "collision", "timeout"):
                summary["recent_" + key + "_rate"] = float(np.mean([row[key] for row in recent])) if recent else None
            _append_json(run_dir / "training.jsonl", summary)
            print(f"step={global_step} episodes={completed_episodes} "
                  f"success={summary['recent_success_rate']} collision={summary['recent_collision_rate']} "
                  f"sps={summary['environment_steps_per_second']:.1f}", flush=True)
            if global_step >= next_validation_step or global_step == arguments.steps:
                validation, validation_rows = evaluate(agent, config, settings["eval_episodes"])
                last_validation_step = global_step
                _append_json(run_dir / "metrics.jsonl", dict(global_step=global_step, **validation))
                for row in validation_rows:
                    _append_json(run_dir / "validation_episodes.jsonl", dict(global_step=global_step, **row))
                score = selection_score(validation)
                validation_score = list(score)
                if best is None or score > best:
                    best = score
                    agent.save(run_dir / "best.pt", metadata())
                print(f"validation step={global_step} success={validation['success_rate']:.3f} "
                      f"collision={validation['collision_rate']:.3f} "
                      f"spread={validation['arrival_spread_mean_all_arrived']}", flush=True)
                next_validation_step = _next_period(global_step, settings["eval_every"], arguments.steps)
            if settings["checkpoint_every"] and global_step >= next_checkpoint_step:
                agent.save(run_dir / "checkpoints" / f"step_{global_step:09d}.pt", metadata())
                next_checkpoint_step = _next_period(global_step, settings["checkpoint_every"], arguments.steps)
            agent.save(run_dir / "latest.pt", metadata())
    except KeyboardInterrupt:
        interrupted = True
        # Keep current weights/optimizer; incomplete on-policy rollout is discarded.
        agent.buffer.clear()
        print("Interrupted: saving current weights; resume starts a fresh episode.", flush=True)
    except Exception:
        failed = True
        agent.buffer.clear()
        raise
    finally:
        agent.save(run_dir / "latest.pt", metadata())
    _append_json(run_dir / "training.jsonl", dict(event="interrupted" if interrupted else "finished",
                                                 **metadata()))
    print(f"Saved {run_dir / 'latest.pt'}; "
          f"validated best exists={(run_dir / 'best.pt').exists()}", flush=True)
    return 130 if interrupted else 0


if __name__ == "__main__":
    raise SystemExit(main())
