"""Prepare, train and evaluate the frozen remote-PC study."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import math
import os
from pathlib import Path
import subprocess
import sys

import torch

from .artifacts import assert_frozen, read_json, write_json
from .benchmark import METHODS, PLANTS
from .config import EnvConfig
from .missions import build_manifest
from .train import source_hashes

PACKAGE = Path(__file__).resolve().parent
VARIANTS = {"attention_pp": ("attention", "pp"),
            "attention_mpc": ("attention", "mpc"), "mean_mpc": ("mean", "mpc")}
DEFAULT_METHODS = ("attention_pp", "attention_pp_mpc", "attention_mpc", "mean_mpc", "goal_mpc", "lattice_mpc")


def prepare(args):
    root = args.root.resolve()
    if root.exists() and any(root.iterdir()):
        raise ValueError("Prepare requires a new/empty study directory")
    if (args.steps < 1 or args.curriculum_steps < 1 or args.per_family < 1
            or len(set(args.seeds)) != len(args.seeds) or any(s < 0 or s >= 1_000_000_000 for s in args.seeds)):
        raise ValueError("Invalid budget, episode count or distinct training seeds")
    cfg = EnvConfig()
    specs = [("target_3", cfg), ("unequal_3", replace(cfg, family="unequal"))]
    for count in sorted(set([3] + args.cars)):
        if count < 1:
            raise ValueError("Team sizes must be positive")
        for mode in ("fixed", "density"):
            if mode == "density" and count == 3:
                continue  # fixed_3 is also the common density-study reference.
            half = 6.0 * (math.sqrt(count / 3) if mode == "density" else 1.0)
            specs.append((f"{mode}_{count}", replace(cfg, n_cars=count,
                         arena_half_min=half, arena_half_max=half)))
    root.mkdir(parents=True, exist_ok=True)
    manifests = {}
    for index, (name, case_config) in enumerate(specs):
        path = Path("missions") / f"{name}.json"
        manifest = build_manifest(case_config, args.per_family,
                                  2_100_000_000 + index * 10_000_000, name=name)
        write_json(root / path, manifest)
        manifests[name] = str(path)
        print(f"Prepared {name}: {len(manifest['missions'])} saved missions", flush=True)
    validation = build_manifest(cfg, args.validation_per_family, 1_500_000_000,
                                split="validation", name="baseline_validation")
    write_json(root / "missions" / "baseline_validation.json", validation)
    protocol = dict(kind="arrival_final_study", source_hashes=source_hashes(),
        env_config=cfg.to_dict(), seeds=args.seeds, steps=args.steps, curriculum_steps=args.curriculum_steps,
        variants={k: dict(pooling=v[0], controller=v[1]) for k, v in VARIANTS.items()},
        cases=manifests, validation_manifest="missions/baseline_validation.json",
        evaluation_methods=list(DEFAULT_METHODS),
        primary_case="target_3", stress_case="unequal_3",
        training=dict(rollout=1024, eval_every=20_000, eval_episodes=50, checkpoint_every=100_000),
        notes=["One million JOINT environment steps per run by default.",
               "Unequal-trip family held out as stress; random missions still have unequal distances.",
               "No static obstacles/walls; goals and shared deadlines supplied before execution.",
               "Best checkpoint selected on validation only; report all training seeds."])
    write_json(root / "study.json", protocol)
    snapshot = root / "source"
    snapshot.mkdir()
    for name in protocol["source_hashes"]:
        (snapshot / name).write_bytes((PACKAGE / name).read_bytes())
    print(f"Frozen study saved to {root / 'study.json'}")


def load_study(root):
    protocol = read_json(root / "study.json")
    assert_frozen(protocol["source_hashes"])
    return protocol


def training_jobs(root, protocol, variants, seeds, device, resume):
    jobs = []
    for variant in variants:
        for seed in seeds:
            if seed not in protocol["seeds"]:
                raise ValueError(f"Seed {seed} is not part of the prepared study")
            run = root / "runs" / f"{variant}_s{seed}"
            latest = run / "latest.pt"
            options = protocol["variants"][variant]
            command = [sys.executable, "-m", "arrival_final.train", "--run-dir", str(run),
                       "--steps", str(protocol["steps"]), "--device", device]
            if latest.exists():
                if not resume:
                    raise FileExistsError(f"Run exists: {run}; use --resume")
                saved = torch.load(latest, map_location="cpu", weights_only=True)
                assert_frozen(saved["source_hashes"])
                if (saved["train_config"]["seed"] != seed or
                        saved["algorithm"]["pooling"] != options["pooling"] or
                        saved["env_config"]["controller"] != options["controller"]):
                    raise ValueError("Existing run does not match the study variant")
                if saved["global_step"] >= protocol["steps"]:
                    if saved.get("failed") or saved.get("interrupted") or not saved.get("checkpoint_validated"):
                        raise ValueError(f"{run} reached its budget without clean final validation; inspect its logs")
                    print(f"Already completed: {run.name}")
                    continue
                command += ["--resume", str(latest)]
            else:
                command += ["--seed", str(seed), "--pooling", options["pooling"],
                            "--controller", options["controller"], "--curriculum-steps", str(protocol["curriculum_steps"])]
                for name, value in protocol["training"].items():
                    command += ["--" + name.replace("_", "-"), str(value)]
            jobs.append((run.name, command))
    return jobs


def evaluation_jobs(root, protocol, methods, seeds, cases, plants, resume):
    jobs = []
    for case in cases:
        if case not in protocol["cases"]:
            raise ValueError(f"Unknown case {case}; choose from {list(protocol['cases'])}")
        for plant in plants:
            for method in methods:
                pooling, trained_controller, _ = METHODS[method]
                for seed in seeds if pooling else [None]:
                    if seed is not None and seed not in protocol["seeds"]:
                        raise ValueError("Seed not in prepared study")
                    suffix = f"_s{seed}" if seed is not None else ""
                    label = f"{case}__{plant}__{method}{suffix}"
                    command = [sys.executable, "-m", "arrival_final.benchmark", "--manifest",
                               str(root / protocol["cases"][case]), "--method", method,
                               "--plant", plant, "--output", str(root / "results" / f"{label}.json")]
                    if pooling:
                        variant = f"{pooling}_{trained_controller}"
                        checkpoint = root / "runs" / f"{variant}_s{seed}" / "best.pt"
                        if not checkpoint.exists():
                            raise FileNotFoundError(f"Missing trained checkpoint: {checkpoint}")
                        command += ["--checkpoint", str(checkpoint)]
                    if method == "lattice_mpc":
                        parameters = root / "baseline_parameters.json"
                        if not parameters.exists():
                            raise FileNotFoundError("Tune the lattice baseline on validation first: study tune-baseline")
                        command += ["--baseline-config", str(parameters)]
                    if resume:
                        command += ["--resume"]
                    jobs.append((label, command))
    return jobs


def launch(root, jobs, workers=1, dry_run=False):
    import shlex
    if workers < 1:
        raise ValueError("workers must be positive")
    if dry_run:
        for _, command in jobs:
            print(shlex.join(command))
        return
    logs = root / "logs"
    logs.mkdir(exist_ok=True)
    env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
               MKL_NUM_THREADS="1", PYTHONUNBUFFERED="1")

    def run(job):
        label, command = job
        print(f"Starting {label}; log={logs / (label + '.log')}", flush=True)
        with (logs / (label + ".log")).open("a") as stream:
            result = subprocess.run(command, cwd=PACKAGE.parent, env=env,
                                    stdout=stream, stderr=subprocess.STDOUT)
        print(f"Finished {label}: exit={result.returncode}", flush=True)
        return result.returncode

    with ThreadPoolExecutor(max_workers=workers) as executor:
        codes = list(executor.map(run, jobs))
    if any(codes):
        raise RuntimeError("One or more jobs failed; inspect their logs. Checkpoints/episode files are preserved.")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    commands = p.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("prepare")
    prep.add_argument("--root", type=Path, required=True)
    prep.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    prep.add_argument("--steps", type=int, default=1_000_000)
    prep.add_argument("--curriculum-steps", type=int, default=1_000_000)
    prep.add_argument("--cars", type=int, nargs="+", default=[3, 5, 8, 10])
    prep.add_argument("--per-family", type=int, default=25)
    prep.add_argument("--validation-per-family", type=int, default=10)
    train = commands.add_parser("train")
    train.add_argument("--variants", nargs="+", choices=VARIANTS, default=list(VARIANTS))
    train.add_argument("--device", default="cpu")
    evaluate = commands.add_parser("evaluate")
    evaluate.add_argument("--methods", nargs="+", choices=METHODS, default=list(DEFAULT_METHODS))
    evaluate.add_argument("--cases", nargs="+")
    evaluate.add_argument("--plants", nargs="+", choices=PLANTS, default=["kinematic"])
    for command in (train, evaluate):
        command.add_argument("--root", type=Path, required=True)
        command.add_argument("--seeds", nargs="+", type=int)
        command.add_argument("--workers", type=int, default=1)
        command.add_argument("--resume", action="store_true")
        command.add_argument("--dry-run", action="store_true")
    tune = commands.add_parser("tune-baseline")
    tune.add_argument("--root", type=Path, required=True)
    tune.add_argument("--resume", action="store_true")
    args = p.parse_args()
    if args.command == "prepare":
        return prepare(args)
    root = args.root.resolve()
    protocol = load_study(root)
    if args.command == "tune-baseline":
        command = [sys.executable, "-m", "arrival_final.tune_baseline", "--manifest",
                   str(root / protocol["validation_manifest"]), "--output", str(root / "baseline_parameters.json")]
        if args.resume:
            command += ["--resume"]
        return launch(root, [("tune_baseline", command)])
    seeds = args.seeds if args.seeds is not None else protocol["seeds"]
    jobs = (training_jobs(root, protocol, args.variants, seeds, args.device, args.resume)
            if args.command == "train" else evaluation_jobs(root, protocol, args.methods, seeds,
                args.cases or list(protocol["cases"]), args.plants, args.resume))
    launch(root, jobs, args.workers, args.dry_run)


if __name__ == "__main__":
    main()
