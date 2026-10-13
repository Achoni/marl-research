"""Select lattice parameters on a validation manifest only, never final tests."""
import argparse
from dataclasses import asdict
from pathlib import Path

import torch

from .artifacts import read_json, write_json
from .baselines import LatticeConfig
from .benchmark import run_benchmark
from .missions import load_manifest
from .train import selection_score


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()
    manifest = load_manifest(args.manifest)
    if manifest["split"] != "validation":
        p.error("Baseline tuning is restricted to validation missions")
    if args.output.exists():
        saved = read_json(args.output)
        if args.resume and saved.get("validation_manifest_id") == manifest["manifest_id"]:
            print(f"Already selected: {args.output}")
            return
        p.error("Parameters already selected; --resume requires the same validation manifest")
    torch.set_num_threads(1)
    directory = args.output.parent / (args.output.stem + "_validation")
    trials = []
    for schedule in (4.0, 12.0):
        for gain in (0.95, 1.15):
            parameters = LatticeConfig(schedule_weight=schedule, pacing_gain=gain)
            result = run_benchmark(args.manifest, "lattice_mpc", directory / f"trial_{len(trials)}.json",
                baseline_parameters=parameters, resume=args.resume)
            trials.append(dict(parameters=asdict(parameters), metrics=result["metrics"]))
    best = max(trials, key=lambda trial: selection_score(trial["metrics"]))
    write_json(args.output, dict(parameters=best["parameters"], validation_manifest_id=manifest["manifest_id"],
                                 selection_rule="same lexicographic validation objective as learned policies",
                                 trials=trials))
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
