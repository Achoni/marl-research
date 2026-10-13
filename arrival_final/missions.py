"""Save actual paired missions, rather than relying only on sampler seeds."""
from dataclasses import replace
import argparse
from pathlib import Path

import numpy as np

from .artifacts import digest, read_json, write_json
from .config import EnvConfig
from .scenarios import Scenario, ScenarioSampler
from .train import source_hashes, validation_families


def scenario_from_record(record):
    return Scenario(np.asarray(record["starts"], dtype=float),
                    np.asarray(record["goals"], dtype=float), float(record["deadline"]),
                    record["family"], float(record["arena_half"]), record["metadata"])


def build_manifest(config, per_family=25, seed=2_100_000_000, split="test", name="target_3"):
    if per_family < 1 or split not in ("validation", "test"):
        raise ValueError("positive per-family count and validation/test split required")
    if not ((1_000_000_000 <= seed < 2_000_000_000) if split == "validation"
            else (2_000_000_000 <= seed)):
        raise ValueError("Validation seeds must be in [1e9,2e9); test seeds must be >=2e9")
    weights = validation_families(config)
    records = []
    for family_index, family in enumerate(sorted(weights)):
        sampler = ScenarioSampler(replace(config, family=family))
        for episode in range(per_family):
            scenario_seed = seed + family_index * 1_000_000 + episode
            if split == "validation" and scenario_seed >= 2_000_000_000:
                raise ValueError("Validation seed range overlaps the test range")
            sampler.seed(scenario_seed)
            sc = sampler.sample()  # Failure is explicit; never silently drop hard samples.
            record = dict(seed=scenario_seed, starts=sc.starts.tolist(), goals=sc.goals.tolist(),
                          deadline=sc.deadline, family=family, arena_half=sc.arena_half,
                          metadata=sc.metadata)
            record["mission_id"] = digest(record)
            records.append(record)
    manifest = dict(kind="arrival_final_missions", schema_version=1, name=name, split=split,
                    env_config=config.to_dict(), family_weights=weights,
                    source_hashes=source_hashes(), missions=records)
    manifest["manifest_id"] = digest(manifest)
    return manifest


def load_manifest(path):
    manifest = read_json(path)
    payload = {k: v for k, v in manifest.items() if k != "manifest_id"}
    if manifest.get("kind") != "arrival_final_missions" or digest(payload) != manifest.get("manifest_id"):
        raise ValueError("Invalid or modified mission manifest")
    seen = set()
    for row in manifest["missions"]:
        if row["mission_id"] in seen or digest({k: v for k, v in row.items()
                                              if k != "mission_id"}) != row["mission_id"]:
            raise ValueError("Duplicate or modified mission")
        seen.add(row["mission_id"])
    return manifest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--cars", type=int, default=3)
    p.add_argument("--per-family", type=int, default=25)
    p.add_argument("--seed", type=int, default=2_100_000_000)
    p.add_argument("--family", default="mixed")
    p.add_argument("--split", choices=("validation", "test"), default="test")
    args = p.parse_args()
    if args.output.exists():
        p.error("Choose a fresh manifest path")
    write_json(args.output, build_manifest(EnvConfig(n_cars=args.cars, family=args.family),
        args.per_family, args.seed, args.split, args.output.stem))


if __name__ == "__main__":
    main()
