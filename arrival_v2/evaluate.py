"""Headless kinematic evaluation of a new-format checkpoint; no training updates."""
import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    __package__ = Path(__file__).resolve().parent.name

import torch

from .config import EnvConfig
from .mappo import MAPPOAgent
from .train import evaluate, source_hashes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=2_000_000_000)
    parser.add_argument("--cars", type=int, help="Override car count, without retraining")
    parser.add_argument("--neighbors", type=int, help="Override actor K; weights do not depend on K")
    parser.add_argument("--controller", choices=("pp", "mpc"), help="Explicit frozen-policy controller ablation")
    parser.add_argument("--distribution", choices=("target", "training"), default="target",
                        help="Use final target distribution or the saved current curriculum stage")
    parser.add_argument("--stratified", action=argparse.BooleanOptionalAction, default=True,
                        help="Allocate mixed evaluation episodes across each family")
    parser.add_argument("--family", choices=("mixed", "random", "crossing", "converging", "unequal", "parallel"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--torch-threads", type=int, default=1)
    args = parser.parse_args()
    if args.episodes < 1 or args.torch_threads < 1 or args.seed < 0:
        parser.error("episodes/torch-threads must be positive; seed must be nonnegative")
    if args.output.exists():
        parser.error("output already exists; choose a fresh JSON path")
    torch.set_num_threads(args.torch_threads)
    agent, checkpoint = MAPPOAgent.from_checkpoint(args.checkpoint, device=args.device)
    saved_config = checkpoint.get("active_env_config", checkpoint["env_config"]) if args.distribution == "training" else checkpoint["env_config"]
    cfg = EnvConfig(**saved_config)
    overrides = {key: value for key, value in {
        "n_cars": args.cars, "max_neighbors": args.neighbors, "family": args.family,
        "controller": args.controller,
    }.items() if value is not None}
    cfg = replace(cfg, **overrides).validate()
    metrics, episodes = evaluate(agent, cfg, args.episodes, seed_base=args.seed, stratified=args.stratified)
    result = dict(checkpoint=str(Path(args.checkpoint).resolve()),
                  checkpoint_sha256=hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest(),
                  trained_steps=checkpoint.get("global_step"), env_config=cfg.to_dict(),
                  training_env_config=checkpoint["env_config"], overrides=overrides,
                  distribution=args.distribution, stratified=args.stratified,
                  evaluation_source_hashes=source_hashes(),
                  training_source_hashes=checkpoint.get("source_hashes"),
                  evaluation_seed=args.seed, metrics=metrics, episodes=episodes)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(metrics, indent=2, allow_nan=False))
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
