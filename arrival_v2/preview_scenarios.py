"""Inspect broad start/target sampling without training or importing an MPC."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys
import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    __package__ = Path(__file__).resolve().parent.name

from .config import EnvConfig
from .scenarios import ScenarioSampler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cars", type=int, default=3)
    parser.add_argument("--family", default="mixed", choices=("mixed", "random", "crossing", "converging", "unequal", "parallel"))
    parser.add_argument("--output", type=Path, default=Path("scenario_preview.png"))
    args = parser.parse_args()
    if args.episodes < 1:
        parser.error("episodes must be positive")
    if args.output.exists() or args.output.with_suffix(".json").exists():
        parser.error("preview output exists; choose a fresh path")
    cfg = EnvConfig(n_cars=args.cars, family=args.family).validate()
    sampler = ScenarioSampler(cfg, seed=args.seed)
    scenarios = [sampler.sample() for _ in range(args.episodes)]
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cols = min(4, args.episodes)
    rows = (args.episodes + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows), squeeze=False)
    colors = plt.get_cmap("tab10")
    for ax, sc in zip(axes.ravel(), scenarios):
        for i in range(args.cars):
            color = colors(i % 10)
            start, goal = sc.starts[i, :2], sc.goals[i]
            ax.plot([start[0], goal[0]], [start[1], goal[1]], "--", color=color, alpha=0.35)
            ax.add_patch(plt.Circle(start, cfg.footprint_radius, fill=False, color=color))
            ax.scatter(*goal, marker="*", s=95, color=color)
            direction = np.array([np.cos(sc.starts[i, 2]), np.sin(sc.starts[i, 2])]) * 0.6
            ax.arrow(*start, *direction, color=color, head_width=0.15, length_includes_head=True)
        ax.set(xlim=(-sc.arena_half - 0.6, sc.arena_half + 0.6),
               ylim=(-sc.arena_half - 0.6, sc.arena_half + 0.6),
               title=f"{sc.family} | T={sc.deadline:.1f}s", aspect="equal")
        ax.grid(alpha=0.2)
    for ax in axes.ravel()[args.episodes:]:
        ax.set_visible(False)
    fig.suptitle("Random starts / goals — dashed lines are assignments, not driven paths")
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=130)
    plt.close(fig)
    data = [dict(starts=s.starts.tolist(), goals=s.goals.tolist(), deadline=s.deadline,
                 family=s.family, arena_half=s.arena_half, metadata=s.metadata) for s in scenarios]
    args.output.with_suffix(".json").write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    print(dict(Counter(s.family for s in scenarios)))
    print(args.output)


if __name__ == "__main__":
    main()
