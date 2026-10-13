"""Paper tables/figures with separate training-seed and scenario uncertainty."""
import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path

import numpy as np

from .artifacts import read_json, write_json


def success_matrix(results):
    ids = [e["mission_id"] for e in results[0]["episodes"]]
    if any([e["mission_id"] for e in r["episodes"]] != ids for r in results):
        raise ValueError("Results are not paired on identical ordered missions")
    return np.asarray([[e["success"] for e in r["episodes"]] for r in results], dtype=float)


def bootstrap(matrix, reference, replicates=2000, seed=104729):
    """Resample training runs and paired missions within each family.

    For a difference matrix the SAME selected runs/missions are used for both
    methods. With one training run the interval is scenario uncertainty only.
    """
    rng = np.random.default_rng(seed)
    draws = np.zeros(replicates)
    for b in range(replicates):
        runs = rng.integers(0, len(matrix), len(matrix))
        for family, weight in reference["family_weights"].items():
            indices = np.array([i for i, row in enumerate(reference["episodes"]) if row["family"] == family])
            sample = rng.choice(indices, len(indices), replace=True)
            draws[b] += weight * float(matrix[runs][:, sample].mean())
    return np.quantile(draws, [.025, .975]).tolist()


def summarize(paths, output, replicates=2000):
    if replicates < 1:
        raise ValueError("bootstrap count must be positive")
    groups = defaultdict(list)
    seen = set()
    for path in paths:
        r = read_json(path)
        if r.get("kind") != "arrival_final_result" or r["split"] != "test" or r["job_id"] in seen:
            continue
        seen.add(r["job_id"])
        groups[(r["case"], r["plant"], r["method"])].append(r)
    if not groups:
        raise ValueError("No completed test benchmarks found")
    table = []
    for (case, plant, method), results in sorted(groups.items()):
        results.sort(key=lambda r: -1 if r["training_seed"] is None else r["training_seed"])
        seeds = [r["training_seed"] for r in results]
        if len(set(seeds)) != len(seeds):
            raise ValueError("Multiple checkpoints for the same method/seed/condition; select one protocol")
        reference = results[0]
        if any(r["manifest_id"] != reference["manifest_id"] or r["source_hashes"] != reference["source_hashes"]
               or r["baseline_parameters"] != reference["baseline_parameters"] for r in results):
            raise ValueError("Cannot aggregate different protocols or baseline parameters")
        matrix = success_matrix(results)
        rates = [r["metrics"]["success_rate"] for r in results]
        row = dict(case=case, plant=plant, method=method, cars=reference["env_config"]["n_cars"],
                   training_seeds=seeds, runs=len(results), missions_per_run=len(reference["episodes"]),
                   success_mean=float(np.mean(rates)),
                   success_seed_std=float(np.std(rates, ddof=1)) if len(rates) > 1 else None,
                   success_ci95=bootstrap(matrix, reference, replicates),
                   interval_scope="training runs and scenarios" if len(rates) > 1 else "scenarios only; fixed controller",
                   per_seed_success=rates)
        for field in ("collision_rate", "physical_collision_rate", "timeout_rate", "all_arrived_rate",
                      "arrival_spread_mean_all_arrived", "max_deadline_error_mean_all_arrived", "fallback_rate_mean"):
            values = [r["metrics"][field] for r in results if r["metrics"][field] is not None]
            row[field] = float(np.mean(values)) if values else None
        row["solver_p95_seconds"] = [r["metrics"]["timing_seconds"]["solver"]["p95"] for r in results]
        table.append(row)
    comparisons = []
    pairs = [("attention_mpc", "lattice_mpc"), ("attention_mpc", "goal_mpc"),
             ("attention_mpc", "attention_pp_mpc"), ("attention_pp_mpc", "attention_pp"),
             ("attention_mpc", "mean_mpc")]
    for case, plant in sorted({(a, b) for a, b, _ in groups}):
        for a, b in pairs:
            aa, bb = groups.get((case, plant, a)), groups.get((case, plant, b))
            if not aa or not bb:
                continue
            if aa[0]["manifest_id"] != bb[0]["manifest_id"] or aa[0]["source_hashes"] != bb[0]["source_hashes"]:
                raise ValueError("Comparison methods have different test missions or source")
            ma, mb = success_matrix(aa), success_matrix(bb)
            success_matrix([aa[0], bb[0]])
            if bb[0]["training_seed"] is None:
                mb = np.repeat(mb, len(ma), axis=0)
            elif [r["training_seed"] for r in aa] != [r["training_seed"] for r in bb]:
                comparisons.append(dict(case=case, plant=plant, a=a, b=b,
                                        skipped="training seed sets differ; finish the missing runs"))
                continue
            differences = ma - mb
            mean = sum(w * float(differences[:, [i for i, e in enumerate(aa[0]["episodes"])
                                                if e["family"] == f]].mean())
                       for f, w in aa[0]["family_weights"].items())
            comparisons.append(dict(case=case, plant=plant, a=a, b=b, success_difference=mean,
                                    ci95=bootstrap(differences, aa[0], replicates)))
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "summary.json", dict(results=table, paired_comparisons=comparisons,
        bootstrap_replicates=replicates, caveat="No safety/scalability guarantee. Seed std is unavailable for one run."))
    with (output / "results.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)
    lines = ["# Final experiment results", "", "Rates use each manifest's declared family mixture.",
             "Timing errors are conditional on all-car completion; runs receive equal weight.",
             "Intervals with one run cover scenario sampling only.", "",
             "| Case | Plant | Method | Runs | Success (95% CI) | Collision | Timeout |",
             "|---|---|---|---:|---:|---:|---:|"]
    for row in table:
        lo, hi = row["success_ci95"]
        lines.append(f"| {row['case']} | {row['plant']} | {row['method']} | {row['runs']} | "
                     f"{row['success_mean']:.1%} ({lo:.1%}–{hi:.1%}) | {row['collision_rate']:.1%} | {row['timeout_rate']:.1%} |")
    (output / "RESULTS.md").write_text("\n".join(lines) + "\n")
    make_plots(table, output)
    return table


def make_plots(rows, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    main = [r for r in rows if r["case"] == "target_3" and r["plant"] == "kinematic"]
    if main:
        fig, ax = plt.subplots(figsize=(10, 4.8))
        x = np.arange(len(main))
        for i, row in enumerate(main):
            # Percentile intervals need not contain the point estimate; draw directly.
            ax.bar(i, 100 * row["success_mean"], color="#4d79aa")
            lo, hi = np.array(row["success_ci95"]) * 100
            ax.vlines(i, lo, hi, color="black", linewidth=1.5)
            ax.scatter(np.full(row["runs"], i), np.array(row["per_seed_success"]) * 100,
                       color="#ce6b25", s=24, zorder=3)
        ax.set_xticks(x, [r["method"].replace("_", "\n") for r in main])
        ax.set(ylabel="Strict team success (%)", ylim=(0, 105),
               title="Three-car target distribution; points are independent training runs")
        fig.tight_layout()
        for ext in ("png", "pdf"):
            fig.savefig(output / f"main_results.{ext}", dpi=180)
        plt.close(fig)
    scale = [r for r in rows if r["plant"] == "kinematic" and r["case"].startswith(("fixed_", "density_"))]
    if scale:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
        for ax, mode in zip(axes, ("fixed", "density")):
            for method in sorted({r["method"] for r in scale}):
                series = sorted([r for r in scale if r["method"] == method and
                    (r["case"].startswith(mode + "_") or (mode == "density" and r["case"] == "fixed_3"))],
                    key=lambda r: r["cars"])
                if not series:
                    continue
                x = [r["cars"] for r in series]
                ax.plot(x, [100*r["success_mean"] for r in series], "o-", label=method)
                ax.fill_between(x, [100*r["success_ci95"][0] for r in series],
                                [100*r["success_ci95"][1] for r in series], alpha=.12)
            ax.set(title="Fixed arena" if mode == "fixed" else "Area scaled with team size",
                   xlabel="Number of cars", ylim=(0, 105), ylabel="Strict success (%)")
        axes[-1].legend(fontsize=7)
        fig.tight_layout()
        for ext in ("png", "pdf"):
            fig.savefig(output / f"scaling.{ext}", dpi=180)
        plt.close(fig)


def learning_curves(runs, output):
    """Validation on a fixed target distribution, grouped by independent seeds."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    series = defaultdict(list)
    for path in sorted(Path(runs).glob("*_s*/metrics.jsonl")):
        variant = path.parent.name.rsplit("_s", 1)[0]
        records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        if records:
            series[variant].append({r["global_step"]: r["success_rate"] for r in records})
    if not series:
        return
    fig, ax = plt.subplots(figsize=(8, 4.6))
    for variant, records in sorted(series.items()):
        # No extrapolation or averaging three seeds with one late surviving run.
        steps = sorted(set.intersection(*(set(r) for r in records)))
        if not steps:
            continue
        values = np.array([[r[step] for step in steps] for r in records]) * 100
        mean = values.mean(axis=0)
        ax.plot(steps, mean, label=f"{variant} ({len(records)} seeds)")
        if len(records) > 1:
            std = values.std(axis=0, ddof=1)
            ax.fill_between(steps, np.maximum(0, mean - std), np.minimum(100, mean + std), alpha=.15)
    ax.set(xlabel="Joint training steps", ylabel="Validation strict success (%)", ylim=(0, 105),
           title="Fixed target validation; bands show training-seed standard deviation")
    ax.legend()
    fig.tight_layout()
    for extension in ("png", "pdf"):
        fig.savefig(Path(output) / f"learning_curves.{extension}", dpi=180)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--bootstrap", type=int, default=2000)
    p.add_argument("--runs", type=Path, help="also plot validation learning curves")
    args = p.parse_args()
    rows = summarize(sorted(args.results.glob("*.json")), args.output, args.bootstrap)
    if args.runs:
        learning_curves(args.runs, args.output)
    print(f"Wrote {len(rows)} result groups to {args.output}")


if __name__ == "__main__":
    main()
