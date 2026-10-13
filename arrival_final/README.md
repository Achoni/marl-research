# Final simultaneous-arrival experiment

This directory is self-contained: copy **only `arrival_final/`** to the remote PC.
Run commands from its parent directory. It does not import `arrival_v2`, `deadline3`,
`11final`, or your rewritten `src` implementation.

It provides the implementation and experiment protocol. It does **not** contain
trained final models or establish success, safety, superiority, or scalability.
The local smoke checkpoints are deliberately untrained and are not included here.

## 1. Remote setup

Example from your Mac (replace `USER@HOST`):

```bash
ssh USER@HOST 'mkdir -p ~/arrival_project'
rsync -av --exclude='__pycache__' --exclude='experiments' --exclude='runs' \
  arrival_final/ USER@HOST:~/arrival_project/arrival_final/
ssh USER@HOST
```

Then on the remote PC:

```bash
cd ~/arrival_project
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r arrival_final/requirements.txt
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python -m unittest discover -s arrival_final/tests -t . -v
python -m arrival_final.smoke --output smoke_check
```

Use Python 3.10 or newer with compatible wheels. The dependency pins are the
versions verified locally. For a CPU-only Linux installation, you can install
`torch==2.11.0` from PyTorch's CPU wheel index before installing the requirements:

```bash
python -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cpu
```

MuJoCo is optional for training. Install and check it before transfer evaluation:

```bash
python -m pip install -r arrival_final/requirements-mujoco.txt
python -m unittest arrival_final.tests.test_final_pipeline.MujocoTransferTests -v
```

The smoke command trains each variant for **eight steps** and evaluates one mission.
It verifies execution only; ignore its success rates. Use a fresh output directory
for another smoke run, or add `--mujoco` to include a headless transfer rollout.
No desktop, renderer, or display is needed for training or headless evaluation.

## 2. Prepare the final study once

```bash
python -m arrival_final.study prepare --root arrival_final/experiments/paper
```

This saves the protocol, source snapshot, validation missions, and **825 actual
test missions**. Defaults are three training seeds, 1,000,000 steps per run, and
three training variants. Change these only when preparing a new study, e.g.:

```bash
python -m arrival_final.study prepare --root arrival_final/experiments/paper_5seeds \
  --seeds 0 1 2 3 4 --steps 1000000 --per-family 50
```

The default mixture is random/crossing/converging/parallel in proportions
`0.45/0.20/0.15/0.05`, normalized to sum to one. The deliberately extreme `unequal`
family is excluded from training and the primary test; it has its own held-out
stress test. **Random missions still have unequal trip lengths.** This restriction
must be stated in the paper, rather than described as universally fairer sampling.

Do not edit package `.py` files after preparing the study. Source hashes prevent
mixing implementations. For a change, make a separate version and prepare a new
study. Checkpoints from older folders are intentionally incompatible (schema 3).

## 3. Train nine independent runs

Run this inside `tmux` or another persistent remote shell so disconnecting SSH
does not terminate the process:

```bash
python -m arrival_final.study train --root arrival_final/experiments/paper --workers 3
```

| Run names | Actor | Controller used during training | Seeds |
|---|---|---|---|
| `attention_pp_s*` | Masked attention | PP reference controls | 0, 1, 2 |
| `attention_mpc_s*` | Masked attention | Constrained NMPC | 0, 1, 2 |
| `mean_mpc_s*` | Masked mean pooling | Same constrained NMPC | 0, 1, 2 |

**Each run gets 1,000,000 joint environment steps**, not 1,000,000 divided across
variants or seeds. With three cars, one joint step contains three agent actions.
This is nine million joint steps in total. Every seed starts with fresh model
weights, optimizer, and training mission stream.

`--workers 3` runs three independent processes concurrently; it is not three
parallel environments inside one MAPPO update. Use `--workers 1` on a smaller PC.
CPU is the default: MPC's SLSQP solves are CPU work, so a GPU does not accelerate
that main cost. `--device cuda` is available for the networks if your Torch setup
supports it. Measure actual throughput before estimating completion time; no
fixed training-time estimate is assumed.

Useful operations:

```bash
# Inspect commands without starting training.
python -m arrival_final.study train --root arrival_final/experiments/paper --dry-run

# Train a subset first.
python -m arrival_final.study train --root arrival_final/experiments/paper \
  --variants attention_pp --seeds 0 --workers 1

# Continue interrupted runs and skip completed runs.
python -m arrival_final.study train --root arrival_final/experiments/paper \
  --workers 3 --resume

# Follow progress in another SSH shell.
tail -f arrival_final/experiments/paper/logs/attention_mpc_s0.log
```

Each run saves `latest.pt` after every PPO update, `best.pt` selected only on
validation, periodic checkpoints, source/configuration, per-episode diagnostics,
and `metrics.jsonl`. Resume restores weights, optimizer and RNG state, but starts
a fresh episode and discards a partial rollout: it is not a bitwise continuation
of the interrupted physical trajectory. Do not launch two processes for the same
run directory.

The curriculum broadens at 350k, 600k and 800k steps. Validation always uses the
full target distribution. The final 200k steps train on that distribution. This
is a training budget, not a promise of convergence.

## 4. Tune the non-learning comparator on validation

```bash
python -m arrival_final.study tune-baseline --root arrival_final/experiments/paper
```

The trajectory-lattice baseline evaluates local PP/bicycle candidate references
using deadline progress and neighbour forecasts, then uses the same MPC. Four
fixed schedule-weight/pacing-gain combinations are evaluated on **40 separate
validation missions**. The same lexicographic selection criterion used for model
checkpoints picks the parameters, saved in `baseline_parameters.json`. The test
set is never used for this selection. Interrupted tuning supports `--resume`.

This is a documented local heuristic comparator, not a state-of-the-art or globally
optimal planner. The simpler direct goal/time rule is also retained. Beating these
comparators alone would not establish superiority over the wider literature.

## 5. Primary comparison on identical missions

```bash
python -m arrival_final.study evaluate --root arrival_final/experiments/paper \
  --cases target_3 --workers 1 --resume
```

Default methods:

| Label | Trained actor | Execution |
|---|---|---|
| `attention_pp` | PP-trained attention | PP |
| `attention_pp_mpc` | **The exact same PP checkpoint** | MPC added after training |
| `attention_mpc` | MPC-trained attention | MPC |
| `mean_mpc` | MPC-trained mean pooling | MPC |
| `goal_mpc` | No RL; goal/time rule | MPC |
| `lattice_mpc` | No RL; local trajectory lattice | MPC |

All methods replay the same stored positions, headings, speeds, goals and deadlines.
Learned methods use all three training seeds. Deterministic non-learning baselines
run once per mission, not as three fake independent seeds. Results are resumable
per episode. Do not change a checkpoint or baseline parameters halfway through a
result file. Use one evaluation worker for comparable runtime measurements; more
workers can improve throughput but confound latency measurements through contention.

## 6. Zero-shot scalability and stress tests

```bash
python -m arrival_final.study evaluate --root arrival_final/experiments/paper \
  --cases fixed_3 fixed_5 fixed_8 fixed_10 density_5 density_8 density_10 \
  --methods attention_mpc mean_mpc lattice_mpc --workers 1 --resume

python -m arrival_final.study evaluate --root arrival_final/experiments/paper \
  --cases unequal_3 --workers 1 --resume
```

Every actor was trained with three cars. No retraining occurs for 5, 8 or 10 cars.
`fixed_*` uses a sampling square of half-width 6 m. `density_*` scales half-width
as `6 sqrt(N/3)` m. `fixed_3` is the common reference for both studies. These test
increasing crowding and approximately constant spatial density separately;
interaction geometry still changes with team size. Team success is stricter for
larger teams, so arrival fraction and per-car outcomes are retained too.

KNN limits actor inputs to four visible neighbours within 5 m, with a mask.
Attention and mean pooling share this selection. The **MPC uses all locally sensed
neighbours**, with `N-1` allocated slots. Consequently a fixed-size actor does not
imply constant MPC computation or a guarantee of zero-shot performance. Since
training has only three cars, it exposes at most two neighbours per actor.

## 7. Frozen-policy transfer

```bash
python -m arrival_final.study evaluate --root arrival_final/experiments/paper \
  --cases target_3 --plants dynamic mujoco \
  --methods attention_pp attention_pp_mpc attention_mpc mean_mpc goal_mpc lattice_mpc \
  --workers 1 --resume
```

Additional predefined stress plants are `dynamic_low_friction`, `dynamic_lag`, and
`mujoco_low_friction`. Choose these explicitly rather than silently changing plant
parameters. Start with the three-car transfer test before a larger transfer matrix.

The MuJoCo adapter initializes requested pose, velocity, steering and wheel spin;
reads actual state; checks planar stopping speed; and latches inter-car physical
contacts at physics steps. Each process has its own generated scene. The scene is
bundled with no external assets and has no static walls or obstacles, consistent
with the nominal task. There is an actuator bridge from acceleration/steering rate
to drive torque/steering servo. This bridge is part of the declared controller and
must be described in transfer results. It is not retrained online.

All plants use the same conservative footprint collision criterion, with MuJoCo
physical contacts as an additional failure criterion. A footprint collision can
terminate before actual body contact; do not interpret a zero contact rate alone
as physical safety. Tyre/actuator parameters are simulation choices, not hardware
identification. The MPC prediction model remains kinematic in every plant.

## 8. Generate tables and figures

```bash
python -m arrival_final.report \
  --results arrival_final/experiments/paper/results \
  --runs arrival_final/experiments/paper/runs \
  --output arrival_final/experiments/paper/report
```

Outputs include `RESULTS.md`, `results.csv`, `summary.json`, main comparison,
scalability and validation-learning-curve figures in PNG/PDF when the corresponding
data exist. Training seed mean/std and 95% bootstrap intervals are separate. The
bootstrap resamples runs and paired missions within each scenario family; three
seeds give limited information about training variability. A single-run interval
is explicitly scenario-only. Paired comparisons with missing seed matches are
marked skipped. Incomplete experiment groups show their actual run count.

Family rates use the declared mixture, not raw counts from the balanced test set.
Success/collision/timeout use all episodes. Timing error and arrival spread are
conditional on all-car completion, including timing failures; they are never
silently restricted to successes. Report both completion and timing errors.

Every benchmark saves compressed trajectory traces at policy decision times:

```bash
python -m arrival_final.plot_episode --episode 0 \
  --results \
  arrival_final/experiments/paper/results/target_3__kinematic__attention_pp_s0.json \
  arrival_final/experiments/paper/results/target_3__kinematic__attention_pp_mpc_s0.json \
  arrival_final/experiments/paper/results/target_3__kinematic__attention_mpc_s0.json \
  --output arrival_final/experiments/paper/report/paired_episode_0
```

This plots paths, goal distance and speed for one identical mission. Use examples
from successes and failures; outcome captions are generated from saved data.
5 Hz plotted trajectories are for visualization; collision checks use the finer
physics steps. Runtime diagnostics include per-solver, sequential team-control,
actor and entire simulation-step timings, with p95/max and overrun rates. These
are wall-clock measurements on the recorded machine, not distributed-hardware
latency or a real-time guarantee.

## 9. Watch a checkpoint on your desktop

Copy back the whole study or its run directory and use this same package version:

```bash
python -m arrival_final.view \
  --checkpoint arrival_final/experiments/paper/runs/attention_mpc_s0/best.pt --speed 0.5

# macOS native MuJoCo viewer:
mjpython -m arrival_final.view_mujoco \
  --checkpoint arrival_final/experiments/paper/runs/attention_mpc_s0/best.pt --speed 0.5
```

Space pauses, N draws another scenario, R replays in the Matplotlib viewer. Viewer
runs are exploratory; use saved benchmark missions for paper metrics. An ordinary
SSH shell does not provide a desktop window.

## Implementation map and definitions

See [PROTOCOL.md](PROTOCOL.md) for the equations, reward, success criteria and limits.

| File | Responsibility |
|---|---|
| `config.py`, `scenarios.py`, `curriculum.py` | Task constants, sampling and training schedule |
| `model.py`, `reference.py` | Bicycle dynamics, footprint and timed PP references |
| `mpc.py` | Constrained single-shooting NMPC with CasADi derivatives and SciPy SLSQP |
| `env.py` | Local sensing, nested clocks, physical execution, reward/events |
| `mappo.py`, `train.py` | Shared attention/mean actor, centralized critic, PPO and checkpoints |
| `plants.py`, `scene.py` | Independent kinematic/dynamic/MuJoCo plants |
| `missions.py`, `study.py` | Frozen missions and independent job commands |
| `baselines.py`, `tune_baseline.py` | Local non-learning comparators and validation selection |
| `benchmark.py`, `report.py`, `plot_episode.py` | Paired evaluation, uncertainty and figures |

The main change from your earlier trained folder is experimental reproducibility,
plus a tyre-inclusive footprint and full horizon-node separation checks. These
changes can alter performance. Retrain; do not reuse an earlier success rate as
an `arrival_final` result. Deadlocks and timeouts remain possible and are measured.
