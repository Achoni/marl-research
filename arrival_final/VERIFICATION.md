# Local implementation verification

Verified on 2026-10-09 using Python 3.10.13, NumPy 2.2.5, SciPy 1.15.3,
CasADi 3.7.2, Torch 2.11.0 and MuJoCo 3.4.0.

- All **68 regression tests passed**, including the optional headless MuJoCo tests.
- A copy containing only `arrival_final/` passed the same suite outside the repository.
- Default preparation generated all **825 test missions** and 40 baseline-validation
  missions, including the fixed-arena and density-scaled 10-car cases.
- The job planner produced the intended nine independent 1,000,000-step commands.
- Three actual eight-step training runs completed: attention+PP, attention+MPC,
  and mean-pooling+MPC, with validation and saved/reloaded checkpoints.
- All six primary evaluation methods executed on a shared saved smoke mission.
- A frozen smoke actor executed a complete headless MuJoCo episode.
- Baseline parameter selection completed its four trials on a separate one-mission
  validation smoke manifest.
- Report generation, saved-trajectory comparison plots, validation learning curves,
  and Matplotlib viewer drawing executed successfully. The trajectory plot was
  visually inspected.

The smoke runs used tiny budgets and reduced validation. They verify execution,
not learning quality, transfer quality, runtime at scale, or research conclusions.
They are stored outside this package and must not be used as paper results. Full
million-step training and the full benchmark matrix have **not** been run locally.

SLSQP emitted its ordinary internal bound-clipping warning during some smoke
solves. The controller still checks returned solutions for finite values, bounds,
and sampled separation before executing them, with fallback recorded separately.
