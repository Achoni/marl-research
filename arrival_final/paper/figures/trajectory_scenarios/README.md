# Six paired trajectory illustrations

Each numbered PNG shows independent closed-loop executions of the frozen
`runs/mpc_s0/best.pt` actor with pure-pursuit references and NMPC: the kinematic
bicycle plant on the left and the actual MuJoCo contact-dynamics plant on the right.
The checkpoint was saved at 860,000 training steps. No training was performed.

The scenes are random goals, a three-way crossing, converging routes, parallel
journeys, a five-way crossing, and an eight-way crossing. The six seeds were fixed
before executing the controller; no successful-episode screening was performed.
All twelve recorded runs met the environment's synchronization success criterion.
These are illustrations, not statistical evidence of reliability or scalability.

## Visual conventions

- Each car and its assigned goal use the same colour in both views.
- Lines follow recorded rear-axle positions, sampled every policy decision.
- Translucent cars are earlier recorded poses, sampled along distance travelled.
  Opacity increases with progress; the snapshots are not equally spaced in time.
- Opaque cars show terminal poses. S labels the start, G the assigned goal;
  dashed rings show the 0.25 m goal tolerance.
- Overlapping paths or ghost cars do not imply a collision: the positions can
  have been occupied at different times.
- Each panel reports its own environment outcome and arrival-time spread.

PNG figures are 3168 x 1826 pixels. SVG copies keep the 2D panel and typography
as vectors and embed the native MuJoCo raster. `overview.png` is a contact sheet.

## Provenance and limits

`manifest.json` records the checkpoint hash, runtime source hashes, seeds,
configuration, outcomes and pairing details. `data/` contains actual trajectories
and, for MuJoCo, recorded generalized coordinates needed for replay. MuJoCo ghosts
are decorative render geometry; they did not take part in physics.

The runtime is the existing `arrival_v2` code, as used by its viewers. In particular,
the checkpoint predates some collision-geometry settings. Runtime defaults absent
from that checkpoint (including the two-disc footprint and constraint stride) are
explicitly recorded in the manifest. These images do not claim to be a benchmark
of the newer `arrival_final` implementation or the exact original training snapshot.

Initial speeds are zero in both plants so initial conditions match physically.
All other task/controller settings use the checkpoint and current runtime defaults,
apart from the selected scenario family and team size. No moving external obstacles
were included. Outcomes use the existing environment collision metric, not a new
independent physical-contact audit.

`speed_summary.json` reports policy-step samples of actual speed for these episodes,
including mean and 95th percentile for samples above 0.08 m/s. The 2 m/s speed cap
does not mean the cars actually travel at 2 m/s.

## Reproduce

From the repository root, in the existing research Python environment:

```bash
python3 arrival_final/paper/figures/trajectory_scenarios/generate.py --collect-only
python3 arrival_final/paper/figures/trajectory_scenarios/generate.py --render-only
```

The second command requires macOS graphics access. It only renders stored runs.
The first command replays the same six cases and replaces this folder's recorded data.
