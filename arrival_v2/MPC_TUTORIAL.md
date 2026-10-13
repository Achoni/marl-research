# Implementing the NMPC yourself — a staged tutorial

This is written so you can throw away `arrival_v2/mpc.py` and rebuild it from
scratch. Every stage is runnable on its own and has a test that passes only when
that stage is physically correct. Numbers are your real `EnvConfig` defaults, not
textbook ones.

Companion files:

- `mpc_exercise.py` — the skeleton you fill in. Maths in the docstrings, bodies
  raising `NotImplementedError`.
- `tests/test_mpc_exercise.py` — run `pytest arrival_v2/tests/test_mpc_exercise.py`
  after each stage. Tests are grouped by stage and skip cleanly while a stage is
  still unimplemented.

---

## 0. The one idea

At every control tick, you are standing at a known state and you have a
reference saying where you should be for the next 0.8 s. MPC answers one
question:

> Of all the control sequences I could apply over the next 0.8 s, which one puts
> me closest to that reference without hitting anybody?

You solve that, **apply only the first control**, throw the rest away, and solve
again 0.05 s later from the new measured state. That last part is the whole
reason it works: solving over a horizon makes the controller anticipate, and
re-solving every tick makes it robust to the model being wrong.

Three things to be clear about before any code:

- **It is an optimisation problem, not a formula.** PID and pure pursuit compute
  a control. MPC *searches* for one. The code is therefore mostly about setting
  up a search, and only a little about vehicle dynamics.
- **The model is used for prediction only.** You never trust it beyond 0.8 s,
  and you re-measure every tick, so it is allowed to be the simple kinematic
  bicycle even though the plant has slip and actuator lag.
- **Feasibility beats optimality.** A slightly suboptimal control that respects
  the limits is fine. A beautiful optimum that commands 200°/s of steering is
  useless. This is why constraints are the hard part, not the cost.

---

## 1. State, control, and the decision variable

Your state is 5-dimensional:

```
x = [X, Y, psi, v, delta]        position, heading, speed, steering angle
```

Your control is 2-dimensional, and this is the choice that matters most:

```
u = [a, omega]                   acceleration (m/s^2), steering RATE (rad/s)
```

Commanding steering *rate* rather than steering *angle* is deliberate. The rack
on the real car cannot jump; it slews at a finite speed. With `max_steer_rate =
1.5708 rad/s` (90°/s) in the control bounds, the solver physically cannot ask
for a jump. If instead you made `delta` the control (as `mpc/mpc.py` does), the
solver would happily request 0° → 27° in one 0.05 s tick, the servo would refuse,
and the error would surface as unexplained tracking drift rather than as a
solver failure.

The discretised kinematic bicycle, with midpoint evaluation:

```
v_{k+1}     = clip(v_k + dt*a_k, 0, v_max)
delta_{k+1} = clip(delta_k + dt*omega_k, -delta_max, +delta_max)
v_mid       = (v_k + v_{k+1}) / 2
delta_mid   = (delta_k + delta_{k+1}) / 2
dpsi        = dt * v_mid * tan(delta_mid) / L
X_{k+1}     = X_k + dt * v_mid * cos(psi_k + dpsi/2)
Y_{k+1}     = Y_k + dt * v_mid * sin(psi_k + dpsi/2)
psi_{k+1}   = psi_k + dpsi
```

The `v_mid`, `delta_mid` and `psi_k + dpsi/2` are a midpoint (RK2) rule rather
than plain Euler. On a turn at 2 m/s over 0.05 s the difference is small but it
is systematic — Euler consistently cuts corners — and since the reference
generator uses the same integrator, matching it exactly means the MPC's
prediction of the reference trajectory is exact when there are no neighbours.
That property is worth having: it makes disagreement between PP and MPC
*meaningful* rather than an integration artefact.

### The decision variable

With `mpc_horizon = 0.8 s` and `control_dt = 0.05 s` you get **N = 16** steps.
You are solving for 16 controls, each 2-dimensional, so:

```
z = [a_0, omega_0, a_1, omega_1, ..., a_15, omega_15]   in R^32
```

**32 unknowns.** That is the entire problem. Everything else — the states, the
cost, the constraints — is a *function* of these 32 numbers.

### Single shooting vs multiple shooting

There are two ways to handle the states.

**Multiple shooting** makes the states decision variables too (`z` would hold
16×5 states as well, so 112 unknowns) and adds equality constraints
`x_{k+1} - f(x_k, u_k) = 0` to force them to be consistent. More variables, but
the constraints are sparse and it is numerically better conditioned on long
horizons. This is what `do_mpc` does by default.

**Single shooting** keeps only the controls as unknowns and *substitutes* the
dynamics: you roll the model forward in a Python loop, so `x_5` is already a
symbolic expression in `u_0..u_4`. Fewer variables, no equality constraints,
but the expression for `x_15` is a deep nest of trig and the sensitivity of the
end state to `u_0` can get large.

**This code uses single shooting**, and on a 16-step horizon that is the right
call: 32 variables instead of 112, no equality constraints to satisfy, and the
horizon is short enough that the conditioning problem never appears. On a
100-step horizon you would switch to multiple shooting.

The loop at [mpc.py:48-59](mpc.py#L48-L59) is exactly this substitution. Read
it again now and notice there is no state variable anywhere in `controls` — the
`state` variable is being *overwritten* each iteration with a bigger symbolic
expression.

> **Stage 1 exercise:** `rollout()` in `mpc_exercise.py`. Pure NumPy, no
> optimisation. Given a state and 16 controls, return the 17 states. Test checks
> it matches `model.bicycle_step` exactly and that bounds saturate.

---

## 2. The cost

Now write down what "closest to the reference" means. For each step k you have a
reference state `r_k = [X*, Y*, psi*, v*, delta*]` — **a different one for every
k**, because row k of the reference says where you should be at time
`t_0 + k*0.05`. This is what makes it a *trajectory tracker* and what makes the
deadline enforceable. (A setpoint tracker would use the same `r` at every k and
carry no timing information at all.)

The cost, straight from [mpc.py:63-66](mpc.py#L63-L66) and [81-82](mpc.py#L81-L82):

```
J(z) = sum_{k=1}^{16} [  8.0 * ||p_k - p*_k||^2          position
                       + 3.0 * (v_k - v*_k)^2            speed
                       + 0.5 * wrap(psi_k - psi*_k)^2    heading
                       + 0.1 * (delta_k - delta*_k)^2    steering
                       + 0.015 * ||u_k||^2               effort
                       + 0.04 * ||u_k - u_{k-1}||^2  ]   smoothness
     + 16.0 * ||p_16 - p*_16||^2                         terminal position
     + 6.0  * (v_16 - v*_16)^2                           terminal speed
```

Things to understand rather than copy:

**Why position dominates (8 vs 0.5).** Arrival is a position-and-time
requirement. Heading only matters as a means to get somewhere, so it gets a
small weight; forcing heading hard would fight the position term during
avoidance manoeuvres.

**Why the heading error is wrapped.** `psi - psi*` of 359° is really −1°.
`atan2(sin(e), cos(e))` maps the error to (−π, π]. Squaring the unwrapped
difference would make the controller spin a full turn to "correct" a 1° error.
Note `do_mpc`'s version uses `1 - cos(e)` instead, which is a smooth and
perfectly valid alternative — it is ≈ e²/2 for small e and needs no branch.

**Why the terminal weights are doubled (16 and 6).** This is a stand-in for a
terminal cost in the stability sense. Without it the controller is myopic: it
can let the end of the horizon drift badly because nothing penalises it heavily,
and since you re-solve every tick that drift never gets corrected. The classical
result is that a terminal cost equal to the infinite-horizon cost-to-go makes
receding-horizon control stably optimal; you do not have that function, so you
approximate it by weighting the last step harder. **This is the single most
common thing people leave out, and the symptom is a controller that tracks well
and then overshoots at the end.**

**Why `u` and `Δu` are penalised at all (0.015, 0.04).** Two reasons. Physically
it stops chattering between ticks. Numerically it makes the Hessian positive
definite in directions the tracking terms do not constrain — for instance when
`v = 0`, steering does nothing to position, so without these terms the cost is
completely flat in `omega` and the solver has no reason to pick any particular
value. `_previous` ([mpc.py:45](mpc.py#L45), [67](mpc.py#L67)) carries the
*actually executed* command across ticks so the Δu term couples consecutive
solves.

> **Stage 2 exercise:** `tracking_cost()`. Pure NumPy, takes the rollout and the
> reference, returns a scalar. Tests check it is zero on a perfect match,
> positive otherwise, and that heading error wraps.

---

## 3. Make it an MPC: optimise

You now have `J: R^32 -> R`. Hand it to an optimiser and you have model
predictive control. Honestly:

```python
from scipy.optimize import minimize
result = minimize(lambda z: J(z, state, reference), z0,
                  method="SLSQP", bounds=bounds)
control = result.x[:2]        # apply only the first one
```

**Do this stage with finite differences and no CasADi.** It will be slow —
SciPy needs 33 cost evaluations per gradient — but it will work, and you will
have a genuinely functioning MPC that you wrote. Verify it on the easy cases
before adding anything: starting on the reference should give near-zero cost;
starting displaced should give a control that moves you toward the reference.

### Bounds vs constraints

Two different mechanisms, and mixing them up is a classic error:

- **Bounds** are per-variable boxes, `−2.0 ≤ a_k ≤ 1.5` and
  `|omega_k| ≤ 1.5708`. Cheap, always satisfiable, handled by the solver
  natively. These are your actuator limits.
- **Constraints** are functions of several variables, `g(z) ≥ 0`. Expensive, can
  be infeasible, need derivatives. This is where collision avoidance lives.

Put everything you can into bounds. Note that `v_max` and `delta_max` are *not*
bounds — they are enforced inside the rollout by the `clip`/`fmin`/`fmax`, which
is why those appear in the dynamics rather than in `self.bounds`.

> **Stage 3 exercise:** `solve_unconstrained()`. 20 lines. After this you have a
> working single-car MPC. Tests check it reduces cost versus the reference guess
> and respects bounds.

---

## 4. Why CasADi: exact derivatives

SLSQP is a *gradient-based* method. It needs `∇J` (32 numbers) and, once you add
collision constraints, the Jacobian `∂g/∂z` (180×32 for 6 cars). Finite
differences give you these at a cost of one extra cost evaluation per variable,
and with truncation error that makes the solver's convergence test unreliable
near the optimum.

CasADi solves this by **automatic differentiation**: you build the cost out of
symbolic `SX` scalars, and `ca.gradient(cost, controls)` returns an *exact*
expression for the gradient, compiled into a callable C-speed function:

```python
self._obj = ca.Function("tracking_cost", [controls, params],
                        [cost, ca.gradient(cost, controls)])
```

Note that this returns **both** the cost and its gradient from one call, which is
why `objective()` at [mpc.py:120-122](mpc.py#L120-L122) can pass `jac=True` to
SciPy and never evaluate anything twice.

This is the key realisation about CasADi, and it is the thing that makes the file
look intimidating when it is not: **CasADi is not an optimiser.** It is a
calculator that does symbolic differentiation and emits fast functions. SciPy's
SLSQP does the actual optimisation. (`do_mpc` hides both behind one object, which
is pleasant until you want to control either one.)

The measured payoff on your problem: **1.28 ms** per solve for 6 cars with 180
collision constraints, versus **9.1 ms** for `mpc/mpc.py` solving a single car
with no constraints at all. At 24M solves in a 1M-step training run that is 8.5
hours instead of 61+.

### The `params` vector

CasADi functions are compiled once, so everything that changes per tick must be
an *input*, not baked in. Hence the single flat `params` vector
([mpc.py:42-43](mpc.py#L42-L43)):

```
params = [ state (5)
         | reference states (17 x 5 = 85)
         | previous command (2)
         | per neighbour: X, Y, psi, v, delta, present (6 x j) ]
```

`nb0 = 5 + 5*(N+1) + 2 = 92` is just the offset where the neighbour block
starts. The indexing arithmetic is the ugliest part of the file and it is purely
bookkeeping — no physics is hiding in it.

> **Stage 4 exercise:** `build_cost_function()`. Rebuild the stage-2 cost
> symbolically and return a `ca.Function`. Tests check it agrees with your NumPy
> cost to 1e-9 and that the gradient matches finite differences.

---

## 5. Collision constraints

### The geometry

A car is 0.537 m × 0.245 m, and the state is at the **rear axle**, 0.164 m behind
the centre. One circle enclosing that body needs radius ≈ 0.5 m, which is a
28 cm-wide car pretending to be a 1 m ball. Two cars then refuse to pass closer
than 1.0 m centre-to-centre — and that produced a real, measured deadlock: at
6 cars, 71 of 72 failed cars were *frozen*, 91% of them pinned at exactly that
bubble.

The fix is to cover the body with `footprint_discs = 2` smaller circles along the
centre line. From `body_discs()`:

```
radius  = hypot(length/(2*count), width/2) = hypot(0.1343, 0.1225) = 0.1817 m
offsets = [0.0298, 0.2983] m   forward of the rear axle
```

Two cars are clear when *every* pair of discs is clear, so the required
centre-to-centre separation drops from 1.00 m to
`2*0.1817 + 0.08 = 0.4435 m` — and the N=6 success rate went 25% → 80% with
frozen cars 71/72 → 0/0 and collisions still 0%.

**This is the single most instructive part of the whole file**: the controller
was not badly tuned, it was solving the wrong geometry problem. Worth
remembering when something looks like an RL failure.

### The constraint

For ego disc at `c` and neighbour disc at `d`, non-overlap is
`||c − d|| ≥ clearance`. Never write it that way. Square it:

```
g = ||c − d||^2 − clearance^2  >=  0
```

The norm has a non-differentiable kink at zero and a square root the solver does
not need. The squared form is a smooth polynomial, and since both sides are
non-negative the two statements are equivalent.

SciPy's convention for `type="ineq"` is `g(z) ≥ 0`. Get the sign wrong and you
will instruct the cars to *seek* collisions, which is at least an unambiguous
symptom.

### Forecasting the neighbour

You cannot know what the other car will do — that would be the illegal
communication you have carefully avoided. So assume constant velocity along its
current heading ([mpc.py:72](mpc.py#L72)):

```
d_k = p_other + (k+1) * dt * v_other * [cos(psi_other), sin(psi_other)]
```

Be honest about this in the paper: it is a heuristic, wrong under turning, and
it is why the module docstring says the avoidance is not a guarantee. It is
adequate because the horizon is 0.8 s and you re-solve at 20 Hz.

### The presence mask

The CasADi function is compiled for a fixed `j = n_cars − 1` slots, but only
some are occupied (others are out of `sensing_radius = 5 m`). Empty slots are
zero-filled, which would put a phantom car at the origin. The trick at
[mpc.py:80](mpc.py#L80):

```
constraint = present * g + (1 - present)
```

`present = 1` → the constraint is `g ≥ 0`, enforced. `present = 0` → the
constraint is `1 ≥ 0`, trivially true and with **zero gradient**, so it costs the
solver nothing. This is how you get a variable number of neighbours out of a
fixed-size compiled function, and it is a pattern worth internalising.

### The stride

Constraining all 16 steps × 5 neighbours × 2 × 2 discs = 320 constraints is
wasteful when consecutive steps are 0.05 s apart and nearly redundant.
`constraint_stride = 2` enforces every other step, plus always the last
([mpc.py:68](mpc.py#L68)) — 9 steps, **180 constraints** for 6 cars. The
last step must always be included or the horizon end is unprotected.

> **Stage 5 exercise:** `build_constraint_function()` plus `disc_centres()`.
> Tests check head-on neighbours produce negative residuals, far ones positive,
> masked slots exactly `1.0`, and the count matches 180 at 6 cars.

---

## 6. The engineering that makes it survive

Stages 1–5 give a correct MPC. These four things make it work in a 1M-step
training run, and each exists because of a specific observed failure.

**Warm starting** ([mpc.py:182](mpc.py#L182)). The problem at tick t+1 is nearly
the one at tick t shifted by one step. So reuse the solution, shifted, with the
last control duplicated:
```python
self._warm = np.concatenate([candidate[2:], candidate[-2:]])
```
Typically halves the iteration count. Discard it on failure (set to `None`) — a
bad warm start is worse than none.

**The degenerate-guess guard** ([mpc.py:170-175](mpc.py#L170-L175)). The bug that
justifies the whole hand-rolled approach. At `v = 0` the cost is nearly flat in
`omega`, SLSQP's convergence test fires immediately, and it returns the shifted
guess after **one iteration** — a car reported "converged" at cost 44.599 when
the true optimum was 0.606, and it sat still forever. The fix: if the warm start
differs from the reference guess, also evaluate the reference guess, and re-solve
from it if it is cheaper. Keep the better *validated* result. Note the subtlety
in the comment — you must allow an *infeasible* reference guess here, because
requiring feasibility would reinstate the trap exactly when the requested motion
crosses a neighbour forecast.

**Validation before execution** (`valid_result`, [mpc.py:136-141](mpc.py#L136-L141)).
Never trust `result.success`. Check finiteness, bounds, and constraint
satisfaction independently. Optimisers lie, especially SLSQP on nonconvex
problems.

**The braking fallback** ([mpc.py:94-97](mpc.py#L94-L97)). When no valid solution
exists — infeasible, NaN input, exception — do something defined and safe:
maximum brake, steering back to centre. A controller without a fallback will
eventually emit a NaN into your plant at 3 a.m. of a training run.

> **Stage 6 exercise:** `MyMPC.control()`. Assemble everything. Tests cover the
> zero-speed trap, warm-start reuse, NaN handling, and fallback behaviour.

---

## 7. Reading material

Ordered by how useful each will actually be to you.

**Start here.**
- Rawlings, Mayne & Diehl, *Model Predictive Control: Theory, Computation and
  Design*, 2nd ed. The standard reference, and the PDF is free from
  Rawlings' group page. Read **Ch. 2.4** (terminal cost and stability — this is
  your 16/6 weights) and **Ch. 8.1–8.3** (the numerical side). Skip the rest for
  now.
- Diehl & Gros, *Numerical Optimal Control* (free PDF, Freiburg). The single
  best treatment of **single vs multiple shooting**, which is the structural
  choice your file makes. Chapters 8 and 9.
- Borrelli, Bemporad & Morari, *Predictive Control for Linear and Hybrid
  Systems* (free PDF). Ch. 11 for the receding-horizon formulation.

**For the code.**
- CasADi docs, the "Function" and "Nonlinear programming" sections, plus
  `example_pack`'s `direct_single_shooting.py` — it is 60 lines and is
  essentially the skeleton of your file. Read this one first if you only read
  one thing.
- Andersson et al., *CasADi: a software framework for nonlinear optimization and
  optimal control*, Math. Prog. Comp. 2019. The paper to cite.
- Kraft (1988), *A software package for sequential quadratic programming*. The
  original SLSQP report, if you want to say in the paper what the solver
  actually does: it solves a sequence of QP approximations of your NLP.
- `do_mpc` docs for contrast. Reading its `set_nl_cons` and `tvp` pages will
  show you that everything here is expressible there too — the difference is
  speed and control, not capability.

**Vehicle side.**
- Rajamani, *Vehicle Dynamics and Control*, Ch. 2 for the kinematic bicycle and
  exactly when it stops being valid (low speed, small slip — your regime).
- Kong et al. (2015), *Kinematic and dynamic vehicle models for autonomous
  driving control design*, IV 2015. Quantifies kinematic-vs-dynamic error. Good
  citation for your sim2sim section.

**Multi-agent avoidance** — for your related-work section, since reviewers will
ask why not these:
- Van den Berg et al., *Reciprocal n-body collision avoidance* (ORCA). The
  classic velocity-obstacle baseline.
- Zhu, Brito & Alonso-Mora, *Decentralized probabilistic MPC for multi-robot
  collision avoidance*, RA-L 2019. Closest published work to your low level.

**Courses**, if you prefer video: Moritz Diehl's *Numerical Optimal Control*
lectures (Freiburg, on YouTube) are the ones that correspond to the material
above.

---

## 8. Suggested order of work

Spread over a few sessions:

1. Stages 1–3 in one sitting, finite differences, single car, no neighbours.
   You will have your own working MPC. Compare its output against pure pursuit
   on the same reference and reproduce the divergence we measured
   (ref `+0.800 / +90.0°/s` vs MPC `+0.462 / +38.9°/s`) — that will tell you the
   cost is right.
2. Stage 4, swap in CasADi, confirm identical numbers and time the speedup.
3. Stage 5, add neighbours. Test two cars head-on and watch one yield.
4. Stage 6, the robustness layer. Then run `tests/test_mpc.py` against your
   class — if the existing suite passes on your implementation, you are done.
5. Swap it into `env.py` (one line, `from .mpc_exercise import MyMPC as MPCTracker`)
   and rerun `baseline.py`. Matching success rates is your proof.

Then the MPC chapter is yours to defend, and the measured results are unchanged
because the maths is the same.
