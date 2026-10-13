# Simultaneous Arrival of N Cars
### How the final model works, and what it is actually worth

Best checkpoint: `runs/mpc_s0/best.pt`. Every number here was measured on this code.

---

## 0. The question this document starts with

> *If pure pursuit generates the reference and the MPC tracks it, and both use the same
> bicycle model, aren't they the same? Is the MPC only there for collision avoidance?*

Reasonable, and it was my working assumption too. It is wrong, and the measurement says so.
Over 200 random states, comparing the MPC's command against the pure-pursuit reference
control it is tracking:

| situation | mean \|Δ accel\| | mean \|Δ steer rate\| |
|---|---|---|
| **no neighbour at all** | 0.40 m/s² | 0.69 rad/s |
| car 1.2 m ahead | 1.71 m/s² | 0.97 rad/s |

In **96% of free-space cases** the MPC deviates from the reference by more than 0.1 m/s².
Acceleration is bounded to [−2.0, +1.5], so 0.40 is roughly 11% of full range; steering rate
is bounded to ±1.57 rad/s, so 0.69 is 22%.

**Why they differ even with nobody around.** Pure pursuit is a myopic rule evaluated once:
it computes the steering angle that would put the car on an arc through the target and then
demands `(steer_target − steer_now) / dt`, which usually saturates the 90°/s rate limit. The
MPC looks 0.8 s ahead, and its cost contains `0.015·‖u‖² + 0.04·‖u − u_prev‖²`, so it spreads
the same turn over several steps instead of slamming the rack. It also weighs position error
8, speed 3, heading 0.5 and steering 0.1 against each other, trading a little of one for less
of another. Pure pursuit has no notion of that trade.

So the MPC does **two** things, not one:

1. **Smooths and optimises the tracking** — worth 0.40 m/s² of difference in free space.
2. **Enforces separation** — worth 1.71 m/s² when a car is in the way.

The second is the larger effect, so "mostly collision avoidance" is a fair summary. But
"identical apart from avoidance" is measurably false.

**And there is a third thing, which the experiments show is the most important of all:
training *with* the MPC in the loop produces a better policy than bolting it on afterwards.**
Section 4 has the evidence.

---

## 1. The pipeline, end to end

Three nested loops, none holding a plan longer than 1.2 seconds.

```
  every 0.20 s   ATTENTION POLICY          own goal, own state, shared deadline T,
     (5 Hz)                                 up to K sensed neighbours
                        |
                        |  three numbers in [-1, 1]
                        v
  once per       PURE-PURSUIT ROLLOUT      simulate the bicycle forward 1.2 s
  policy step                               -> 25 states x 24 controls, time-indexed
                        |
                        |  a TIMED reference: "be here at t0 + k*0.05 s"
                        v
  every 0.05 s   NONLINEAR MPC             track the reference over 0.8 s subject to
   (20 Hz, x4)                              hard separation from every sensed car
                        |
                        |  [acceleration, steering rate]
                        v
  every 0.01 s   PLANT                     kinematic bicycle | dynamic bicycle | MuJoCo
  (100 Hz, x5)
```

### 1.1 What the policy sees and outputs

Ten ego numbers, all in the car's own body frame, plus up to K neighbour rows of six:

| ego | meaning |
|---|---|
| 0–1 | own goal, relative, body frame (÷10) |
| 2–3 | own speed (÷2.0), own steering angle (÷27°) |
| 4–5 | time remaining, the shared deadline T (÷20) |
| 6–7 | own initial distance (÷10), arrived flag |
| **8** | **required average speed** = distance / time remaining (÷2.0) |
| **9** | **schedule error** = progress − elapsed fraction |

Rows 8 and 9 matter more than they look. They are *ratios*, derivable in principle from rows
0–6, but a small tanh network approximating division across a wide input range does it badly.
An earlier model without them had a correlation of only **+0.08** between the speed the
schedule demanded and the speed it commanded, meaning it drove at a near-constant pace no
matter what the deadline needed. Handing it the ratio removes the hardest part of the problem.

Neighbour rows carry relative position, relative velocity and sin/cos of relative heading,
with a presence mask. **Nothing about another car's goal, action or progress is visible.**

Output, every 0.2 s:

```
forward  = 0.25 + (a0+1)/2 * 1.75     a waypoint 0.25 ... 2.0 m ahead
lateral  = a1 * 1.2                    +-1.2 m to the side
speed    = (a2+1)/2 * 2.0              preferred speed 0 ... 2 m/s
```

This is a **short-term intent**, replaced 0.2 s later. The waypoint is a via point to steer
through, not a place to stop.

The actor is attention-based: one shared encoder per neighbour, the ego queries them, and the
result is a masked weighted sum. Nothing in the network has a shape that depends on K, so the
same weights run three cars or ten. That is also why it generalises across scenario
distributions: an earlier flat-MLP version scored 10% where the attention version scored 88%
on the same unseen distribution.

### 1.2 How the reference is generated

Starting from the car's **measured** state, simulate the bicycle model forward 24 steps of
0.05 s, driven by pure pursuit toward the decoded waypoint:

```
steer_target = atan( 2 * L * lateral_offset / distance^2 )      # arc through the target
accel        = 4.0 * (target_speed - current_speed)
target_speed = min( preferred,
                    sqrt(2 * b * distance_to_goal),             # can still stop at the goal
                    sqrt(2 * b * distance_to_waypoint) )        # and at the waypoint
```

Two properties make this the right interface:

**It is feasible by construction.** The same model, the same actuator limits. Nothing in the
reference is impossible for the car, so the MPC is never asked to chase a fantasy.

**It is indexed by time, not by arc length.** Row *k* says where the car should be at
`t0 + k·0.05 s`. That is what makes the next layer a *trajectory* tracker rather than a path
follower: a car that drives the right curve too slowly is penalised, because reference row *k*
has moved on without it.

The rollout also switches from the waypoint to the goal once the goal is nearer, and brakes
into it. Without that switch an untrained policy drives to its goal and straight past it, which
was a real bug: it ended episodes 8.8 m from a goal it had already reached.

### 1.3 How the MPC tracks it

Single-shooting nonlinear MPC, 16 steps of 0.05 s. Decision variables are acceleration and
steering rate at each step, 32 numbers, box-bounded by the actuator limits.

```
minimise   sum_k  8|p_k - p_ref,k|^2 + 3(v_k - v_ref,k)^2
                + 0.5*wrap(yaw_k - yaw_ref,k)^2 + 0.1(steer_k - steer_ref,k)^2
                + 0.015|u_k|^2 + 0.04|u_k - u_{k-1}|^2
         + 16|p_H - p_ref,H|^2 + 6(v_H - v_ref,H)^2          (heavier terminal)

s.t.     x_{k+1} = bicycle(x_k, u_k)                          same model, in CasADi
         |d_a - d_b| >= 2*disc_radius + margin                 every disc pair, every
                                                               `constraint_stride` steps
```

CasADi builds the cost and constraints **and their exact derivatives** once at construction;
SLSQP solves them, warm-started from the previous solution shifted one step. Only the first
control is applied. Typical solve 2–11 ms. On failure the car brakes straight.

Note the cost contains **no notion of the deadline**. The MPC only knows "be at reference row
*k* at step *k*". Timing reaches it entirely through the speeds the policy requested.

### 1.4 The collision geometry (fixed late, and it mattered)

The car is 0.537 × 0.245 m with the rear axle 0.164 m behind its centre. It used to be
approximated by **one circle of radius 0.50 m centred on the rear axle**, which is correct but
covers **4.8× the car's real area**, because a circle enclosing a long thin car has to be as
wide as the car is long in every direction.

It is now covered by circles laid **along the car's own axis**:

| model | disc radius | required car-car gap | area reserved |
|---|---|---|---|
| 1 disc on the rear axle (old) | 0.449 m | 0.898 m | 4.8× |
| 2 body discs | 0.182 m | 0.363 m | 1.6× |
| **3 body discs (recommended)** | **0.152 m** | **0.303 m** | **1.6×** |

More accurate *and* less restrictive at the same time. The measured effect, on the **same
frozen checkpoint** with nothing retrained:

| cars | collision model | success | arrived | collisions | timeouts | frozen cars |
|---|---|---|---|---|---|---|
| 3 | single disc (as trained) | 95% | 100% | 0% | 0% | 0/0 |
| 3 | 3 body discs | 95% | 100% | 0% | 0% | 0/0 |
| 6 | single disc (as trained) | **25%** | 30% | 0% | 70% | **71/72** |
| 6 | 2 body discs | 75% | 95% | 0% | 5% | 2/2 |
| 6 | **3 body discs** | **80%** | **100%** | **0%** | **0%** | **0/0** |

At six cars, 71 of 72 non-arriving cars were **frozen at a dead stop**, 91% of them pinned
exactly against the old 1.08 m bubble. Every car's MPC refused to approach, so nobody moved.
Fixing the geometry took success from 25% to 80% with collisions still at zero.

Cost: more constraint rows, so the separation is imposed every `constraint_stride` = 2 horizon
steps rather than every step. The car moves under 0.1 m per step so nothing slips through, and
the solve stays at 11 ms instead of 81 ms.

---

## 2. Results of the final model

`runs/mpc_s0/best.pt`, 60 stratified held-out episodes:

| metric | value |
|---|---|
| success (all within ±0.6 s of T, spread ≤ 0.6 s, no collision) | **75%** (peak 83% at step 860k) |
| collisions | **0%**, at all 47 training evaluations |
| all arrived | 88% |
| arrival spread | 0.51 s |
| worst separation ever | 1.079 m against a 1.08 m limit |

**Per scenario family**, which localises everything:

| family | success | arrived | collisions | timeout | spread |
|---|---|---|---|---|---|
| converging | **100%** | 100% | 0% | 0% | 0.11 s |
| crossing | **100%** | 100% | 0% | 0% | 0.07 s |
| parallel | **100%** | 100% | 0% | 0% | 0.08 s |
| random | 70% | 85% | 0% | 15% | 0.51 s |
| unequal | **22%** | 67% | 0% | 33% | 2.16 s |


The unequal is wrong scenario because it sometimes gives us infeasivle scenarios so we can exclude that in the research

Three families solved outright. **All the remaining failure is in `unequal`**, where one car
has a much shorter trip and must wait. 18% of arrivals are early and **zero are late**, which
names the mechanism exactly: short-trip cars rush and then sit.

Worth knowing about that family: its shortest trip averages **1.50 m** against a 15.4 s
deadline, so that car finishes in under a second and must kill 14.6 seconds. That is a
precision-loitering task rather than a coordination task. `--unequal-short-max 4.0` softens it
to a ratio of 0.34, or `--family-weights` drops it entirely.

### Sim2sim transfer, zero-shot, 25 paired episodes per row

| plant | success | collisions | spread |
|---|---|---|---|
| kinematic (training model) | 96% | 0% | 0.18 s |
| **MuJoCo F1TENTH** | **96%** | **0%** | **0.17 s** |
| dynamic, μ = 0.6 | 84% | 4% | 0.30 s |
| dynamic, μ = 0.3 | 84% | 0% | 0.29 s |
| dynamic, lag 0.15/0.20 s | 72% | 0% | 0.29 s |
| dynamic, lag 0.30/0.40 s | **20%** | 8% | 0.42 s |

**MuJoCo matches the training model exactly.** A policy trained in a forty-line kinematic
bicycle drives torque-controlled motors and servo steering racks with real tyre contacts, with
nothing retrained and no measurable loss. Tyre friction never binds, down to μ = 0.3.
**Actuator lag is the only thing that does**, failing above roughly a 0.25 s steering time
constant. Real F1TENTH servos sit at 0.1–0.2 s, inside the working region.

---

## 3. Why transfer works at all

Not adaptation: the weights are frozen and nothing is learned at deployment. Not the MPC
either. The reason is the **decision rate against the model error**. Measured per 0.2 s policy
step during real episodes, MuJoCo deviates from the kinematic prediction by about **one
centimetre**, and an icy dynamic model by 1.3 cm. The policy chooses waypoints 0.25–2 m away,
so a 1 cm error is under 1% of its smallest choice, and the next observation already contains
it.

This is a property of the operating regime and must be stated as scope. The same comparison
while cornering hard: 0.9 cm of divergence per step at 1 m/s, 4.1 cm at 2 m/s, **21 cm at
4 m/s**, 51 cm at 6 m/s. The tyre limit at μ = 0.8 is 7.8 m/s². **This pipeline is valid in
the low-slip regime below roughly 3 m/s.**

Division of labour under a new plant: the **MPC** owns spatial error inside a horizon under one
second; the **policy** owns timing error across the whole episode, because every 0.2 s it
re-reads remaining distance against remaining time and adjusts the requested speed.

---

## 4. The MPC ablation, and why training with it matters

Four configurations, all measured:

| configuration | success | collisions | spread |
|---|---|---|---|
| trained on PP, executed on PP | 52% | **22%** | 0.48 s |
| trained on PP, **MPC swapped in** | 60% | **0%** | 0.72 s |
| trained on MPC, executed on MPC (`arrival_mpc`) | 60% | 0% | 0.49 s |
| **trained on MPC, executed on MPC (`mpc_s0`)** | **75%** | **0%** | 0.51 s |

Three conclusions, in increasing order of importance.

**The MPC eliminates collisions.** 22% → 0% on the identical policy and scenarios. This is the
cleanest safety result in the project, because nothing is confounded with training: same
weights, same seeds, only the executor differs.

**Bolting it on afterwards is not free.** Timeouts rise 5% → 18% and spread degrades 0.48 →
0.72 s. A PP-trained policy learned to dodge by itself because nothing else would, so adding a
second avoidance system makes the pair over-conservative. Crashes become waiting.

**Training with it beats bolting it on.** 75% versus 60%, same environment, same architecture.
This is the answer to your question: the MPC is not a filter you can add at the end without
cost. Its braking and swerving are part of the dynamics the policy learns, and a policy trained
against it learns to *lean* on it, spending its capacity on timing rather than on avoidance
it no longer has to do.

There is also a clean negative result. Training with pure pursuit **alone** on the harder
scenario set failed entirely: 0% success and 10–20% collisions, with the policy never
converging (entropy still 1.30 at 921k steps against 0.24 for the MPC run). Without the
constraint the policy must learn goal-reaching, timing and avoidance simultaneously, and one
million steps is not enough. **The MPC does not merely make the system safe, it makes the
learning problem tractable within a realistic budget.**

---

## 5. Honest assessment

### Solid

- 100% on three of five scenario families with zero collisions.
- Zero collisions in every evaluation across a million steps, with the worst separation ever
  recorded sitting exactly at the constraint limit. The safety layer does what it claims.
- MuJoCo transfer at parity with the training model, zero-shot.
- A clean four-way controller ablation, including the heterogeneous case.
- Zero-shot scaling to six cars at 80% once the collision geometry is right.
- The decentralisation claim is auditable: the actor reads only its own goal, its own state,
  the shared deadline and relative neighbour states.


---

## 6. Commands

```bash
# evaluate (per-family breakdown is in the JSON)
python -m arrival_v2.evaluate --checkpoint runs/mpc_s0/best.pt --episodes 200 --output eval.json

# watch it
python3 -m arrival_v2.view --checkpoint runs/mpc_s0/best.pt --cars 6 --footprint-discs 3
mjpython -m arrival_v2.view_mujoco --checkpoint runs/mpc_s0/best.pt --cars 6

# sim2sim transfer table
python -m arrival_v2.transfer --checkpoint runs/mpc_s0/best.pt --episodes 40 --extra

# the no-learning baseline
python -m arrival_v2.baseline --controller mpc --episodes 50

# train, with the corrected geometry and the unequal family softened
python -m arrival_v2.train --controller mpc --steps 1000000 --seed 1 \
    --run-dir runs/mpc_s1 --unequal-short-max 4.0
```
