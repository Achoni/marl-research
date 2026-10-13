# Frozen implementation and reporting protocol

## Task and information

Each car has state `s = [x, y, yaw, speed, steering]` at its rear axle and a fixed
assigned goal. A shared deadline T is supplied before the episode. T uses the
slowest sampled trip's nominal translation/turning time with randomized slack;
this initialization uses all missions. It is **not** a decentralized deadline
consensus algorithm and is **not** a certificate of joint feasibility.

Execution is decentralized given this setup: one shared actor runs separately
for each car, with its own goal/state/time and sensed nearby relative states.
It receives neither other cars' goals nor the critic's global state. MPC solves
one problem per car using locally sensed neighbours and no exchanged future plans.
The simulator loops over cars centrally for convenience, but all commands in an
interval use the same pre-integration state snapshot. There is no leader.

The arena bounds only where starts and goals are drawn. Driving outside that
square is allowed. There are no static obstacles, map planning, reverse driving,
or explicit inter-car messages. Sensing is exact inside a 5 m radius; sensor
noise, occlusion and communication failures are not part of the default claim.

## Actor and MAPPO

Own observation has 10 entries: local goal x/y, forward speed, steering, remaining
seconds, deadline, initial distance, arrived flag, required mean speed, and distance
progress minus elapsed-time fraction. Scaling is defined in `config.EGO_FEATURES`.
Each neighbour has 6 entries: relative x/y, relative world velocity expressed in
the ego frame, and sine/cosine of relative heading. KNN keeps up to K=4 within 5 m;
missing slots are masked.

Let e be the learned 128-dimensional ego embedding and h_j the learned neighbour
embedding. Attention computes q=Wq e, k_j=Wk h_j,

    alpha_j = masked_softmax(q^T k_j / sqrt(128))
    context = sum_j alpha_j h_j

The ablation replaces alpha by uniform weights over present neighbours, preserving
the embeddings, action head, observation information and training algorithm. It
has fewer parameters because unused query/key projections are removed; parameter
counts are included in benchmark results. Both actors produce zero context when
no neighbours are present. Neighbour order does not affect the pooled result.

The actor head produces the mean of a diagonal Gaussian; a learned log standard
deviation supplies exploration. Training samples latent controls, applies tanh,
and uses the change-of-variables log probability in PPO. Evaluation uses tanh of
the mean. Action outputs decode to forward waypoint distance 0.25–2 m, lateral
offset ±1.2 m, and preferred speed 0–2 m/s. **The third action is speed, not an
arrival timestamp.** Timing is learned through observations and reward.

The centralized critic uses 16 features per car, including absolute mission timing
and arrival/dwell history, with shared per-car encoding and a mean team context.
It is discarded at execution. Training uses GAE, clipped PPO policy updates,
value loss, entropy regularization and separate actor/critic gradient clipping.
An action that achieves arrival still gets credit; inactive agents are masked.
True task termination suppresses bootstrapping; collector/curriculum cuts bootstrap
but stop the GAE trace. Look at saved `algorithm` fields for actual hyperparameters.

All cars receive the same cooperative reward per policy interval:

    3 mean((old_distance - distance)/initial_distance)
    - schedule_weight mean(abs(progress - min(time/T, 1)))
    - 0.01 elapsed_seconds
    + arrival_bonus + team_completion_bonus - collision_or_timeout_penalty

The one-time per-car arrival bonus is `5/N * exp(-0.5*((arrival-T)/0.6)^2)`.
Completion adds `10 exp(-max_deadline_error/0.6 - spread/0.6)`.
A collision subtracts 25; timeout subtracts `10 * fraction_not_arrived`.
PP and MPC use the same reward; solver failure is a diagnostic, not an extra MPC
reward. `schedule_weight` is frozen in the configuration.

## PP reference and nominal dynamics

The waypoint is fixed in world coordinates during one 1.2 s rollout. Pure pursuit
sets geometric steering `atan2(2 L y_local, distance_squared)`, clipped to ±27°,
and a proportional speed controller sets acceleration. Bounds are acceleration
[-2, 1.5] m/s² and steering rate ±90°/s. At each predicted state, preferred speed
is capped by braking-distance heuristics to the goal and waypoint. Goal handoff
requires suitable forward geometry or an already-captured goal, allowing a
lateral waypoint to help recover when a goal is behind a forward-only car.

This produces 25 states and 24 controls at 0.05 s spacing. It is a **timed nominal
reference**, not a guarantee to finish at the goal within 1.2 s. State updates use
bounded speed/steering, their interval averages, and midpoint heading integration:

    v+ = clip(v + dt*a, 0, vmax)
    delta+ = clip(delta + dt*steering_rate, -delta_max, delta_max)
    d_yaw = dt * mean(v,v+) * tan(mean(delta,delta+)) / L
    x+ = x + dt * mean(v,v+) * cos(yaw + d_yaw/2)
    y+ = y + dt * mean(v,v+) * sin(yaw + d_yaw/2)
    yaw+ = yaw + d_yaw

PP execution uses the first four rollout controls over the next 0.2 s, then
regenerates from the measured state. It does not recompute geometric pursuit
from plant feedback at every 0.05 s control tick. MPC instead recomputes its
tracking solution from measured state at every control tick. This feedback-rate
difference is part of the defined comparison and should not be hidden.

## NMPC

Each local problem has 16 future controls `u_k=[a_k, steering_rate_k]`, 32 scalar
decision variables, over 0.8 s. Single shooting eliminates states as independent
decisions: they are generated by repeated nominal bicycle updates. No tyre-slip
or actuator-lag states are included in the prediction model.

The cost is the sum over predicted states at k+1:

    8 ||position - reference_position||²
    + 3 (speed - reference_speed)²
    + 0.5 wrapped_heading_error² + 0.1 steering_error²
    + 0.015 ||u_k||² + 0.04 ||u_k - u_previous||²

The terminal state gets additional position weight 16 and speed weight 6.
Reference controls initialize/reinitialize the optimizer; the cost penalizes
control effort, **not** `u - u_reference`. Even with identical models, the effort,
smoothness and separation terms can make the optimized path differ from PP.
There is no terminal equality to the goal and no stability/recursive-feasibility
proof. State saturation makes the prediction piecewise smooth. CasADi supplies
algorithmic derivatives of these expressions; SLSQP finds a local numerical
solution, not a guaranteed global optimum.

A length 0.55 m by width 0.34 m rectangle encloses the bundled car, including
steered tyres. Two longitudinal slices are each covered by a disc, radius
`sqrt((length/4)^2 + (width/2)^2)`. Disc centres are located relative to the rear
axle. Each ego/neighbour disc pair must have predicted centre separation at least
`2*radius + 0.08 m`, at every prediction node. Neighbours are extrapolated at
constant speed and heading; in transfer this forward-velocity approximation does
not model lateral slip. All neighbours sensed within 5 m enter the MPC, independent
of actor KNN truncation. Out-of-range slots are masked.

Warm-start controls are shifted from the previous solution. The PP reference is
also tried if the shifted solution is invalid or more expensive, addressing a
numerical stopping trap. Returned commands must pass finite/bound/constraint
checks. Otherwise the controller brakes and straightens steering. These changes
can reduce numerical stalls but cannot eliminate symmetric yielding deadlocks
or physical infeasibility. Failure, fallback and timeout must be reported.

The collision check used by the environment sweeps disc centres between successive
0.01 s physics states and tests physical contacts in MuJoCo. MPC constraints are
sampled forecasts, not continuous-time or robust safety guarantees. Changing
neighbour actions can invalidate a forecast; braking itself is not always safe.

## Clocks, stopping and success

The actor updates every 0.2 s (5 Hz). Each interval has four control ticks of
0.05 s (20 Hz), and each tick has five environment physics updates of 0.01 s.
MuJoCo internally integrates its smaller configured timestep to advance those
0.01 s intervals exactly. A new reference is constructed at each policy update.

Arrival requires remaining inside the goal tolerance, with planar speed at most
0.08 m/s and a stopping-distance reserve fitting within 0.25 m, for a 0.20 s dwell.
The recorded arrival time is the beginning of this dwell. Departing the parked
region revokes arrival, and the car remains a physical obstacle while parked.
An episode ends on collision, all-car arrival, or T+3 s.

Strict team success requires all of the following:

1. All cars satisfy the stopped-and-dwelling arrival rule.
2. No footprint collision or recorded MuJoCo inter-car contact.
3. `max_i |arrival_i - T| <= 0.60 s`.
4. `max_i arrival_i - min_i arrival_i <= 0.60 s`.

These timing tests are distinct: being close to T does not automatically bound
the total arrival spread to the same tolerance. Completion outside these timing
windows is retained as a failure, with timing errors still reported.

## What the experiments can establish

Independent seeds address training variability. Matched stored missions address
scenario comparability. Attention-vs-mean tests learned relevance weighting under
the same KNN selection; it does not isolate the benefit of KNN itself. PP/posthoc
MPC/MPC-trained comparisons measure the effect of controller fusion. The local
non-learning baselines test whether learning improves over these explicitly
specified heuristics. Larger teams test empirical zero-shot transfer, separately
from increased density. Dynamic/MuJoCo plants test simulation transfer.

The package alone establishes none of the following: formal safety, arbitrary-N
scalability, state-of-the-art superiority, sensor robustness, hardware transfer,
or globally feasible deadlines. Those claims require their own evidence.
