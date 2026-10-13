"""Local-observation simultaneous arrival, trained in a kinematic bicycle plant.

The PP experiment executes four successive reference controls before replanning.
The MPC experiment substitutes constrained tracking of exactly the same reference.
Actor K-nearest sensing and MPC neighbor capacity are intentionally independent.
"""
import math
import time
import numpy as np
from .config import EnvConfig, EGO_DIM, NEIGHBOR_DIM, CRITIC_DIM
from .model import bicycle_step, body_discs, disc_centres
from .reference import ReferenceGenerator
from .scenarios import ScenarioSampler


class ArrivalEnv:
    def __init__(self, config=None, seed=None, plant=None):
        """plant: None uses the nominal kinematic model the policy was trained in.

        Pass a plants.DynamicPlant or plants.MujocoPlant to run the SAME frozen
        policy and MPC over different physics (zero-shot sim2sim). The plant only
        replaces the integrator; observation, reward and events are unchanged.
        """
        self.config = (config or EnvConfig()).validate()
        self.plant = plant
        c = self.config
        self.n = c.n_cars
        self.K = c.max_neighbors
        self.sampler = ScenarioSampler(c, seed=seed)
        self.generator = ReferenceGenerator(c)
        self.trackers = []
        if c.controller == "mpc":
            # Optimization dependencies are imported only for MPC execution.
            from .mpc import MPCTracker
            self.trackers = [MPCTracker(c) for _ in range(self.n)]
        self.n_ctrl = round(c.policy_dt / c.control_dt)
        self.n_phys = round(c.control_dt / c.physics_dt)

    def reset(self, seed=None, scenario=None):
        if seed is not None:
            self.sampler.seed(seed)
        sc = self.sampler.sample() if scenario is None else scenario
        if (sc.starts.shape != (self.n, 5) or sc.goals.shape != (self.n, 2)
                or not np.isfinite(sc.starts).all() or not np.isfinite(sc.goals).all()
                or not math.isfinite(sc.deadline) or sc.deadline <= 0
                or np.any(np.linalg.norm(sc.goals - sc.starts[:, :2], axis=1) < 1e-8)):
            raise ValueError("Invalid supplied scenario")
        self.scenario = sc
        self.states = sc.starts.copy()
        self.goals = sc.goals.copy()
        self.deadline = sc.deadline
        self.family = sc.family
        self.initial_dist = self._distances()
        self.time = 0.0
        self.arrived = np.zeros(self.n, dtype=bool)
        self.arrival_times = np.full(self.n, np.nan)
        self._settle = np.full(self.n, np.nan)
        self._arrival_bonus_paid = np.zeros(self.n, dtype=bool)
        self.collision = self.done = False
        self.episode_reward = 0.0
        self.min_separation = self._pairwise_min()
        self.control_count = self.fallback_count = 0
        self.solve_time_total = self.solve_time_max = 0.0
        self.solve_times, self.control_interval_times = [], []
        self.physical_collision = False
        for tracker in self.trackers:
            tracker.reset()
        if self.plant is not None:
            self.plant.reset(self.states)
            self.states = self.plant.states.copy()
            self.initial_dist = self._distances()
            if hasattr(self.plant, "set_goals"):
                self.plant.set_goals(self.goals)
        return self.observe()

    def visible(self, i, actor=True):
        """Nearest cars within sensing range; only the actor truncates to K."""
        distance = np.linalg.norm(self.states[:, :2] - self.states[i, :2], axis=1)
        ids = np.flatnonzero((distance <= self.config.sensing_radius) & (np.arange(self.n) != i))
        ids = ids[np.argsort(distance[ids], kind="stable")]
        return ids[:self.K] if actor else ids

    def observe(self):
        c = self.config
        ego = np.zeros((self.n, EGO_DIM), dtype=np.float32)
        neighbors = np.zeros((self.n, self.K, NEIGHBOR_DIM), dtype=np.float32)
        mask = np.zeros((self.n, self.K), dtype=bool)
        velocity = self.world_velocities()
        for i, (x, y, yaw, v, steer) in enumerate(self.states):
            co, si = math.cos(yaw), math.sin(yaw)
            rotation = np.array([[co, si], [-si, co]])
            goal = rotation @ (self.goals[i] - self.states[i, :2])
            distance = float(np.linalg.norm(goal))
            remaining = self.deadline - self.time
            required_speed = distance / max(remaining, c.policy_dt)
            progress = 1.0 - distance / self.initial_dist[i]
            ego[i] = [goal[0] / 10, goal[1] / 10, v / c.max_speed, steer / c.max_steer,
                      remaining / 20, self.deadline / 20,
                      self.initial_dist[i] / 10, float(self.arrived[i]),
                      np.clip(required_speed / c.max_speed, 0.0, 4.0),
                      np.clip(progress - min(self.time / self.deadline, 1.0), -2.0, 2.0)]
            for slot, j in enumerate(self.visible(i)):
                rel_p = rotation @ (self.states[j, :2] - self.states[i, :2]) / c.sensing_radius
                rel_v = rotation @ (velocity[j] - velocity[i]) / (2 * c.max_speed)
                angle = self.states[j, 2] - yaw
                neighbors[i, slot] = [*rel_p, *rel_v, math.sin(angle), math.cos(angle)]
                mask[i, slot] = True
        return {"ego": ego, "neighbors": neighbors, "mask": mask}

    def critic_state(self):
        """Permutation-structured global/task state, for training only.

        Absolute deadline, remaining seconds, initial distance, dwell state and
        arrival history prevent distinct timing/reward states from aliasing.
        Local execution calls observe(); it never receives this global tensor.
        """
        c = self.config
        out = np.zeros((self.n, CRITIC_DIM), dtype=np.float32)
        for i, (x, y, yaw, speed, steer) in enumerate(self.states):
            distance = float(np.linalg.norm(self.goals[i] - self.states[i, :2]))
            settled = 0.0 if np.isnan(self._settle[i]) else np.clip(
                (self.time - self._settle[i]) / c.dwell_time, 0.0, 1.0)
            arrival_error = ((self.arrival_times[i] - self.deadline) / 20
                             if self.arrived[i] else 0.0)
            out[i] = [x / 10, y / 10, math.sin(yaw), math.cos(yaw), speed / c.max_speed,
                      steer / c.max_steer, self.goals[i, 0] / 10, self.goals[i, 1] / 10,
                      (self.deadline - self.time) / 20, self.deadline / 20,
                      self.initial_dist[i] / 10, float(self.arrived[i]),
                      1.0 - distance / self.initial_dist[i], arrival_error,
                      settled, float(self._arrival_bonus_paid[i])]
        return out

    def _distances(self):
        return np.linalg.norm(self.goals - self.states[:, :2], axis=1)

    def world_velocities(self):
        if self.plant is not None:
            return self.plant.world_velocities
        return self.states[:, 3, None] * np.column_stack((
            np.cos(self.states[:, 2]), np.sin(self.states[:, 2])))

    def _pairwise_min(self, previous_states=None):
        """Smallest FOOTPRINT GAP between any two cars, swept over the last sub-step.

        Each car is covered by `footprint_discs` circles along its own axis (see
        model.body_discs). The returned number is the closest disc-centre approach
        MINUS the two radii, so it reads directly as clearance:

            > 0   metres of daylight between the two cars
            = 0   the footprints touch
            < 0   they overlap; this is a collision

        Passing the previous states sweeps each disc pair along its straight-line
        motion over the sub-step, so a fast crossing cannot tunnel between samples.
        """
        radius = body_discs(self.config)[1]
        centres = [disc_centres(self.states[i], self.config) for i in range(self.n)]
        old_centres = (None if previous_states is None else
                       [disc_centres(previous_states[i], self.config) for i in range(self.n)])
        best = math.inf
        for i in range(self.n):
            for j in range(i + 1, self.n):
                distance = float(np.linalg.norm(
                    centres[i][:, None, :] - centres[j][None, :, :], axis=-1).min())
                if old_centres is not None:
                    for a in range(len(centres[i])):
                        for b in range(len(centres[j])):
                            start = old_centres[i][a] - old_centres[j][b]
                            change = (centres[i][a] - centres[j][b]) - start
                            length2 = float(change @ change)
                            alpha = (float(np.clip(-(start @ change) / length2, 0.0, 1.0))
                                     if length2 > 1e-20 else 0.0)
                            distance = min(distance, float(np.linalg.norm(start + alpha * change)))
                best = min(best, distance)
        return best - 2.0 * radius

    def _events(self, previous):
        c = self.config
        separation = self._pairwise_min(previous)
        self.min_separation = min(self.min_separation, separation)
        self.collision |= separation < 0.0        # separation is already a footprint gap
        self.physical_collision |= bool(getattr(self.plant, "contact_pairs", ()))
        self.collision |= self.physical_collision
        distances = self._distances()
        planar_speeds = np.linalg.norm(self.world_velocities(), axis=1)
        for i in range(self.n):
            speed = planar_speeds[i]
            stopped_inside = (distances[i] + speed * speed / (2 * c.max_brake) <= c.goal_tolerance
                              and speed <= c.stop_speed)
            if not stopped_inside:
                self._settle[i] = np.nan
                # Parked cars remain physical and are monitored until team completion.
                if self.arrived[i]:
                    self.arrived[i] = False
                    self.arrival_times[i] = np.nan
            elif not self.arrived[i]:
                if np.isnan(self._settle[i]):
                    self._settle[i] = self.time
                if self.time - self._settle[i] >= c.dwell_time - 1e-9:
                    self.arrived[i] = True
                    self.arrival_times[i] = self._settle[i]
        self.done = bool(self.collision or np.all(self.arrived) or
                         self.time >= self.deadline + c.timeout_margin - 1e-9)

    def step(self, actions):
        if self.done:
            raise RuntimeError("reset the environment after an episode ends")
        actions = np.asarray(actions, dtype=float)
        if actions.shape != (self.n, 3) or not np.isfinite(actions).all():
            raise ValueError(f"expected finite actions with shape {(self.n, 3)}")
        actions = np.clip(actions, -1, 1)
        c = self.config
        old_distance, was_arrived, start_time = self._distances(), self.arrived.copy(), self.time
        references = [self.generator(self.states[i], actions[i], self.goals[i]) for i in range(self.n)]
        for tick in range(self.n_ctrl):
            control_started = time.perf_counter()
            controls = np.zeros((self.n, 2))
            for i in range(self.n):
                if self.arrived[i]:
                    controls[i] = [-c.max_brake, 0.0]
                elif c.controller == "pp":
                    # Exactly drop MPC: execute the existing pursuit rollout control.
                    # The reference and actor are refreshed from measured state at 5 Hz.
                    controls[i] = references[i][1][tick]
                else:
                    controls[i] = self.trackers[i].control(
                        self.states[i], *references[i], offset=tick,
                        neighbors=self.states[self.visible(i, actor=False)])
                    stats = self.trackers[i].last_info
                    self.control_count += 1
                    self.fallback_count += int(stats.get("fallback", False))
                    elapsed = float(stats.get("solve_time", 0.0))
                    self.solve_time_total += elapsed
                    self.solve_time_max = max(self.solve_time_max, elapsed)
                    self.solve_times.append(elapsed)
            self.control_interval_times.append(time.perf_counter() - control_started)
            for _ in range(self.n_phys):
                previous = self.states.copy()
                controls[self.arrived] = [-c.max_brake, 0.0]
                if self.plant is None:
                    self.states = np.stack([bicycle_step(s, u, c.physics_dt, c)
                                            for s, u in zip(self.states, controls)])
                else:
                    self.states = self.plant.step(controls, c.physics_dt)
                self.time += c.physics_dt
                self._events(previous)
                if self.done:
                    break
            if self.done:
                break
        distance = self._distances()
        terms = {
            "progress": 3.0 * float(np.mean((old_distance - distance) / self.initial_dist)),
            "schedule": -c.schedule_weight * float(np.mean(np.abs(
                1 - distance / self.initial_dist - min(self.time / self.deadline, 1.0)))),
            "elapsed": -0.01 * (self.time - start_time),
            "arrival": 0.0, "completion": 0.0, "collision": 0.0, "timeout": 0.0,
        }
        # A departed/re-entered goal must not pay the same car's arrival bonus
        # repeatedly. Parking validity is still rechecked throughout the episode.
        newly = self.arrived & ~was_arrived & ~self._arrival_bonus_paid
        if newly.any():
            error = (self.arrival_times[newly] - self.deadline) / c.deadline_window
            terms["arrival"] = 5.0 * float(np.exp(-0.5 * error ** 2).sum()) / self.n
            self._arrival_bonus_paid[newly] = True
        if self.collision:
            terms["collision"] = -25.0
        elif np.all(self.arrived):
            error = float(np.max(np.abs(self.arrival_times - self.deadline)))
            spread = float(np.ptp(self.arrival_times))
            terms["completion"] = 10.0 * math.exp(-error / c.deadline_window - spread / c.synchrony_window)
        elif self.done:
            terms["timeout"] = -10.0 * float(np.mean(~self.arrived))
        # Same task reward in both variants. Solver fallback is a diagnostic,
        # not a controller-specific reward that would confound the comparison.
        reward = float(sum(terms.values()))
        self.episode_reward += reward
        info = self.info()
        info["reward_terms"] = terms
        return self.observe(), np.full(self.n, reward, dtype=np.float32), self.done, info

    def info(self):
        all_arrived = bool(np.all(self.arrived))
        spread = float(np.ptp(self.arrival_times)) if all_arrived else None
        error = float(np.max(np.abs(self.arrival_times - self.deadline))) if all_arrived else None
        c = self.config
        valid_arrivals = self.arrival_times[np.isfinite(self.arrival_times)]
        early = int(np.sum(valid_arrivals < self.deadline - c.deadline_window))
        late = int(np.sum(valid_arrivals > self.deadline + c.deadline_window))
        return {
            "time": self.time, "deadline": self.deadline, "family": self.family,
            "arena_half": self.scenario.arena_half,
            "arrival_times": [float(t) if np.isfinite(t) else None for t in self.arrival_times],
            "arrived": self.arrived.tolist(),
            "final_goal_distances": self._distances().tolist(),
            "initial_distances": self.initial_dist.tolist(),
            "early_arrival_count": early, "late_arrival_count": late,
            "on_time_arrival_count": len(valid_arrivals) - early - late,
            "all_arrived": all_arrived, "arrival_fraction": float(self.arrived.mean()),
            "arrival_spread": spread, "max_deadline_error": error,
            "success": bool(all_arrived and not self.collision and error <= c.deadline_window
                            and spread <= c.synchrony_window),
            "collision": bool(self.collision),
            "physical_collision": bool(self.physical_collision),
            "contact_pairs": sorted(getattr(self.plant, "contact_pairs", ())),
            "timeout": bool(self.done and not all_arrived and not self.collision),
            "min_separation": self.min_separation if np.isfinite(self.min_separation) else None,
            "fallback_rate": self.fallback_count / max(self.control_count, 1),
            "mpc_solve_count": self.control_count,
            "mpc_solve_mean_seconds": self.solve_time_total / max(self.control_count, 1),
            "mpc_solve_max_seconds": self.solve_time_max,
            "episode_reward": self.episode_reward,
        }
