"""Non-learning local reference policies with exactly the actor's observation API.

The lattice baseline ranks short bicycle/PP rollouts using deadline progress and
constant-velocity neighbours, then the SAME low-level controller executes its
chosen reference. This is a documented heuristic comparator, not a claimed
state-of-the-art planner or a collision/deadlock guarantee.
"""
from dataclasses import asdict, dataclass
import math

import numpy as np

from .model import body_discs, disc_centres
from .reference import ReferenceGenerator


@dataclass
class LatticeConfig:
    schedule_weight: float = 8.0
    clearance_weight: float = 150.0
    passing_bias: float = 0.02
    pacing_gain: float = 1.0

    def validate(self):
        if any(not math.isfinite(v) or v < 0 for v in asdict(self).values()):
            raise ValueError("baseline parameters must be nonnegative and finite")
        return self


def encode(forward, lateral, speed, cfg):
    return np.clip([2 * (forward - cfg.waypoint_forward_min) /
                    (cfg.waypoint_forward_max - cfg.waypoint_forward_min) - 1,
                    lateral / cfg.waypoint_lateral_max,
                    2 * speed / cfg.max_speed - 1], -1, 1)


class GoalTimePolicy:
    def __init__(self, cfg, gain=0.95):
        self.cfg, self.gain = cfg, gain

    def predict(self, observation):
        c = self.cfg
        actions = []
        for own in observation["ego"]:
            goal = own[:2] * 10
            speed = self.gain * np.linalg.norm(goal) / max(own[4] * 20, c.policy_dt)
            actions.append(encode(goal[0], goal[1], speed, c))
        return np.asarray(actions)


class LocalLatticePolicy:
    """Uses local observations only; no agent IDs, other goals or joint state.

    Candidates use five lateral offsets and three pacing speeds, plus stopping.
    A small shared right-passing preference breaks otherwise symmetric scores.
    It is a fixed convention, not an exchanged priority or agent identifier.
    """
    def __init__(self, cfg, parameters=None):
        self.cfg = cfg
        self.parameters = (parameters or LatticeConfig()).validate()
        self.generator = ReferenceGenerator(cfg)

    def _neighbor_forecasts(self, own, neighbors, mask):
        c = self.cfg
        rows = neighbors[mask]
        time = np.arange(1, self.generator.K + 1) * c.control_dt
        forecasts = []
        offsets, _ = body_discs(c)
        for row in rows:
            position = row[:2] * c.sensing_radius
            velocity = row[2:4] * (2 * c.max_speed) + [own[2] * c.max_speed, 0]
            yaw = math.atan2(row[4], row[5])
            centres = position + offsets[:, None] * np.array([math.cos(yaw), math.sin(yaw)])
            forecasts.append(centres[None, :, :] + time[:, None, None] * velocity)
        return forecasts

    def predict(self, observation):
        c, p = self.cfg, self.parameters
        actions = []
        for own, neighbors, mask in zip(observation["ego"], observation["neighbors"], observation["mask"]):
            if own[7] > 0.5:
                actions.append(np.array([-1.0, 0.0, -1.0]))
                continue
            goal = own[:2] * 10
            distance = float(np.linalg.norm(goal))
            remaining = max(float(own[4] * 20), c.policy_dt)
            state = np.array([0., 0., 0., own[2] * c.max_speed, own[3] * c.max_steer])
            desired = min(c.max_speed, p.pacing_gain * distance / remaining)
            forecasts = self._neighbor_forecasts(own, neighbors, mask)
            expected = distance * max(0.0, 1 - c.reference_horizon / remaining)
            forward = np.clip(goal[0], c.waypoint_forward_min, c.waypoint_forward_max)
            direct_lateral = np.clip(goal[1], -c.waypoint_lateral_max, c.waypoint_lateral_max)
            lateral_options = np.unique(np.clip(direct_lateral + np.array([-1.2, -.6, 0, .6, 1.2]),
                                                -c.waypoint_lateral_max, c.waypoint_lateral_max))
            candidates = [encode(forward, lat, min(c.max_speed, desired * scale), c)
                          for lat in lateral_options for scale in (0.6, 1.0, 1.4)]
            candidates.append(encode(forward, direct_lateral, 0.0, c))
            best_action, best_cost = None, math.inf
            _, radius = body_discs(c)
            for action in candidates:
                states, controls = self.generator(state, action, goal)
                end_distance = np.linalg.norm(states[-1, :2] - goal)
                cost = p.schedule_weight * ((end_distance - expected) / max(distance, 1.0)) ** 2
                cost += 0.1 * end_distance / max(distance, 1.0)
                cost += 0.002 * float(np.mean(controls ** 2))
                cost += p.passing_bias * max(0.0, action[1])
                ego_discs = np.stack([disc_centres(s, c) for s in states[1:]])
                for other in forecasts:
                    gaps = np.linalg.norm(ego_discs[:, :, None, :] - other[:, None, :, :], axis=-1)
                    deficits = np.maximum(0., 2 * radius + c.collision_margin - gaps)
                    cost += p.clearance_weight * float(np.mean(np.max(deficits, axis=(1, 2)) ** 2))
                    cost += 10 * p.clearance_weight * float(np.max(deficits) ** 2)
                if cost < best_cost:
                    best_action, best_cost = action, cost
            actions.append(best_action)
        return np.asarray(actions)
