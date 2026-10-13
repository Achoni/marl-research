"""Convert an RL waypoint/speed action into a nominal timed trajectory.

Pure pursuit rolls the nominal bicycle forward from the measured state once
per policy step. In the PP experiment the first policy interval's reference
controls are applied directly. In the MPC experiment the same states/controls
are given to the tracker. Neither path guarantees arrival within the reference
horizon, exact stopping, or collision avoidance.
"""
import math
import numpy as np

from .model import bicycle_step, bound_control


def decode_action(state, action, cfg):
    """Decode normalized [forward, lateral, speed] in the current car frame."""
    action = np.asarray(action, dtype=float)
    if action.shape != (3,) or not np.isfinite(action).all():
        raise ValueError("action must contain three finite numbers")
    action = np.clip(action, -1.0, 1.0)
    forward = cfg.waypoint_forward_min + 0.5 * (action[0] + 1.0) * (
        cfg.waypoint_forward_max - cfg.waypoint_forward_min)
    lateral = action[1] * cfg.waypoint_lateral_max
    speed = 0.5 * (action[2] + 1.0) * cfg.max_speed
    c, s = math.cos(state[2]), math.sin(state[2])
    waypoint = np.array([state[0] + c * forward - s * lateral,
                         state[1] + s * forward + c * lateral])
    return waypoint, speed


def pure_pursuit(state, target, target_speed, dt, cfg):
    """Bounded geometric steering plus proportional speed control."""
    dx, dy = np.asarray(target) - state[:2]
    lateral = -math.sin(state[2]) * dx + math.cos(state[2]) * dy
    distance_sq = max(dx * dx + dy * dy, 1e-5)
    steer = np.clip(math.atan2(2.0 * cfg.wheelbase * lateral, distance_sq),
                    -cfg.max_steer, cfg.max_steer)
    return bound_control([
        cfg.pursuit_speed_gain * (target_speed - state[3]),
        (steer - state[4]) / dt,
    ], cfg)


class ReferenceGenerator:
    """Fixed-horizon rollout with braking caps and a goal-switching heuristic.

    The decoded waypoint stays fixed in world coordinates throughout a rollout.
    Switching is reconsidered at each predicted state: pursue an ahead-of-car
    goal when it is nearer than the waypoint or the waypoint is captured. A
    goal already inside its tolerance may be pursued from either direction.
    Outside that parking region, a goal behind the car must not replace the
    actor's lateral waypoint: a forward-only car needs that waypoint to turn.
    This is a geometric heuristic, not a guarantee of passing the waypoint or
    reaching the goal before the horizon ends.
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self.dt = cfg.control_dt
        self.K = int(round(cfg.reference_horizon / self.dt))

    def __call__(self, state, action, goal):
        state = np.asarray(state, dtype=float)
        goal = np.asarray(goal, dtype=float)

        if state.shape != (5,) or goal.shape != (2,):
            raise ValueError("expected state shape (5,) and goal shape (2,)")
        if not np.isfinite(state).all() or not np.isfinite(goal).all():
            raise ValueError("state and goal must be finite")
        
        cfg = self.cfg
        waypoint, preferred = decode_action(state, action, cfg)
        states = np.empty((self.K + 1, 5))
        controls = np.empty((self.K, 2))
        states[0] = state
        for k in range(self.K):
            current = states[k]
            goal_delta = goal - current[:2]
            goal_distance = np.linalg.norm(goal_delta)
            waypoint_distance = np.linalg.norm(waypoint - current[:2])
            goal_forward = (math.cos(current[2]) * goal_delta[0]
                            + math.sin(current[2]) * goal_delta[1])
            goal_cap = math.sqrt(2.0 * cfg.max_brake * max(
                0.0, goal_distance - 0.5 * cfg.goal_tolerance))
            goal_is_captured = goal_distance <= cfg.goal_tolerance
            handoff = (goal_forward >= 0.0
                       and (goal_distance <= waypoint_distance
                            or waypoint_distance < cfg.waypoint_capture))
            if goal_is_captured or handoff:
                target, speed = goal, min(preferred, goal_cap)
            else:
                waypoint_cap = math.sqrt(2.0 * cfg.max_brake * waypoint_distance)
                target, speed = waypoint, min(preferred, goal_cap, waypoint_cap)
            controls[k] = pure_pursuit(current, target, speed, self.dt, cfg)
            states[k + 1] = bicycle_step(current, controls[k], self.dt, cfg)
        return states, controls

    def window(self, ref_states, ref_controls, offset, n):
        """Slice a timed reference; extend a short tail with bounded braking."""
        if offset < 0 or n < 1:
            raise ValueError("offset must be nonnegative and n must be positive")
        states = np.asarray(ref_states, dtype=float)
        controls = np.asarray(ref_controls, dtype=float)
        if (states.ndim != 2 or states.shape[1] != 5 or controls.ndim != 2
                or controls.shape[1] != 2 or len(states) != len(controls) + 1):
            raise ValueError("reference must have shapes (K + 1, 5) and (K, 2)")
        if offset + n <= len(controls):
            return states[offset:offset + n + 1], controls[offset:offset + n]
        states, controls = list(states), list(controls)
        while len(controls) < offset + n:
            command = bound_control([
                -self.cfg.max_brake, -states[-1][4] / self.dt,
            ], self.cfg)
            controls.append(command)
            states.append(bicycle_step(states[-1], command, self.dt, self.cfg))
        return np.asarray(states[offset:offset + n + 1]), np.asarray(controls[offset:offset + n])
