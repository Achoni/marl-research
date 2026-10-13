"""Nominal nonlinear MPC with sampled neighbor-separation constraints.

This module is loaded only when controller="mpc". The controller uses the same
kinematic bicycle as the reference generator; it does not model slip or actuator
lag. Constant-velocity forecasts and a bounded braking fallback are practical
heuristics and do not establish a physical collision-avoidance guarantee.
"""
import time
import numpy as np

from .model import bound_control, body_discs
from .reference import ReferenceGenerator


class MPCTracker:
    """Single-shooting SLSQP with CasADi derivatives and shifted warm starts.

    All sensed neighbors fit the solver's ``n_cars - 1`` slots. The actor's
    separate ``max_neighbors`` setting never truncates controller constraints.
    Body-aligned discs cover the configured enclosing car rectangle.
    """

    def __init__(self, cfg):
        import casadi as ca
        from scipy.optimize import minimize

        self.cfg = cfg
        self._minimize = minimize
        self.n = int(round(cfg.mpc_horizon / cfg.control_dt))
        self.j = cfg.n_cars - 1
        self.dt = cfg.control_dt
        self.disc_offsets, self.disc_radius = body_discs(cfg)
        # One constraint per (ego disc, neighbour disc) pair per horizon step.
        self.clearance = 2.0 * self.disc_radius + cfg.collision_margin
        self._reference = ReferenceGenerator(cfg)
        self._lower = np.tile([-cfg.max_brake, -cfg.max_steer_rate], self.n)
        self._upper = np.tile([cfg.max_accel, cfg.max_steer_rate], self.n)
        self.bounds = list(zip(self._lower, self._upper))
        controls = ca.SX.sym("controls", 2 * self.n)
        # Each neighbor supplies its five sensed states and one presence mask.
        nb0 = 5 + 5 * (self.n + 1) + 2
        params = ca.SX.sym("params", nb0 + self.j * 6)
        state = params[:5]
        previous = params[nb0 - 2:nb0]
        cost = 0
        constraints = []
        for k in range(self.n):
            command = controls[2 * k:2 * k + 2]
            speed = ca.fmin(cfg.max_speed, ca.fmax(0, state[3] + self.dt * command[0]))
            steer = ca.fmin(cfg.max_steer, ca.fmax(-cfg.max_steer, state[4] + self.dt * command[1]))
            speed_mid = 0.5 * (state[3] + speed)
            steer_mid = 0.5 * (state[4] + steer)
            yaw_change = self.dt * speed_mid * ca.tan(steer_mid) / cfg.wheelbase
            state = ca.vertcat(
                state[0] + self.dt * speed_mid * ca.cos(state[2] + 0.5 * yaw_change),
                state[1] + self.dt * speed_mid * ca.sin(state[2] + 0.5 * yaw_change),
                state[2] + yaw_change, speed, steer,
            )
            reference = params[5 + 5 * (k + 1):5 + 5 * (k + 2)]
            yaw_error = ca.atan2(ca.sin(state[2] - reference[2]),
                                 ca.cos(state[2] - reference[2]))
            cost += 8.0 * ca.sumsqr(state[:2] - reference[:2])
            cost += 3.0 * (state[3] - reference[3]) ** 2
            cost += 0.5 * yaw_error ** 2 + 0.1 * (state[4] - reference[4]) ** 2
            cost += 0.015 * ca.sumsqr(command) + 0.04 * ca.sumsqr(command - previous)
            previous = command
            constrain_here = (k % cfg.constraint_stride == 0) or (k == self.n - 1)
            for j in range(self.j if constrain_here else 0):
                other = params[nb0 + 6 * j:nb0 + 6 * j + 6]
                heading = ca.vertcat(ca.cos(other[2]), ca.sin(other[2]))
                predicted_position = other[:2] + (k + 1) * self.dt * other[3] * heading
                ego_heading = ca.vertcat(ca.cos(state[2]), ca.sin(state[2]))
                for ego_offset in self.disc_offsets:
                    ego_disc = state[:2] + float(ego_offset) * ego_heading
                    for other_offset in self.disc_offsets:
                        other_disc = predicted_position + float(other_offset) * heading
                        separation = ca.sumsqr(ego_disc - other_disc) - self.clearance ** 2
                        # a masked-out slot contributes 1 >= 0, i.e. no constraint
                        constraints.append(other[5] * separation + (1.0 - other[5]))
        cost += 16.0 * ca.sumsqr(state[:2] - reference[:2])
        cost += 6.0 * (state[3] - reference[3]) ** 2
        constraints = ca.vertcat(*constraints) if constraints else ca.SX.zeros(0, 1)
        self._obj = ca.Function("tracking_cost", [controls, params], [cost, ca.gradient(cost, controls)])
        self._con = ca.Function("separation", [controls, params],
                                [constraints, ca.jacobian(constraints, controls)])
        self.reset()

    def reset(self):
        self._warm = None
        self._previous = np.zeros(2)
        self.last_info = dict(fallback=False, solve_time=0.0, neighbor_count=0, iterations=0)

    def _fallback(self, state):
        # An invalid measurement cannot support a steering correction.
        steer = state[4] if np.isfinite(state[4]) else 0.0
        return bound_control([-self.cfg.max_brake, -steer / self.dt], self.cfg)

    def control(self, state, ref_states, ref_controls, offset=0, neighbors=None):
        started = time.perf_counter()
        state = np.asarray(state, dtype=float)
        if state.shape != (5,):
            raise ValueError("MPC state must have shape (5,)")
        references, commands = self._reference.window(ref_states, ref_controls, offset, self.n)
        others = np.empty((0, 5)) if neighbors is None else np.asarray(neighbors, dtype=float).reshape(-1, 5)
        if len(others) > self.j:
            raise ValueError("more neighbors supplied than configured n_cars - 1")
        finite_inputs = np.isfinite(state).all() and np.isfinite(others).all()
        if finite_inputs and len(others):
            distance = np.linalg.norm(others[:, :2] - state[:2], axis=1)
            others = others[distance <= self.cfg.sensing_radius]
        padded = np.zeros((self.j, 6))
        padded[:len(others), :5] = others
        padded[:len(others), 5] = 1.0
        params = np.concatenate([state, references.ravel(), self._previous, padded.ravel()])
        reference_guess = np.clip(commands.ravel(), self._lower, self._upper)
        initial = reference_guess.copy() if self._warm is None else np.clip(
            self._warm, self._lower, self._upper)

        def objective(candidate):
            value, gradient = self._obj(candidate, params)
            return float(value), np.asarray(gradient).ravel()

        def constraint(candidate):
            return np.asarray(self._con(candidate, params)[0]).ravel()

        def constraint_jacobian(candidate):
            return np.asarray(self._con(candidate, params)[1])

        def feasible(candidate):
            if not len(others):
                return True
            residuals = constraint(candidate)
            return bool(np.isfinite(residuals).all() and np.min(residuals) >= -1e-6)

        def valid_result(result):
            return bool(result.success and np.isfinite(result.fun)
                        and np.isfinite(result.x).all()
                        and np.all(result.x >= self._lower - 1e-7)
                        and np.all(result.x <= self._upper + 1e-7)
                        and feasible(result.x))

        result = None
        valid = False
        message = "nonfinite controller input"
        iterations = 0
        if finite_inputs and np.isfinite(params).all() and np.isfinite(reference_guess).all():
            try:
                if len(others) and not feasible(initial):
                    brake_guess = np.tile(self._fallback(state), self.n)
                    if feasible(brake_guess):
                        initial = brake_guess
                constraints = [dict(type="ineq", fun=constraint, jac=constraint_jacobian)] if len(others) else ()

                def solve(guess):
                    return self._minimize(
                        objective, guess, jac=True, method="SLSQP", bounds=self.bounds,
                        constraints=constraints,
                        options=dict(maxiter=self.cfg.mpc_maxiter, ftol=2e-4, disp=False))

                result = solve(initial)
                iterations += int(getattr(result, "nit", 0))
                valid = valid_result(result)
                # At zero speed SLSQP can remain at a degenerate shifted guess.
                # A reference guess can start infeasible: SLSQP must then find
                # a feasible nearby trajectory. Requiring a feasible guess
                # here would preserve the stalled warm-start trap whenever the
                # requested motion initially intersects a neighbor forecast.
                # Only a strictly validated final solution is ever executed.
                if (not np.array_equal(initial, reference_guess)
                        and (not valid or objective(reference_guess)[0] < result.fun - 1e-3)):
                    second = solve(reference_guess)
                    iterations += int(getattr(second, "nit", 0))
                    if valid_result(second) and (not valid or second.fun < result.fun):
                        result, valid = second, True
                message = str(result.message)
            except (ValueError, RuntimeError, FloatingPointError) as exc:
                valid = False
                message = str(exc)
        candidate = result.x if valid else np.tile(self._fallback(state), self.n)
        command = bound_control(candidate[:2], self.cfg)
        self._warm = np.concatenate([candidate[2:], candidate[-2:]]) if valid else None
        self._previous = command.copy()
        self.last_info = dict(fallback=not valid, solve_time=time.perf_counter() - started,
                              neighbor_count=len(others), iterations=iterations, message=message)
        return command
