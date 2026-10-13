"""Behaviour tests for your own MPC in mpc_exercise.py.

    pytest arrival_v2/tests/test_mpc_exercise.py -v

Unimplemented stages SKIP, so the suite stays readable while you work. Implement
in order; each class is one stage of MPC_TUTORIAL.md.

These check physics and numerics, not code shape. Any correct implementation
passes, whatever it looks like inside.
"""
from dataclasses import replace

import numpy as np
import pytest

from ..config import EnvConfig
from ..model import bicycle_step, body_discs, wrap_angle
from ..reference import ReferenceGenerator
from .. import mpc_exercise as ex


def cfg(**changes):
    return replace(EnvConfig(), **changes).validate() if changes else EnvConfig()


def horizon(configuration):
    return int(round(configuration.mpc_horizon / configuration.control_dt))


def stage(call, *args, **kwargs):
    """Run `call`, or skip the test if that stage is not implemented yet."""
    try:
        return call(*args, **kwargs)
    except NotImplementedError as exc:
        pytest.skip(str(exc))


def straight_reference(configuration, speed=1.0, n=None):
    """A reference that drives straight along +x at constant speed."""
    n = horizon(configuration) if n is None else n
    state = np.array([0.0, 0.0, 0.0, speed, 0.0])
    states, controls = [state.copy()], []
    for _ in range(n):
        command = np.zeros(2)
        controls.append(command)
        state = bicycle_step(state, command, configuration.control_dt, configuration)
        states.append(state.copy())
    return np.asarray(states), np.asarray(controls)


# ---------------------------------------------------------------------------
# Stage 1
# ---------------------------------------------------------------------------

class TestStage1Rollout:

    def test_shape_and_first_row_is_the_given_state(self):
        c = cfg()
        state = np.array([1.0, -2.0, 0.3, 1.2, 0.05])
        controls = np.zeros((horizon(c), 2))
        states = stage(ex.rollout, state, controls, c)
        assert np.asarray(states).shape == (horizon(c) + 1, 5)
        assert np.allclose(states[0], state), "row 0 must be the state you were given"

    def test_matches_bicycle_step_exactly(self):
        """Your integrator must be the one the reference generator uses."""
        c = cfg()
        state = np.array([0.4, 0.1, -0.2, 0.9, -0.08])
        rng = np.random.default_rng(0)
        controls = np.column_stack([rng.uniform(-2.0, 1.5, horizon(c)),
                                    rng.uniform(-1.57, 1.57, horizon(c))])
        states = np.asarray(stage(ex.rollout, state, controls, c))
        expected = state.copy()
        for k, command in enumerate(controls):
            expected = bicycle_step(expected, command, c.control_dt, c)
            assert np.allclose(states[k + 1], expected, atol=1e-12), (
                f"step {k} diverges from model.bicycle_step -- most likely you used "
                f"Euler instead of the midpoint rule")

    def test_speed_never_leaves_zero_to_max(self):
        c = cfg()
        state = np.array([0.0, 0.0, 0.0, 0.5, 0.0])
        hard_accel = np.column_stack([np.full(horizon(c), 1.5), np.zeros(horizon(c))])
        states = np.asarray(stage(ex.rollout, state, hard_accel, c))
        assert states[:, 3].max() <= c.max_speed + 1e-9, "speed clip missing"
        hard_brake = np.column_stack([np.full(horizon(c), -2.0), np.zeros(horizon(c))])
        states = np.asarray(stage(ex.rollout, state, hard_brake, c))
        assert states[:, 3].min() >= -1e-9, "speed must not go negative"

    def test_steering_saturates_at_max_steer(self):
        c = cfg()
        state = np.array([0.0, 0.0, 0.0, 1.0, 0.0])
        controls = np.column_stack([np.zeros(horizon(c)),
                                    np.full(horizon(c), c.max_steer_rate)])
        states = np.asarray(stage(ex.rollout, state, controls, c))
        assert states[:, 4].max() <= c.max_steer + 1e-9, "steering clip missing"
        assert states[-1, 4] == pytest.approx(c.max_steer, abs=1e-6), (
            "16 steps at 90 deg/s should hit the 27 deg limit")

    def test_a_stopped_car_with_no_acceleration_does_not_move(self):
        c = cfg()
        state = np.array([1.0, 1.0, 0.5, 0.0, 0.2])
        controls = np.column_stack([np.zeros(horizon(c)),
                                    np.full(horizon(c), 0.5)])
        states = np.asarray(stage(ex.rollout, state, controls, c))
        assert np.allclose(states[:, :3], state[:3]), (
            "steering at zero speed must not translate or rotate the car")

    def test_positive_steering_turns_left(self):
        c = cfg()
        state = np.array([0.0, 0.0, 0.0, 1.5, 0.2])
        states = np.asarray(stage(ex.rollout, state, np.zeros((horizon(c), 2)), c))
        assert states[-1, 2] > 1e-3, "positive steering angle must increase heading"
        assert states[-1, 1] > 1e-3, "and move the car to +y"


# ---------------------------------------------------------------------------
# Stage 2
# ---------------------------------------------------------------------------

class TestStage2Cost:

    def test_perfect_tracking_costs_almost_nothing(self):
        c = cfg()
        ref_states, ref_controls = straight_reference(c)
        states = np.asarray(stage(ex.rollout, ref_states[0], ref_controls, c))
        cost = stage(ex.tracking_cost, states, ref_controls, ref_states, np.zeros(2), c)
        assert float(cost) == pytest.approx(0.0, abs=1e-9), (
            "following the reference exactly with zero commands must cost ~0")

    def test_cost_is_positive_and_grows_with_error(self):
        c = cfg()
        ref_states, ref_controls = straight_reference(c)
        costs = []
        for offset in (0.1, 0.5, 1.0):
            start = ref_states[0].copy()
            start[1] += offset
            states = np.asarray(stage(ex.rollout, start, ref_controls, c))
            costs.append(float(stage(ex.tracking_cost, states, ref_controls,
                                     ref_states, np.zeros(2), c)))
        assert all(value > 0 for value in costs)
        assert costs[0] < costs[1] < costs[2], "cost must increase with position error"

    def test_heading_error_is_wrapped(self):
        """psi and psi + 2*pi are the same heading and must cost the same."""
        c = cfg()
        ref_states, ref_controls = straight_reference(c, speed=0.0)
        states = np.asarray(stage(ex.rollout, ref_states[0], ref_controls, c))
        wrapped = states.copy()
        wrapped[:, 2] += 2.0 * np.pi
        plain = float(stage(ex.tracking_cost, states, ref_controls, ref_states,
                            np.zeros(2), c))
        turned = float(stage(ex.tracking_cost, wrapped, ref_controls, ref_states,
                             np.zeros(2), c))
        assert turned == pytest.approx(plain, abs=1e-6), (
            "heading error must be wrapped with wrap_angle, not squared raw")

    def test_terminal_state_is_weighted_harder_than_a_mid_state(self):
        """The doubled terminal weights are what stop end-of-horizon drift."""
        c = cfg()
        ref_states, ref_controls = straight_reference(c)
        n = horizon(c)
        base = np.asarray(stage(ex.rollout, ref_states[0], ref_controls, c))

        def cost_with_error_at(index):
            states = base.copy()
            states[index, 1] += 0.2
            return float(stage(ex.tracking_cost, states, ref_controls, ref_states,
                               np.zeros(2), c))

        assert cost_with_error_at(n) > cost_with_error_at(n // 2) + 1e-6, (
            "an error at the final step must cost more than the same error mid-horizon")

    def test_effort_and_smoothness_are_penalised(self):
        c = cfg()
        ref_states, ref_controls = straight_reference(c)
        states = np.asarray(stage(ex.rollout, ref_states[0], ref_controls, c))
        quiet = float(stage(ex.tracking_cost, states, ref_controls, ref_states,
                            np.zeros(2), c))
        busy_controls = ref_controls.copy()
        busy_controls[:, 1] = 0.8
        busy = float(stage(ex.tracking_cost, states, busy_controls, ref_states,
                           np.zeros(2), c))
        assert busy > quiet, "nonzero commands must cost something (W_EFFORT)"

    def test_a_jump_from_the_previous_command_costs_extra(self):
        c = cfg()
        ref_states, ref_controls = straight_reference(c)
        states = np.asarray(stage(ex.rollout, ref_states[0], ref_controls, c))
        matched = float(stage(ex.tracking_cost, states, ref_controls, ref_states,
                              np.zeros(2), c))
        jumped = float(stage(ex.tracking_cost, states, ref_controls, ref_states,
                             np.array([1.5, 1.5]), c))
        assert jumped > matched, (
            "previous_command must enter the smoothness term, coupling consecutive solves")


# ---------------------------------------------------------------------------
# Stage 3
# ---------------------------------------------------------------------------

class TestStage3Solve:

    def test_bounds_alternate_accel_then_steer_rate(self):
        c = cfg()
        n = horizon(c)
        lower, upper = stage(ex.control_bounds, c, n)
        lower, upper = np.asarray(lower), np.asarray(upper)
        assert lower.shape == upper.shape == (2 * n,)
        assert np.allclose(lower[0::2], -c.max_brake)
        assert np.allclose(upper[0::2], c.max_accel)
        assert np.allclose(lower[1::2], -c.max_steer_rate)
        assert np.allclose(upper[1::2], c.max_steer_rate)

    def test_returns_a_finite_in_bounds_command(self):
        c = cfg()
        ref_states, ref_controls = straight_reference(c)
        command = np.asarray(stage(ex.solve_unconstrained, ref_states[0], ref_states,
                                   ref_controls, np.zeros(2), c))
        assert command.shape == (2,)
        assert np.isfinite(command).all()
        assert -c.max_brake - 1e-7 <= command[0] <= c.max_accel + 1e-7
        assert abs(command[1]) <= c.max_steer_rate + 1e-7

    def test_on_reference_it_barely_acts(self):
        c = cfg()
        ref_states, ref_controls = straight_reference(c)
        command = np.asarray(stage(ex.solve_unconstrained, ref_states[0], ref_states,
                                   ref_controls, np.zeros(2), c))
        assert np.allclose(command, 0.0, atol=5e-2), (
            "already on the reference, the optimal command is near zero")

    def test_it_steers_toward_a_reference_offset_to_the_left(self):
        c = cfg()
        ref_states, ref_controls = straight_reference(c)
        start = ref_states[0].copy()
        start[1] -= 0.4                       # car is 0.4 m to the right of the path
        command = np.asarray(stage(ex.solve_unconstrained, start, ref_states,
                                   ref_controls, np.zeros(2), c))
        assert command[1] > 1e-3, "must steer left (positive rate) to close a +y error"

    def test_it_brakes_for_a_slower_reference(self):
        c = cfg()
        ref_states, ref_controls = straight_reference(c, speed=0.3)
        start = ref_states[0].copy()
        start[3] = 1.8                        # much faster than the reference
        command = np.asarray(stage(ex.solve_unconstrained, start, ref_states,
                                   ref_controls, np.zeros(2), c))
        assert command[0] < 0.0, "travelling faster than the reference, it must decelerate"

    def test_it_beats_the_reference_guess(self):
        """Optimising must actually improve on the initial guess."""
        c = cfg()
        ref_states, ref_controls = straight_reference(c)
        start = ref_states[0].copy()
        start[1] += 0.5
        start[2] += 0.3
        n = horizon(c)

        def cost_of(flat):
            controls = np.asarray(flat).reshape(n, 2)
            states = np.asarray(stage(ex.rollout, start, controls, c))
            return float(stage(ex.tracking_cost, states, controls, ref_states,
                               np.zeros(2), c))

        guess_cost = cost_of(ref_controls.ravel())
        command = np.asarray(stage(ex.solve_unconstrained, start, ref_states,
                                   ref_controls, np.zeros(2), c))
        improved = ref_controls.copy()
        improved[0] = command
        assert cost_of(improved.ravel()) <= guess_cost + 1e-6


# ---------------------------------------------------------------------------
# Stage 4
# ---------------------------------------------------------------------------

def pack_params(state, ref_states, previous):
    return np.concatenate([np.asarray(state, float).ravel(),
                           np.asarray(ref_states, float).ravel(),
                           np.asarray(previous, float).ravel()])


class TestStage4Casadi:

    def test_signature_and_shapes(self):
        c = cfg()
        n = horizon(c)
        function = stage(ex.build_cost_function, c)
        assert function.n_in() == 2, "expected (controls, params)"
        assert function.n_out() == 2, "expected (cost, gradient)"
        assert function.size1_in(0) == 2 * n, f"{2 * n} decision variables expected"
        assert function.size1_in(1) == 5 + 5 * (n + 1) + 2
        assert function.size1_out(0) == 1
        assert function.size1_out(1) == 2 * n

    def test_agrees_with_the_numpy_cost(self):
        """Stage 4 must be the same maths as stage 2, only differentiable."""
        c = cfg()
        n = horizon(c)
        function = stage(ex.build_cost_function, c)
        rng = np.random.default_rng(7)
        ref_states, _ = straight_reference(c)
        for trial in range(5):
            state = ref_states[0] + np.concatenate([rng.normal(0, 0.3, 3), [0.0, 0.0]])
            controls = np.column_stack([rng.uniform(-2.0, 1.5, n),
                                        rng.uniform(-1.5, 1.5, n)])
            previous = rng.uniform(-0.5, 0.5, 2)
            symbolic = float(function(controls.ravel(),
                                      pack_params(state, ref_states, previous))[0])
            states = np.asarray(stage(ex.rollout, state, controls, c))
            numeric = float(stage(ex.tracking_cost, states, controls, ref_states,
                                  previous, c))
            assert symbolic == pytest.approx(numeric, rel=1e-9, abs=1e-9), (
                f"trial {trial}: CasADi cost {symbolic} != NumPy cost {numeric}")

    def test_gradient_matches_finite_differences(self):
        c = cfg()
        n = horizon(c)
        function = stage(ex.build_cost_function, c)
        rng = np.random.default_rng(11)
        ref_states, _ = straight_reference(c)
        state = ref_states[0] + np.array([0.1, -0.2, 0.15, 0.0, 0.0])
        params = pack_params(state, ref_states, np.zeros(2))
        z = rng.uniform(-0.4, 0.4, 2 * n)
        gradient = np.asarray(function(z, params)[1]).ravel()
        step = 1e-6
        for index in rng.choice(2 * n, size=8, replace=False):
            plus, minus = z.copy(), z.copy()
            plus[index] += step
            minus[index] -= step
            numeric = (float(function(plus, params)[0])
                       - float(function(minus, params)[0])) / (2 * step)
            assert gradient[index] == pytest.approx(numeric, rel=2e-4, abs=2e-5), (
                f"gradient component {index} wrong")


# ---------------------------------------------------------------------------
# Stage 5
# ---------------------------------------------------------------------------

class TestStage5Constraints:

    def neighbour_params(self, c, state, ref_states, others):
        """others: list of (x, y, psi, v, delta, present) tuples, padded to j."""
        j = c.n_cars - 1
        padded = np.zeros((j, 6))
        for index, row in enumerate(others):
            padded[index] = row
        return np.concatenate([pack_params(state, ref_states, np.zeros(2)),
                               padded.ravel()])

    def test_constraint_count_matches_stride_and_discs(self):
        c = cfg(n_cars=6)
        n = horizon(c)
        function = stage(ex.build_constraint_function, c)
        offsets, _ = body_discs(c)
        steps = sum(1 for k in range(n)
                    if k % c.constraint_stride == 0 or k == n - 1)
        expected = steps * (c.n_cars - 1) * len(offsets) ** 2
        assert function.size1_out(0) == expected, (
            f"expected {expected} residuals = {steps} steps x {c.n_cars - 1} "
            f"neighbours x {len(offsets)}^2 disc pairs")
        assert expected == 180, "with the shipped defaults this is 180 at 6 cars"

    def test_masked_slots_contribute_exactly_one(self):
        """An absent neighbour must be a free, zero-gradient constraint."""
        c = cfg(n_cars=3)
        n = horizon(c)
        function = stage(ex.build_constraint_function, c)
        ref_states, ref_controls = straight_reference(c)
        params = self.neighbour_params(c, ref_states[0], ref_states, [])
        residuals, jacobian = function(ref_controls.ravel(), params)
        residuals = np.asarray(residuals).ravel()
        assert np.allclose(residuals, 1.0), (
            "with no neighbours present every residual must be exactly 1.0 -- "
            "if you see other values you are constraining a phantom car at the origin")
        assert np.allclose(np.asarray(jacobian), 0.0), (
            "masked constraints must have zero gradient")

    def test_a_distant_neighbour_is_satisfied(self):
        c = cfg(n_cars=3)
        function = stage(ex.build_constraint_function, c)
        ref_states, ref_controls = straight_reference(c)
        params = self.neighbour_params(
            c, ref_states[0], ref_states, [(0.0, 8.0, 0.0, 0.0, 0.0, 1.0)])
        residuals = np.asarray(function(ref_controls.ravel(), params)[0]).ravel()
        assert residuals.min() > 0.0, "a neighbour 8 m away constrains nothing"

    def test_an_overlapping_neighbour_is_violated(self):
        c = cfg(n_cars=3)
        function = stage(ex.build_constraint_function, c)
        ref_states, ref_controls = straight_reference(c)
        params = self.neighbour_params(
            c, ref_states[0], ref_states, [(0.3, 0.0, 0.0, 0.0, 0.0, 1.0)])
        residuals = np.asarray(function(ref_controls.ravel(), params)[0]).ravel()
        assert residuals.min() < 0.0, (
            "a neighbour 0.3 m ahead is inside the clearance and must violate")

    def test_the_forecast_moves_the_neighbour(self):
        """A neighbour driving away is less constraining than a stationary one."""
        c = cfg(n_cars=3)
        function = stage(ex.build_constraint_function, c)
        ref_states, ref_controls = straight_reference(c)
        still = self.neighbour_params(
            c, ref_states[0], ref_states, [(1.2, 0.0, 0.0, 0.0, 0.0, 1.0)])
        fleeing = self.neighbour_params(
            c, ref_states[0], ref_states, [(1.2, 0.0, 0.0, 2.0, 0.0, 1.0)])
        still_min = np.asarray(function(ref_controls.ravel(), still)[0]).ravel().min()
        fleeing_min = np.asarray(function(ref_controls.ravel(), fleeing)[0]).ravel().min()
        assert fleeing_min > still_min, (
            "constant-velocity forecast must carry the neighbour forward")

    def test_clearance_comes_from_body_discs(self):
        """Two discs must demand far less separation than one 0.5 m circle."""
        c = cfg(n_cars=3)
        function = stage(ex.build_constraint_function, c)
        ref_states, _ = straight_reference(c, speed=0.0)
        offsets, radius = body_discs(c)
        clearance = 2.0 * radius + c.collision_margin
        assert clearance == pytest.approx(0.4435, abs=1e-3)
        zero = np.zeros(2 * horizon(c))
        # Place a neighbour just inside and just outside the clearance, abeam so
        # the disc offsets cancel and the gap is purely lateral.
        inside = self.neighbour_params(
            c, ref_states[0], ref_states, [(0.0, clearance - 0.05, 0.0, 0.0, 0.0, 1.0)])
        outside = self.neighbour_params(
            c, ref_states[0], ref_states, [(0.0, clearance + 0.05, 0.0, 0.0, 0.0, 1.0)])
        assert np.asarray(function(zero, inside)[0]).ravel().min() < 0.0
        assert np.asarray(function(zero, outside)[0]).ravel().min() > 0.0

    def test_jacobian_matches_finite_differences(self):
        c = cfg(n_cars=3)
        n = horizon(c)
        function = stage(ex.build_constraint_function, c)
        ref_states, _ = straight_reference(c)
        params = self.neighbour_params(
            c, ref_states[0], ref_states, [(1.0, 0.25, 0.0, 0.5, 0.0, 1.0)])
        rng = np.random.default_rng(13)
        z = rng.uniform(-0.3, 0.3, 2 * n)
        jacobian = np.asarray(function(z, params)[1])
        step = 1e-6
        for index in rng.choice(2 * n, size=5, replace=False):
            plus, minus = z.copy(), z.copy()
            plus[index] += step
            minus[index] -= step
            numeric = (np.asarray(function(plus, params)[0]).ravel()
                       - np.asarray(function(minus, params)[0]).ravel()) / (2 * step)
            assert np.allclose(jacobian[:, index], numeric, rtol=2e-4, atol=2e-5), (
                f"constraint Jacobian column {index} wrong")


# ---------------------------------------------------------------------------
# Stage 6
# ---------------------------------------------------------------------------

class TestStage6Controller:

    def build(self, **changes):
        c = cfg(**changes)
        return c, stage(ex.MyMPC, c)

    def test_reset_initialises_diagnostics(self):
        c, controller = self.build()
        stage(controller.reset)
        for key in ("fallback", "solve_time", "neighbor_count", "iterations"):
            assert key in controller.last_info, f"last_info must report {key}"

    def test_returns_a_valid_command_with_no_neighbours(self):
        c, controller = self.build()
        stage(controller.reset)
        ref_states, ref_controls = straight_reference(c, n=horizon(c) + 4)
        command = np.asarray(stage(controller.control, ref_states[0], ref_states,
                                   ref_controls))
        assert command.shape == (2,)
        assert np.isfinite(command).all()
        assert -c.max_brake - 1e-7 <= command[0] <= c.max_accel + 1e-7
        assert abs(command[1]) <= c.max_steer_rate + 1e-7
        assert not controller.last_info["fallback"], "this problem is easy; no fallback"

    def test_a_stopped_car_still_accelerates_toward_a_moving_reference(self):
        """The degenerate warm-start trap: cost is flat in steering at v = 0.

        This is the bug that cost a car an entire episode -- SLSQP returned the
        shifted guess after one iteration at cost 44.599 when 0.606 was
        available. Call control() repeatedly so a warm start builds up.
        """
        c, controller = self.build()
        stage(controller.reset)
        ref_states, ref_controls = straight_reference(c, speed=1.0, n=horizon(c) + 8)
        state = ref_states[0].copy()
        state[3] = 0.0                        # dead stop, reference wants 1 m/s
        commands = []
        for _ in range(6):
            command = np.asarray(stage(controller.control, state, ref_states,
                                       ref_controls))
            commands.append(command)
            state = bicycle_step(state, command, c.control_dt, c)
        assert max(command[0] for command in commands) > 0.1, (
            "a stopped car must accelerate; if every command is ~0 you have hit the "
            "degenerate warm-start trap -- re-solve from the reference guess")
        assert state[3] > 0.05, "and it must actually start moving"

    def test_warm_start_does_not_leak_across_resets(self):
        c, controller = self.build()
        stage(controller.reset)
        ref_states, ref_controls = straight_reference(c, n=horizon(c) + 4)
        stage(controller.control, ref_states[0], ref_states, ref_controls)
        stage(controller.reset)
        assert getattr(controller, "_warm", None) is None, (
            "reset() must clear the warm start -- a trajectory from another episode "
            "is a bad initial guess")

    def test_nonfinite_state_falls_back_instead_of_propagating_nan(self):
        c, controller = self.build()
        stage(controller.reset)
        ref_states, ref_controls = straight_reference(c, n=horizon(c) + 4)
        broken = ref_states[0].copy()
        broken[0] = np.nan
        command = np.asarray(stage(controller.control, broken, ref_states, ref_controls))
        assert np.isfinite(command).all(), "a NaN must never reach the plant"
        assert controller.last_info["fallback"], "and the fallback must be reported"

    def test_fallback_brakes(self):
        c, controller = self.build()
        stage(controller.reset)
        state = np.array([0.0, 0.0, 0.0, 1.0, 0.1])
        command = np.asarray(stage(controller._fallback, state))
        assert command[0] < 0.0, "the fallback must decelerate"

    def test_it_yields_to_a_head_on_neighbour(self):
        c, controller = self.build(n_cars=3)
        stage(controller.reset)
        ref_states, ref_controls = straight_reference(c, speed=1.5, n=horizon(c) + 8)
        state = ref_states[0].copy()
        # Another car 1.2 m ahead driving straight at us.
        neighbours = np.array([[1.2, 0.0, np.pi, 1.5, 0.0]])
        free = np.asarray(stage(controller.control, state, ref_states, ref_controls))
        stage(controller.reset)
        blocked = np.asarray(stage(controller.control, state, ref_states, ref_controls,
                                   0, neighbours))
        assert np.isfinite(blocked).all()
        assert (blocked[0] < free[0] - 1e-3) or (abs(blocked[1]) > abs(free[1]) + 1e-3), (
            "facing a head-on neighbour the controller must brake or steer away; "
            "an identical command means the constraints are not binding")
        assert controller.last_info["neighbor_count"] == 1

    def test_neighbours_beyond_the_sensing_radius_are_dropped(self):
        c, controller = self.build(n_cars=3)
        stage(controller.reset)
        ref_states, ref_controls = straight_reference(c, n=horizon(c) + 4)
        far = np.array([[c.sensing_radius + 3.0, 0.0, 0.0, 0.0, 0.0]])
        stage(controller.control, ref_states[0], ref_states, ref_controls, 0, far)
        assert controller.last_info["neighbor_count"] == 0, (
            "filter neighbours to cfg.sensing_radius before building params")

    def test_too_many_neighbours_is_an_error(self):
        c, controller = self.build(n_cars=3)
        stage(controller.reset)
        ref_states, ref_controls = straight_reference(c, n=horizon(c) + 4)
        crowd = np.tile([1.0, 1.0, 0.0, 0.0, 0.0], (c.n_cars, 1))
        with pytest.raises(ValueError):
            controller.control(ref_states[0], ref_states, ref_controls, 0, crowd)

    def test_a_short_reference_tail_is_handled(self):
        """Near the goal the reference runs out; window() extends it by braking."""
        c, controller = self.build()
        stage(controller.reset)
        ref_states, ref_controls = straight_reference(c, n=3)
        command = np.asarray(stage(controller.control, ref_states[0], ref_states,
                                   ref_controls))
        assert np.isfinite(command).all(), (
            "use ReferenceGenerator.window() so a short reference is extended")
