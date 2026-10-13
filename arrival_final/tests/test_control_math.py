"""Verify dynamics limits and the action/reference contract without training."""
import math
import unittest
import numpy as np

from ..config import EnvConfig
from ..model import bicycle_step, bound_control, wrap_angle
from ..reference import ReferenceGenerator, decode_action


class ControlMathTests(unittest.TestCase):
    def setUp(self):
        self.cfg = EnvConfig()
        self.generator = ReferenceGenerator(self.cfg)

    def test_straight_acceleration_and_state_limits(self):
        state = np.zeros(5)
        result = bicycle_step(state, [1.0, 0.0], 0.2, self.cfg)
        np.testing.assert_allclose(result, [0.02, 0, 0, 0.2, 0], atol=1e-12)
        state = np.array([0, 0, 0, 1.9, 0.4])
        result = bicycle_step(state, [100, 100], 0.5, self.cfg)
        self.assertEqual(result[3], self.cfg.max_speed)
        self.assertEqual(result[4], self.cfg.max_steer)
        stopped = bicycle_step(np.zeros(5), [-100, 0], 0.5, self.cfg)
        self.assertEqual(stopped[3], 0.0)

    def test_car_frame_action_decoding(self):
        state = np.array([2, 3, math.pi / 2, 0, 0])
        waypoint, speed = decode_action(state, [1, 1, 1], self.cfg)
        np.testing.assert_allclose(waypoint, [0.8, 5.0], atol=1e-12)
        self.assertEqual(speed, self.cfg.max_speed)
        np.testing.assert_allclose(bound_control([100, -100], self.cfg),
                                   [self.cfg.max_accel, -self.cfg.max_steer_rate])

    def test_finite_horizon_is_not_a_terminal_goal_constraint(self):
        states, controls = self.generator(np.zeros(5), [1, 0, 1], [10, 0])
        self.assertEqual(states.shape, (25, 5))
        self.assertEqual(controls.shape, (24, 2))
        self.assertLess(states[-1, 0], self.cfg.max_speed * self.cfg.reference_horizon)
        self.assertGreater(np.linalg.norm(states[-1, :2] - [10, 0]), 7.0)
        np.testing.assert_array_equal(states[0], np.zeros(5))

    def test_reference_is_equivariant_to_rigid_pose_changes(self):
        state = np.array([1.0, -0.5, 0.2, 0.4, 0.05])
        goal = np.array([4.0, 1.0])
        action = [0.1, 0.5, 0.4]
        states, commands = self.generator(state, action, goal)
        angle, translation = 1.0, np.array([2.0, -3.0])
        rotation = np.array([[math.cos(angle), -math.sin(angle)],
                             [math.sin(angle), math.cos(angle)]])
        moved_state = state.copy()
        moved_state[:2] = rotation @ state[:2] + translation
        moved_state[2] = wrap_angle(state[2] + angle)
        moved_goal = rotation @ goal + translation
        moved_states, moved_commands = self.generator(moved_state, action, moved_goal)
        np.testing.assert_allclose(moved_states[:, :2], states[:, :2] @ rotation.T + translation,
                                   atol=1e-10)
        np.testing.assert_allclose(wrap_angle(moved_states[:, 2] - states[:, 2]), angle,
                                   atol=1e-10)
        np.testing.assert_allclose(moved_commands, commands, atol=1e-10)

    def test_goal_at_rest_remains_at_rest(self):
        state = np.array([2.0, -1.0, 0.3, 0.0, 0.0])
        states, _ = self.generator(state, [1, 1, 1], state[:2])
        np.testing.assert_allclose(states, np.tile(state, (25, 1)), atol=1e-12)

    def test_behind_goal_does_not_erase_lateral_recovery_action(self):
        # Previously d_goal < d_waypoint sent PP straight toward the point
        # behind the car, yielding zero steering for both lateral actions.
        state = np.array([0, 0, 0, 0.2, 0], dtype=float)
        for lateral, sign in ((1.0, 1), (-1.0, -1), (0.1, 1)):
            with self.subTest(lateral=lateral):
                states, commands = self.generator(state, [-1, lateral, 0], [-0.5, 0])
                self.assertGreater(sign * commands[0, 1], 0)
                self.assertGreater(sign * states[1, 4], 0)

    def test_ahead_goal_handoff_still_brakes_to_park(self):
        states, commands = self.generator(np.zeros(5), [1, 1, 1], [0.15, 0])
        self.assertEqual(commands[0, 1], 0)
        self.assertLess(states[-1, 0], self.cfg.goal_tolerance)
        self.assertLess(states[-1, 3], 0.10)

    def test_rollout_controls_respect_actuator_and_state_limits(self):
        rng = np.random.default_rng(80)
        for _ in range(40):
            state = np.r_[rng.uniform(-2, 2, 2), rng.uniform(-math.pi, math.pi),
                          rng.uniform(0, self.cfg.max_speed),
                          rng.uniform(-self.cfg.max_steer, self.cfg.max_steer)]
            states, commands = self.generator(state, rng.uniform(-1, 1, 3),
                                               rng.uniform(-5, 5, 2))
            self.assertTrue(np.all(commands[:, 0] >= -self.cfg.max_brake))
            self.assertTrue(np.all(commands[:, 0] <= self.cfg.max_accel))
            self.assertTrue(np.all(np.abs(commands[:, 1]) <= self.cfg.max_steer_rate))
            self.assertTrue(np.all((states[:, 3] >= 0) & (states[:, 3] <= self.cfg.max_speed)))
            self.assertTrue(np.all(np.abs(states[:, 4]) <= self.cfg.max_steer))

    def test_reference_and_substepped_plant_agree_over_policy_interval(self):
        # Numerical integration rates differ, but the common ideal model should
        # not introduce material nominal tracking error in the applied interval.
        rng = np.random.default_rng(818)
        for _ in range(30):
            state = np.r_[rng.uniform(-2, 2, 2), rng.uniform(-math.pi, math.pi),
                          rng.uniform(0, self.cfg.max_speed),
                          rng.uniform(-self.cfg.max_steer, self.cfg.max_steer)]
            states, controls = self.generator(state, rng.uniform(-1, 1, 3),
                                               rng.uniform(-5, 5, 2))
            simulated = state.copy()
            ticks = round(self.cfg.policy_dt / self.cfg.control_dt)
            for control in controls[:ticks]:
                for _ in range(round(self.cfg.control_dt / self.cfg.physics_dt)):
                    simulated = bicycle_step(simulated, control, self.cfg.physics_dt, self.cfg)
            self.assertLess(np.linalg.norm(simulated[:2] - states[ticks, :2]), 0.005)
            self.assertAlmostEqual(simulated[3], states[ticks, 3], places=10)

    def test_reference_tail_brakes_and_does_not_mutate_input(self):
        states = np.array([[0.0, 0.0, 0.0, 0.4, 0.1]])
        original = states.copy()
        window_states, window_controls = self.generator.window(states, np.empty((0, 2)), 0, 8)
        self.assertEqual(window_states.shape, (9, 5))
        self.assertEqual(window_controls.shape, (8, 2))
        self.assertEqual(window_states[-1, 3], 0.0)
        self.assertAlmostEqual(window_states[-1, 4], 0.0)
        np.testing.assert_array_equal(states, original)


if __name__ == "__main__":
    unittest.main()
