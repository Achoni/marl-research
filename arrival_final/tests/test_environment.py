"""Behavioral checks shared unchanged by the PP and MPC experiment folders.

These tests select controller='pp' explicitly: importing or running them never
requires MPC, SciPy, or CasADi. They focus on task semantics and controller flow.
"""
import importlib
import math
import unittest
from unittest.mock import patch

import numpy as np

from ..config import EnvConfig, EGO_DIM, CRITIC_DIM, CRITIC_FEATURES
from ..env import ArrivalEnv
from ..model import body_discs
from ..model import bicycle_step


class EnvironmentTests(unittest.TestCase):
    def test_absolute_deadline_and_history_are_visible_to_critic(self):
        env = ArrivalEnv(EnvConfig(controller="pp", family="converging"), seed=7)
        env.reset()
        self.assertEqual(env.observe()["ego"].shape, (3, EGO_DIM))
        self.assertEqual(env.critic_state().shape, (3, CRITIC_DIM))
        early = env.critic_state()
        env.deadline *= 2
        later = env.critic_state()
        np.testing.assert_array_equal(early[:, :8], later[:, :8])
        self.assertFalse(np.array_equal(early[:, 8:10], later[:, 8:10]))
        env.initial_dist[0] *= 2
        self.assertNotEqual(later[0, 10], env.critic_state()[0, 10])
        env._settle[0] = env.time - .1
        self.assertAlmostEqual(env.critic_state()[0, CRITIC_FEATURES.index("dwell_fraction")], .5)

    def test_arrival_bonus_cannot_be_farmed_by_leaving_and_reentering(self):
        env = self.make_env([[-1, 0, 0, 0, 0], [0, 4, 0, 0, 0]], [[0, 0], [4, 4]])
        env.states[0, :2] = env.goals[0]
        env.time = 9.8
        stay = np.array([[1, 0, -1], [1, 0, -1]], dtype=float)
        paid = 0.0
        for _ in range(2):
            _, _, _, info = env.step(stay)
            paid += info["reward_terms"]["arrival"]
        self.assertTrue(env.arrived[0])
        self.assertGreater(paid, 0.0)
        env.states[0, 0] = .6
        env._events(env.states.copy())
        self.assertFalse(env.arrived[0])
        env.states[0, 0] = 0.0
        for _ in range(2):
            _, _, _, info = env.step(stay)
            self.assertEqual(info["reward_terms"]["arrival"], 0.0)
        self.assertTrue(env.arrived[0])

    def make_env(self, states, goals, **settings):
        cfg = EnvConfig(controller="pp", n_cars=len(states), **settings)
        env = ArrivalEnv(cfg, seed=19)
        env.reset()
        env.states = np.asarray(states, dtype=float)
        env.goals = np.asarray(goals, dtype=float)
        env.initial_dist = np.maximum(env._distances(), 1.0)
        env.deadline = 10.0
        env.min_separation = env._pairwise_min()
        return env

    def test_head_on_footprints_collide_at_point_eight_meters(self):
        env = self.make_env(
            [[-0.4, 0, 0, 0, 0], [0.4, 0, math.pi, 0, 0]],
            [[3, 0], [-3, 0]],
        )
        _, _, done, info = env.step(np.full((2, 3), -1.0))
        self.assertTrue(done)
        self.assertTrue(info["collision"])
        self.assertFalse(info["success"])
        self.assertLessEqual(info["min_separation"], 0.8)

    def test_swept_collision_is_detected_between_separated_endpoints(self):
        env = self.make_env(
            [[1, 0, 0, 0, 0], [-1, 0, math.pi, 0, 0]],
            [[3, 0], [-3, 0]],
        )
        # The cars swapped sides during the sub-step: they start at -1/+1 and end at +1/-1,
        # so the straight-line sweep passes through zero separation even though both
        # endpoints look clear. `previous` is now full states, and _pairwise_min reports a
        # FOOTPRINT GAP (disc-centre distance minus the two radii), so a clear pair is
        # positive and an overlap is negative.
        previous = np.array([[-1.0, 0.0, 0.0, 0.0, 0.0],
                             [1.0, 0.0, math.pi, 0.0, 0.0]])
        offsets, radius = body_discs(env.config)
        # both cars face outward, so their NEAREST discs are the rearmost ones
        expected = 2.0 + 2 * float(offsets.min()) - 2 * radius
        self.assertAlmostEqual(env._pairwise_min(), expected, places=6)
        self.assertLess(env._pairwise_min(previous), 0.0)
        env._events(previous)
        self.assertTrue(env.collision)
        self.assertTrue(env.done)

    def test_deadline_and_synchrony_are_separate_requirements(self):
        env = self.make_env([[0, 0, 0, 0, 0], [0, 3, 0, 0, 0]], [[4, 0], [4, 3]])
        env.arrived[:] = True
        env.arrival_times[:] = [9.5, 10.5]
        info = env.info()
        self.assertLess(info["max_deadline_error"], env.config.deadline_window)
        self.assertGreater(info["arrival_spread"], env.config.synchrony_window)
        self.assertFalse(info["success"])
        env.arrival_times[:] = [9.8, 10.2]
        self.assertTrue(env.info()["success"])
        env.arrival_times[:] = [10.7, 10.7]
        self.assertEqual(env.info()["arrival_spread"], 0.0)
        self.assertFalse(env.info()["success"])
        env.arrival_times[:] = [10, 10]
        env.collision = True
        self.assertFalse(env.info()["success"])

    def test_dwell_requires_stopping_and_resets_on_departure(self):
        env = self.make_env([[0, 0, 0, 0, 0], [0, 4, 0, 0, 0]], [[0, 0], [4, 4]])
        previous = env.states.copy()
        # Signed backward speed must not count as stopped.
        env.states[0, 3] = -0.5
        env.time = 1.0
        env._events(previous)
        self.assertTrue(np.isnan(env._settle[0]))
        env.states[0, 3] = 0.0
        env._events(previous)
        self.assertFalse(env.arrived[0])
        env.time = 1.0 + env.config.dwell_time - 0.01
        env._events(previous)
        self.assertFalse(env.arrived[0])
        env.time = 1.0 + env.config.dwell_time
        env._events(previous)
        self.assertTrue(env.arrived[0])
        self.assertEqual(env.arrival_times[0], 1.0)
        self.assertFalse(env.done)  # The other car still needs to arrive.
        env.states[0, 0] = env.config.goal_tolerance + 0.01
        env.time += 0.01
        env._events(previous)
        self.assertFalse(env.arrived[0])
        self.assertTrue(np.isnan(env.arrival_times[0]))
        self.assertTrue(np.isnan(env._settle[0]))

    def test_stopping_distance_must_fit_inside_goal_tolerance(self):
        env = self.make_env([[0, 0, 0, 0, 0]], [[1, 0]])
        env.states[0, 0] = 1.0 - env.config.goal_tolerance + 1e-5
        env.states[0, 3] = env.config.stop_speed
        env.time = 2.0
        env._events(env.states.copy())
        self.assertTrue(np.isnan(env._settle[0]))

    def test_actor_observation_has_only_local_peer_state(self):
        env = self.make_env(
            [[0, 0, math.pi / 2, 0.2, 0.1],
             [0, 1, math.pi / 2, 0.4, 0],
             [8, 8, 0, 0.5, 0]],
            [[0, 4], [4, 1], [-4, -4]], max_neighbors=4, sensing_radius=3.0,
        )
        before = env.observe()
        np.testing.assert_array_equal(before["mask"][0], [True, False, False, False])
        np.testing.assert_allclose(before["ego"][0, :2], [0.4, 0.0], atol=1e-7)
        np.testing.assert_allclose(before["neighbors"][0, 0], [1 / 3, 0, 0.05, 0, 0, 1], atol=1e-7)
        np.testing.assert_array_equal(before["neighbors"][0, 1:], 0.0)
        old_critic = env.critic_state()
        env.goals[1:] = [[-9, 4], [8, -7]]
        env.initial_dist[1:] = [99, 103]
        env.arrived[1:] = True
        env.states[2] = [20, -20, 1.7, 1.9, 0.4]
        self.assertFalse(np.array_equal(env.critic_state(), old_critic))
        with patch.object(env, "critic_state", side_effect=AssertionError("actor accessed critic")):
            after = env.observe()
        for name in ("ego", "neighbors", "mask"):
            np.testing.assert_array_equal(before[name][0], after[name][0])

    def test_actor_knn_cap_does_not_cap_controller_visibility(self):
        env = self.make_env(
            [[0, 0, 0, 0, 0], [0, 1.5, 0, 0, 0], [0, 3, 0, 0, 0]],
            [[4, 0], [4, 1.5], [4, 3]], max_neighbors=1,
        )
        np.testing.assert_array_equal(env.visible(0), [1])
        np.testing.assert_array_equal(env.visible(0, actor=False), [1, 2])

    def test_pp_executes_first_four_reference_controls_at_physics_rate(self):
        env = self.make_env([[0, 0, 0.2, 0.4, 0.05]], [[6, 3]])
        action = np.array([[0.6, 0.25, 0.3]])
        reference_states, reference_controls = env.generator(env.states[0], action[0], env.goals[0])
        initial = env.states[0].copy()
        module = importlib.import_module(ArrivalEnv.__module__)
        applied = []

        def record_step(state, control, dt, cfg):
            applied.append((np.asarray(control).copy(), dt))
            return bicycle_step(state, control, dt, cfg)

        with patch.object(module, "bicycle_step", side_effect=record_step):
            _, _, done, info = env.step(action)
        self.assertFalse(done)
        self.assertEqual(len(applied), 20)
        expected = initial
        for k, (control, dt) in enumerate(applied):
            np.testing.assert_array_equal(control, reference_controls[k // 5])
            self.assertEqual(dt, env.config.physics_dt)
            expected = bicycle_step(expected, reference_controls[k // 5], dt, env.config)
        np.testing.assert_allclose(env.states[0], expected, rtol=0, atol=1e-14)
        self.assertAlmostEqual(info["time"], 0.2)
        self.assertEqual(info["mpc_solve_count"], 0)
        # Distinguish physical integration from replacing state with a prediction.
        self.assertGreater(np.linalg.norm(env.states[0, :2] - reference_states[4, :2]), 1e-8)

    def test_single_car_reaches_and_stops_at_goal_with_pp(self):
        env = self.make_env([[0, 0, 0, 0, 0]], [[3, 0]])
        for _ in range(80):
            _, _, done, info = env.step([[1.0, 0.0, 0.0]])
            if done:
                break
        self.assertTrue(done)
        self.assertTrue(info["all_arrived"])
        self.assertFalse(info["collision"])
        self.assertFalse(info["timeout"])
        self.assertLessEqual(np.linalg.norm(env.states[0, :2] - env.goals[0]), env.config.goal_tolerance)
        self.assertLessEqual(abs(env.states[0, 3]), env.config.stop_speed)
        self.assertGreaterEqual(env.time - env.arrival_times[0], env.config.dwell_time - 1e-9)
        self.assertFalse(info["success"])  # Arriving well before T is not synchronization success.


if __name__ == "__main__":
    unittest.main()
