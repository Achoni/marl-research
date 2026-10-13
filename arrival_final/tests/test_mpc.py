"""Small solver checks; no training or simulator installation required."""
from types import SimpleNamespace
import importlib.util
import unittest
import numpy as np

from ..config import EnvConfig
from ..mpc import MPCTracker
from ..reference import ReferenceGenerator


@unittest.skipUnless(importlib.util.find_spec("casadi") and importlib.util.find_spec("scipy"),
                     "MPC tests require optional scipy and casadi dependencies")
class MPCTrackerTests(unittest.TestCase):
    @staticmethod
    def setup_tracker(**kwargs):
        cfg = EnvConfig(controller="mpc", **kwargs)
        state = np.zeros(5)
        states, controls = ReferenceGenerator(cfg)(state, [1, 0, 1], [8, 0])
        return cfg, state, states, controls, MPCTracker(cfg)

    def test_nominal_motion_is_finite_and_accelerates(self):
        cfg, state, states, controls, tracker = self.setup_tracker()
        command = tracker.control(state, states, controls)
        self.assertFalse(tracker.last_info["fallback"])
        self.assertGreater(command[0], 0.0)
        self.assertLessEqual(command[0], cfg.max_accel)
        self.assertTrue(np.isfinite(command).all())

    def test_one_car_has_no_constraint_slots(self):
        _, state, states, controls, tracker = self.setup_tracker(n_cars=1)
        command = tracker.control(state, states, controls)
        self.assertEqual(tracker.j, 0)
        self.assertEqual(tracker.last_info["neighbor_count"], 0)
        self.assertFalse(tracker.last_info["fallback"])
        self.assertTrue(np.isfinite(command).all())

    def test_controller_neighbors_are_independent_of_actor_k(self):
        _, state, states, controls, tracker = self.setup_tracker(n_cars=4, max_neighbors=0)
        neighbors = np.array([[0, 4, 0, 0, 0], [0, -4, 0, 0, 0], [-4, 0, 0, 0, 0]])
        tracker.control(state, states, controls, neighbors=neighbors)
        self.assertEqual(tracker.j, 3)
        self.assertEqual(tracker.last_info["neighbor_count"], 3)
        self.assertFalse(tracker.last_info["fallback"])

    def test_invalid_solver_result_uses_finite_bounded_braking(self):
        cfg, state, states, controls, tracker = self.setup_tracker()
        tracker._minimize = lambda *args, **kwargs: SimpleNamespace(
            success=True, x=np.full(2 * tracker.n, np.nan), fun=0.0, nit=1, message="invalid")
        command = tracker.control(state, states, controls)
        self.assertTrue(tracker.last_info["fallback"])
        np.testing.assert_allclose(command, [-cfg.max_brake, 0.0])
        self.assertIsNone(tracker._warm)

    def test_warm_start_retries_cheaper_initially_infeasible_reference(self):
        _, state, states, controls, tracker = self.setup_tracker()
        tracker._warm = np.zeros(2 * tracker.n)
        # Discs sit along the body, so two cars nose-to-tail have their nearest discs
        # closer than their rear axles by the disc span. Place the neighbour far enough
        # that the all-zero guess is genuinely feasible, which is what this test needs.
        span = float(tracker.disc_offsets.max() - tracker.disc_offsets.min())
        neighbors = np.array([[tracker.clearance + span + 0.15, 0, 0, 0, 0]])
        guesses = []

        def stalled_solver(objective, guess, **kwargs):
            guesses.append(np.asarray(guess).copy())
            stationary = np.zeros(2 * tracker.n)
            return SimpleNamespace(success=True, x=stationary,
                                   fun=objective(stationary)[0], nit=1, message="stalled")

        tracker._minimize = stalled_solver
        tracker.control(state, states, controls, neighbors=neighbors)
        self.assertEqual(len(guesses), 2)
        self.assertFalse(np.array_equal(guesses[0], guesses[1]))
        self.assertFalse(tracker.last_info["fallback"])


if __name__ == "__main__":
    unittest.main()
