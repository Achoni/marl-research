"""Checks for distribution coverage and physical endpoint validity."""
import math
import unittest

import numpy as np

from ..config import EnvConfig
from ..scenarios import FAMILY_WEIGHTS, ScenarioSampler


class ScenarioTests(unittest.TestCase):
    def test_same_seed_same_missions_independent_of_controller(self):
        pp = ScenarioSampler(EnvConfig(controller="pp"), seed=91)
        mpc = ScenarioSampler(EnvConfig(controller="mpc"), seed=91)
        for _ in range(30):
            a, b = pp.sample(), mpc.sample()
            np.testing.assert_array_equal(a.starts, b.starts)
            np.testing.assert_array_equal(a.goals, b.goals)
            self.assertEqual((a.deadline, a.family, a.arena_half, a.metadata),
                             (b.deadline, b.family, b.arena_half, b.metadata))
        pp.seed(91)
        fresh = ScenarioSampler(EnvConfig(), seed=91).sample()
        np.testing.assert_array_equal(pp.sample().starts, fresh.starts)

    def test_each_family_valid_for_single_three_and_eight_cars(self):
        for n in (1, 3, 8):
            for family in FAMILY_WEIGHTS:
                cfg = EnvConfig(n_cars=n, family=family, arena_half_min=4, arena_half_max=4)
                sampler = ScenarioSampler(cfg, seed=123)
                with self.subTest(n=n, family=family):
                    for _ in range(12):
                        scene = sampler.sample()
                        self.assertEqual(scene.starts.shape, (n, 5))
                        self.assertEqual(scene.goals.shape, (n, 2))
                        self.assertTrue(np.all(np.abs(scene.starts[:, :2]) <= scene.arena_half))
                        self.assertTrue(np.all(np.abs(scene.goals) <= scene.arena_half))
                        trips = np.linalg.norm(scene.goals - scene.starts[:, :2], axis=1)
                        self.assertTrue(np.all(trips >= cfg.min_trip))
                        self.assertTrue(np.all(scene.starts[:, 3] >= 0))
                        self.assertTrue(np.all(scene.starts[:, 3] <= cfg.initial_speed_max))
                        self.assertTrue(np.all(scene.starts[:, 4] == 0))
                        self.assertTrue(np.all(np.abs(scene.metadata["heading_errors"]) <= cfg.heading_jitter))
                        self.assertGreater(scene.deadline, float(trips.max() / cfg.max_speed))
                        if n > 1:
                            self.assertGreaterEqual(scene.metadata["minimum_start_separation"], sampler.start_gap)
                            self.assertGreaterEqual(scene.metadata["minimum_goal_separation"], sampler.goal_gap)

    def test_mixture_covers_off_center_corners_headings_and_label_permutations(self):
        sampler = ScenarioSampler(EnvConfig(), seed=42)
        scenes = [sampler.sample() for _ in range(400)]
        self.assertEqual({s.family for s in scenes}, {k for k, v in EnvConfig().family_weights.items() if v > 0})
        counts = {family: sum(s.family == family for s in scenes) for family in FAMILY_WEIGHTS}
        weights = EnvConfig().family_weights
        probability = weights["random"] / sum(weights.values())
        self.assertLess(abs(counts["random"] - 400 * probability),
                        4 * math.sqrt(400 * probability * (1 - probability)))
        normalized_goals = np.concatenate([s.goals / s.arena_half for s in scenes])
        self.assertGreater(np.mean(np.linalg.norm(normalized_goals, axis=1) > 0.75), 0.2)
        self.assertGreater(np.sum(np.all(np.abs(normalized_goals) > 0.7, axis=1)), 15)
        for sx, sy in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
            self.assertTrue(np.any((normalized_goals[:, 0] * sx > 0.7) & (normalized_goals[:, 1] * sy > 0.7)))
        errors = np.concatenate([s.metadata["heading_errors"] for s in scenes])
        self.assertGreater(np.sum(errors > math.radians(45)), 50)
        self.assertGreater(np.sum(errors < -math.radians(45)), 50)
        self.assertGreater(len({tuple(s.metadata["car_permutation"]) for s in scenes}), 4)
        self.assertGreater(max(s.arena_half for s in scenes) - min(s.arena_half for s in scenes), 2.8)

    def test_unequal_family_produces_short_and_long_trips(self):
        sampler = ScenarioSampler(EnvConfig(family="unequal"), seed=7)
        for _ in range(30):
            scene = sampler.sample()
            trips = np.asarray(scene.metadata["trip_lengths"])
            self.assertLessEqual(trips.min(), 2.0 + 1e-10)
            self.assertGreaterEqual(trips.max(), 0.85 * scene.arena_half)

    def test_custom_mixture_weights_are_normalized_and_zero_is_never_sampled(self):
        cfg = EnvConfig(family_weights={"unequal": 3, "parallel": 1, "random": 0})
        sampler = ScenarioSampler(cfg, seed=728)
        families = [sampler.sample().family for _ in range(160)]
        self.assertEqual(set(families), {"unequal", "parallel"})
        self.assertGreater(families.count("unequal"), 95)
        self.assertLess(families.count("unequal"), 145)
        self.assertEqual(cfg.family_weights, {"unequal": 3, "parallel": 1, "random": 0})

    def test_bad_mixture_weights_are_rejected(self):
        for weights in ({}, {"mixed": 1}, {"typo": 1}, {"random": -1},
                        {"random": 0}, {"random": float("nan")},
                        {"random": float("inf")}, {"random": "bad"}):
            with self.subTest(weights=weights), self.assertRaises(ValueError):
                ScenarioSampler(EnvConfig(family_weights=weights))

    def test_narrow_curriculum_is_randomized_but_goal_facing_and_at_rest(self):
        cfg = EnvConfig(family="converging", heading_jitter=0, initial_speed_max=0)
        sampler = ScenarioSampler(cfg, seed=913)
        scenes = [sampler.sample() for _ in range(80)]
        ratios = []
        for scene in scenes:
            goal_delta = scene.goals - scene.starts[:, :2]
            expected_yaw = np.arctan2(goal_delta[:, 1], goal_delta[:, 0])
            np.testing.assert_allclose(np.sin(scene.starts[:, 2]), np.sin(expected_yaw), atol=1e-12)
            np.testing.assert_allclose(np.cos(scene.starts[:, 2]), np.cos(expected_yaw), atol=1e-12)
            self.assertTrue(np.all(scene.starts[:, 3:] == 0))
            lengths = np.asarray(scene.metadata["trip_lengths"])
            ratios.append(lengths.min() / lengths.max())
            self.assertFalse(scene.metadata["deadline_is_feasibility_certificate"])
            self.assertTrue(scene.metadata["arena_is_sampling_region_only"])
        self.assertGreater(np.mean(ratios), 0.78)
        self.assertGreater(np.std([s.metadata["hub"][0] for s in scenes]), 0.5)
        self.assertGreater(len({tuple(s.metadata["car_permutation"]) for s in scenes}), 4)

    def test_random_missions_have_offcenter_goals_and_unequal_lengths(self):
        sampler = ScenarioSampler(EnvConfig(family="random"), seed=150)
        scenes = [sampler.sample() for _ in range(100)]
        goal_centers = np.array([scene.goals.mean(axis=0) for scene in scenes])
        ratios = [min(s.metadata["trip_lengths"]) / max(s.metadata["trip_lengths"])
                  for s in scenes]
        self.assertGreater(np.sum(np.linalg.norm(goal_centers, axis=1) > 2), 15)
        self.assertGreater(np.sum(np.asarray(ratios) < 0.5), 25)

    def test_impossible_packing_errors_instead_of_relabeling_or_expanding(self):
        cfg = EnvConfig(n_cars=3, family="converging", arena_half_min=0.6,
                        arena_half_max=0.6, min_trip=0.2)
        with self.assertRaisesRegex(ValueError, "Could not sample 'converging'"):
            ScenarioSampler(cfg, seed=12).sample()

    def test_unknown_family_rejected(self):
        with self.assertRaises(ValueError):
            ScenarioSampler(EnvConfig(family="typo"))


if __name__ == "__main__":
    unittest.main()
