"""Regression tests for the failure modes observed in the earlier experiment."""
from contextlib import redirect_stdout
from dataclasses import replace
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from ..config import EnvConfig
from ..curriculum import next_boundary, stage_index, training_config
from .. import train


def metrics(**changes):
    value = dict(success_rate=0.0, collision_rate=0.0, all_arrived_rate=0.0,
                 arrival_fraction_mean=0.0, arrival_spread_mean_all_arrived=None,
                 max_deadline_error_mean_all_arrived=None)
    value.update(changes)
    return value


class RecordingAgent:
    def choose(self, observation, state):
        return dict(actions=np.zeros((2, 3)), latent=np.zeros((2, 3)),
                    log_probs=np.zeros(2), values=np.zeros(2))

    def estimate_values(self, state):
        return np.array([2.0, 3.0])

    def remember(self, *arguments):
        self.arguments = arguments


class ArrivingEnv:
    def __init__(self, terminal):
        self.arrived = np.array([False, True])
        self.terminal = terminal

    def step(self, actions):
        # Mutating in place deliberately catches a view/reference mask bug.
        self.arrived[:] = True
        return {}, np.ones(2), self.terminal, {}

    def critic_state(self):
        return np.ones((2, 16))


class TrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_timeout_does_not_beat_completion_with_equal_collisions(self):
        timeout = metrics()
        completes_late = metrics(all_arrived_rate=0.6, arrival_fraction_mean=0.8,
                                 arrival_spread_mean_all_arrived=4.2,
                                 max_deadline_error_mean_all_arrived=5.0)
        self.assertGreater(train.selection_score(completes_late), train.selection_score(timeout))
        partial = metrics(arrival_fraction_mean=0.3)
        self.assertGreater(train.selection_score(partial), train.selection_score(timeout))
        finite = metrics(arrival_spread_mean_all_arrived=0.5,
                         max_deadline_error_mean_all_arrived=0.5)
        self.assertGreater(train.selection_score(finite), train.selection_score(timeout))

    def test_success_then_collision_have_lexicographic_priority(self):
        success = metrics(success_rate=0.1, collision_rate=0.5)
        self.assertGreater(train.selection_score(success), train.selection_score(metrics()))
        completed_but_collision = metrics(collision_rate=0.1, all_arrived_rate=0.8)
        self.assertGreater(train.selection_score(metrics()), train.selection_score(completed_but_collision))

    def test_arrival_transition_is_active_and_terminal_has_no_bootstrap(self):
        agent, env = RecordingAgent(), ArrivingEnv(terminal=True)
        train.collect_transition(agent, env, {}, np.zeros((2, 16)), truncate=True)
        args = agent.arguments
        np.testing.assert_array_equal(args[-1], [True, False])
        np.testing.assert_array_equal(args[6], [0.0, 0.0])
        self.assertTrue(args[7])
        self.assertFalse(args[8])

    def test_curriculum_cut_bootstraps_and_stops_gae_trace(self):
        agent, env = RecordingAgent(), ArrivingEnv(terminal=False)
        train.collect_transition(agent, env, {}, np.zeros((2, 16)), truncate=True)
        np.testing.assert_array_equal(agent.arguments[6], [2.0, 3.0])
        self.assertFalse(agent.arguments[7])
        self.assertTrue(agent.arguments[8])

    def test_curriculum_exact_boundaries_and_no_smoke_rescaling(self):
        for step, expected in ((0, 0), (349999, 0), (350000, 1), (599999, 1),
                               (600000, 2), (799999, 2), (800000, 3), (1000001, 3)):
            self.assertEqual(stage_index(step, 1_000_000), expected)
        self.assertEqual(next_boundary(350000, 1_000_000), 600000)
        self.assertIsNone(next_boundary(800000, 1_000_000))
        args = train.parser().parse_args(["--run-dir", "unused", "--steps", "8"])
        target, settings, _ = train._configuration(args)
        self.assertEqual(settings["curriculum_steps"], 1_000_000)
        self.assertEqual(training_config(target, 8, settings["curriculum_steps"])[0], "converging")

    def test_curriculum_keeps_controller_reward_and_physics_comparable(self):
        pp, mpc = EnvConfig(controller="pp"), EnvConfig(controller="mpc")
        for step in (0, 350000, 600000, 800000):
            _, a = training_config(pp, step, 1_000_000)
            _, b = training_config(mpc, step, 1_000_000)
            self.assertEqual(replace(a, controller="mpc"), b)
            self.assertEqual(a.schedule_weight, pp.schedule_weight)
            self.assertEqual(a.physics_dt, pp.physics_dt)
        name, config = training_config(pp, 0, 1_000_000, "none")
        self.assertEqual(config, pp)
        self.assertEqual(name, "full")

    def test_validation_fixed_strata_seeds_and_target_weights(self):
        target = EnvConfig()
        plan = train.stratified_plan(target, 50, train.VALIDATION_SEED_BASE)
        self.assertEqual(len(plan), 50)
        self.assertEqual(len(set(plan)), 50)
        self.assertEqual({family for family, _ in plan}, set(train.DEFAULT_FAMILY_WEIGHTS))
        self.assertTrue(all(train.VALIDATION_SEED_BASE <= seed < train.FINAL_TEST_SEED_BASE
                            for _, seed in plan))
        self.assertEqual(plan, train.stratified_plan(target, 50, train.VALIDATION_SEED_BASE))
        for serial in range(100):
            self.assertLess(train._episode_seed(77, serial), train.VALIDATION_SEED_BASE)
        args = train.parser().parse_args(["--run-dir", "unused", "--eval-episodes", "4"])
        with self.assertRaisesRegex(ValueError, "every target family"):
            train._configuration(args)

    def test_step_budget_rollout_cuts_and_resume_settings(self):
        self.assertEqual(train.rollout_length(999990, 1000000, 1024, 1000000, 1000000), 10)
        self.assertEqual(train.rollout_length(349950, 1000000, 1024, 360000, 400000, 350000), 50)
        args = train.parser().parse_args(["--run-dir", "unused", "--steps", "999"])
        config, settings, algorithm = train._configuration(args)
        saved = dict(env_config=config.to_dict(), train_config=settings, algorithm=algorithm)
        args.curriculum_steps = 999
        with self.assertRaisesRegex(ValueError, "curriculum-steps"):
            train._configuration(args, saved)
        args.curriculum_steps = None
        args.controller = "mpc"
        with self.assertRaisesRegex(ValueError, "environment configuration"):
            train._configuration(args, saved)

    def test_trainer_exact_budget_resume_and_validation_only_best(self):
        validation = metrics(arrival_fraction_mean=0.1)
        validation["all_target_families_covered"] = True
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            run = Path(directory) / "run"
            with patch.object(train, "evaluate", return_value=(validation, [])):
                self.assertEqual(train.main(["--run-dir", str(run), "--steps", "3",
                    "--rollout", "2", "--epochs", "1", "--hidden", "16", "--batch", "6",
                    "--eval-episodes", "5", "--checkpoint-every", "2"]), 0)
                saved = torch.load(run / "latest.pt", weights_only=True)
                self.assertEqual(saved["global_step"], 3)
                self.assertEqual(saved["train_config"]["curriculum_steps"], 1_000_000)
                self.assertTrue(saved["checkpoint_validated"])
                self.assertTrue((run / "best.pt").exists())
                self.assertTrue((run / "checkpoints" / "step_000000002.pt").exists())
                self.assertEqual(train.main(["--run-dir", str(run), "--resume", str(run / "latest.pt"),
                                              "--steps", "5"]), 0)
                self.assertEqual(torch.load(run / "latest.pt", weights_only=True)["global_step"], 5)
            log = [json.loads(line) for line in (run / "training.jsonl").read_text().splitlines()]
            self.assertEqual([row["global_step"] for row in log if row["event"] == "update"], [2, 3, 4, 5])
            interrupted_run = Path(directory) / "interrupted"
            with patch.object(train, "collect_transition", side_effect=KeyboardInterrupt):
                self.assertEqual(train.main(["--run-dir", str(interrupted_run), "--steps", "2",
                                              "--hidden", "16", "--eval-episodes", "5"]), 130)
            self.assertFalse((interrupted_run / "best.pt").exists())
            saved = torch.load(interrupted_run / "latest.pt", weights_only=True)
            self.assertFalse(saved["checkpoint_validated"])
            self.assertTrue(saved["interrupted"])


if __name__ == "__main__":
    unittest.main()
