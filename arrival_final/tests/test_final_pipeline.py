"""Protocol and transfer regressions added for the final experiment package."""
from argparse import Namespace
from contextlib import redirect_stdout
from dataclasses import replace
import io
import tempfile
from pathlib import Path
import unittest

import numpy as np
import torch

from ..artifacts import digest, read_json, write_json
from ..baselines import LocalLatticePolicy
from ..benchmark import run_benchmark, timing_summary
from ..config import EnvConfig
from ..curriculum import training_config
from ..env import ArrivalEnv
from ..mappo import MAPPOAgent, LocalActor
from ..missions import build_manifest, load_manifest, scenario_from_record
from ..plants import DynamicPlant, MujocoPlant
from ..report import bootstrap, success_matrix, summarize
from ..study import prepare, load_study, training_jobs, evaluation_jobs
from .test_learning import sample_observation, add_transition


class FinalPipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_mean_pooling_mask_permutation_and_empty_set(self):
        obs = sample_observation(3, 4)
        ego, neighbors = torch.tensor(obs['ego']), torch.tensor(obs['neighbors'])
        mask = torch.tensor([[1, 0, 1, 0], [0, 0, 0, 0], [1, 1, 1, 1]], dtype=torch.bool)
        actor = LocalActor(hidden=16, pooling='mean')
        original = actor(ego, neighbors, mask).mean
        permutation = [2, 0, 3, 1]
        torch.testing.assert_close(actor(ego, neighbors[:, permutation], mask[:, permutation]).mean, original)
        neighbors[~mask] = float('nan')
        torch.testing.assert_close(actor(ego, neighbors, mask).mean, original)
        empty = actor(ego, neighbors[:, :0], mask[:, :0]).mean
        torch.testing.assert_close(empty[1], original[1])
        self.assertTrue(torch.isfinite(empty).all())

    def test_mean_actor_updates_and_checkpoint_preserves_architecture(self):
        agent = MAPPOAgent(pooling='mean', hidden=16, epochs=1, batch=3)
        before = [p.clone().detach() for p in agent.actor.parameters()]
        add_transition(agent)
        agent.learn()
        self.assertTrue(any(not torch.equal(a, b) for a, b in zip(before, agent.actor.parameters())))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'agent.pt'
            agent.save(path)
            restored, metadata = MAPPOAgent.from_checkpoint(path)
            self.assertEqual(metadata['model_spec']['pooling'], 'mean')
            obs = sample_observation(8, 4)
            np.testing.assert_array_equal(agent.predict(obs), restored.predict(obs))
            metadata['model_spec']['pooling'] = 'attention'
            torch.save(metadata, path)
            with self.assertRaises(ValueError):
                MAPPOAgent.from_checkpoint(path)

    def test_omitted_families_do_not_reappear_in_curriculum(self):
        target = EnvConfig(family_weights={'random': 1.0})
        for step in (0, 350000, 600000, 800000):
            _, cfg = training_config(target, step, 1_000_000)
            if cfg.family == 'mixed':
                self.assertEqual({k for k, v in cfg.family_weights.items() if v > 0}, {'random'})
            else:
                self.assertEqual(cfg.family, 'random')

    def test_manifest_identity_and_exact_reset(self):
        cfg = EnvConfig(n_cars=3, family='parallel')
        manifest = build_manifest(cfg, per_family=2)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'missions.json'
            write_json(path, manifest)
            restored = load_manifest(path)
            scenario = scenario_from_record(restored['missions'][0])
            env = ArrivalEnv(cfg)
            env.reset(scenario=scenario)
            np.testing.assert_array_equal(env.states, scenario.starts)
            env.states[0, 0] += 1
            self.assertNotEqual(env.states[0, 0], scenario.starts[0, 0])
            restored['missions'][0]['deadline'] += 1
            write_json(path, restored)
            with self.assertRaisesRegex(ValueError, 'modified'):
                load_manifest(path)

    def test_split_ranges_and_local_baseline(self):
        with self.assertRaises(ValueError):
            build_manifest(EnvConfig(), seed=20, split='test')
        cfg = EnvConfig()
        env = ArrivalEnv(cfg, seed=17)
        obs = env.reset()
        policy = LocalLatticePolicy(cfg)
        actions = policy.predict(obs)
        self.assertEqual(actions.shape, (3, 3))
        self.assertTrue(np.isfinite(actions).all())
        self.assertTrue(np.all(np.abs(actions) <= 1))
        permutation = [2, 0, 3, 1]
        reordered = dict(ego=obs['ego'], neighbors=obs['neighbors'][:, permutation], mask=obs['mask'][:, permutation])
        np.testing.assert_allclose(policy.predict(reordered), actions)

    def test_plan_has_nine_independent_runs_and_reuses_pp_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            root = Path(directory)
            prepare(Namespace(root=root, steps=1_000_000, curriculum_steps=1_000_000,
                              per_family=1, validation_per_family=1, seeds=[0, 1, 2], cars=[3]))
            protocol = load_study(root)
            jobs = training_jobs(root, protocol, list(protocol['variants']), [0, 1, 2], 'cpu', False)
            self.assertEqual(len(jobs), 9)
            self.assertEqual(len({j[0] for j in jobs}), 9)
            checkpoint = root / 'runs' / 'attention_pp_s0' / 'best.pt'
            checkpoint.parent.mkdir(parents=True)
            checkpoint.touch()  # Only the job planner needs a path; no policy is loaded here.
            evals = evaluation_jobs(root, protocol, ['attention_pp', 'attention_pp_mpc'], [0],
                                   ['target_3'], ['kinematic'], False)
            self.assertEqual(evals[0][1][-1], str(checkpoint))
            self.assertEqual(evals[1][1][-1], str(checkpoint))

    def test_benchmark_resume_and_report_with_saved_mission(self):
        cfg = EnvConfig(n_cars=1, family='parallel', timeout_margin=0.2)
        manifest = build_manifest(cfg, per_family=1)
        # A short deliberately artificial mission makes this an integration test,
        # not a trained-policy quality estimate.
        row = manifest['missions'][0]
        row.update(starts=[[0., 0., 0., 0., 0.]], goals=[[0.2, 0.]], deadline=0.5)
        row['mission_id'] = digest({k: v for k, v in row.items() if k != 'mission_id'})
        manifest['manifest_id'] = digest({k: v for k, v in manifest.items() if k != 'manifest_id'})
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            mission, output = root / 'mission.json', root / 'result.json'
            write_json(mission, manifest)
            result = run_benchmark(mission, 'goal_pp', output, progress=False)
            self.assertEqual(result['metrics']['episodes'], 1)
            with np.load(root / result['episodes'][0]['trajectory_file']) as trace:
                self.assertEqual(len(trace['states']), len(trace['time']))
                np.testing.assert_array_equal(trace['states'][0], row['starts'])
            repeated = run_benchmark(mission, 'goal_pp', output, resume=True, progress=False)
            self.assertEqual(result, repeated)
            with self.assertRaises(FileExistsError):
                run_benchmark(mission, 'goal_pp', output, progress=False)
            with self.assertRaises(FileExistsError):
                run_benchmark(mission, 'goal_mpc', output, resume=True, progress=False)
            from unittest.mock import patch
            with patch('arrival_final.report.make_plots'):
                table = summarize([output], root / 'report', replicates=20)
            self.assertIsNone(table[0]['success_seed_std'])
            self.assertTrue((root / 'report' / 'RESULTS.md').exists())
            tampered = dict(result, episodes=[dict(result['episodes'][0], mission_id='other')])
            with self.assertRaisesRegex(ValueError, 'paired'):
                success_matrix([result, tampered])

    def test_paired_bootstrap_and_overrun_count(self):
        reference = dict(family_weights={'a': 0.75, 'b': 0.25},
                         episodes=[dict(family='a'), dict(family='b')])
        self.assertEqual(bootstrap(np.ones((3, 2)), reference, replicates=30), [1., 1.])
        self.assertEqual(bootstrap(np.zeros((3, 2)), reference, replicates=30), [0., 0.])
        self.assertEqual(timing_summary([0.01, 0.10], 0.05)['overrun_rate'], 0.5)
        self.assertIsNone(timing_summary([], 0.05)['p95'])

    def test_lateral_motion_is_not_misclassified_as_arrival(self):
        cfg = EnvConfig(n_cars=1, family='parallel')
        plant = DynamicPlant(cfg)
        env = ArrivalEnv(cfg, plant=plant)
        env.reset(seed=1)
        plant.raw[0, 3:6] = [0., 0.3, 0.]
        env.states = plant.states
        env.goals[:] = env.states[:, :2]
        env._events(env.states.copy())
        env.time += cfg.dwell_time + 0.01
        env._events(env.states.copy())
        self.assertFalse(env.arrived[0])
        self.assertGreater(np.linalg.norm(plant.world_velocities[0]), cfg.stop_speed)


try:
    import mujoco
except ImportError:
    mujoco = None


@unittest.skipIf(mujoco is None, 'optional MuJoCo dependency not installed')
class MujocoTransferTests(unittest.TestCase):
    def test_reset_velocity_steering_and_contact_ownership(self):
        cfg = EnvConfig(n_cars=2)
        plant = MujocoPlant(cfg)
        try:
            states = np.array([[0., 0., 0.5, 0.4, 0.1], [3., 3., -1., 0.2, -0.2]])
            plant.reset(states)
            np.testing.assert_allclose(plant.states, states, atol=1e-12)
            expected = states[:, 3, None] * np.column_stack((np.cos(states[:, 2]), np.sin(states[:, 2])))
            np.testing.assert_allclose(plant.world_velocities, expected, atol=1e-10)
            self.assertEqual(plant.contact_pairs, set())
            plant.step(np.zeros((2, 2)), 0.01)
            self.assertAlmostEqual(plant.data.time, 0.01)
            self.assertTrue(np.isfinite(plant.states).all())
            with self.assertRaisesRegex(ValueError, 'integer multiple'):
                plant.step(np.zeros((2, 2)), 0.0105)
            states[:] = 0
            states[1, 1] = 0.1
            plant.reset(states)
            self.assertIn((0, 1), plant.contact_pairs)
        finally:
            directory = Path(plant._directory.name)
            plant.close()
            self.assertFalse(directory.exists())

    def test_each_plant_has_its_own_scene(self):
        first = MujocoPlant(EnvConfig(n_cars=1))
        second = MujocoPlant(EnvConfig(n_cars=1))
        try:
            self.assertNotEqual(first._directory.name, second._directory.name)
        finally:
            first.close()
            second.close()
