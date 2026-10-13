"""Regression tests for credit assignment, CTDE structure and PPO numerics."""
from __future__ import annotations

import copy
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from arrival_final.config import EGO_DIM, NEIGHBOR_DIM, CRITIC_DIM
from arrival_final.mappo import (
    CentralizedCritic, LocalActor, MAPPOAgent, MODEL_SPEC, compute_gae,
)


def sample_observation(n=3, k=2, seed=3):
    rng = np.random.default_rng(seed)
    return dict(ego=rng.normal(size=(n, EGO_DIM)).astype(np.float32),
                neighbors=rng.normal(size=(n, k, NEIGHBOR_DIM)).astype(np.float32),
                mask=np.ones((n, k), dtype=bool))


def add_transition(agent, observation=None, active=None, seed=3):
    observation = observation or sample_observation(seed=seed)
    n = len(observation["ego"])
    state = np.random.default_rng(seed).normal(size=(n, CRITIC_DIM)).astype(np.float32)
    choice = agent.choose(observation, state)
    rewards = choice["values"] + np.linspace(1.0, 3.0, n, dtype=np.float32)
    if active is None:
        active = np.ones(n, dtype=bool)
    arguments = dict(observation=observation, state=state, latent=choice["latent"],
                     log_probs=choice["log_probs"], values=choice["values"], rewards=rewards,
                     next_values=np.zeros(n, dtype=np.float32), terminated=True,
                     truncated=False, active=active)
    agent.remember(**arguments)
    return arguments


class LearningTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        torch.manual_seed(12)

    def test_critic_is_equivariant_to_car_order_for_variable_team_sizes(self):
        critic = CentralizedCritic(hidden=16)
        for n in (1, 3, 9):
            states = torch.randn(4, n, CRITIC_DIM)
            permutation = torch.randperm(n)
            torch.testing.assert_close(critic(states[:, permutation]),
                                       critic(states)[:, permutation], atol=1e-6, rtol=1e-6)

    def test_critic_uses_new_absolute_timing_features(self):
        self.assertEqual(CRITIC_DIM, 16)
        critic = CentralizedCritic(hidden=16)
        # Remaining seconds/deadline (8/9) distinguish states that shared the old
        # remaining-time fraction. Later fields retain pacing/arrival history.
        state = torch.zeros(1, 3, CRITIC_DIM)
        initial = critic(state)
        for feature in range(8, CRITIC_DIM):
            changed = state.clone()
            changed[0, 0, feature] = 1.0
            self.assertGreater(float((critic(changed) - initial).abs().max().detach()), 1e-7)

    def test_actor_masking_empty_neighbors_and_neighbor_permutation(self):
        actor = LocalActor(hidden=16)
        ego = torch.randn(4, EGO_DIM)
        neighbors = torch.randn(4, 5, NEIGHBOR_DIM)
        mask = torch.tensor([[1, 0, 1, 0, 0], [0, 0, 0, 0, 0],
                             [1, 1, 1, 1, 1], [0, 1, 0, 1, 0]], dtype=torch.bool)
        original = actor(ego, neighbors, mask).mean
        permutation = torch.tensor([2, 4, 0, 1, 3])
        shuffled = actor(ego, neighbors[:, permutation], mask[:, permutation]).mean
        torch.testing.assert_close(shuffled, original, atol=1e-7, rtol=1e-6)
        invalid_padding = neighbors.clone()
        invalid_padding[~mask] = float("nan")
        torch.testing.assert_close(actor(ego, invalid_padding, mask).mean, original)
        no_neighbors = actor(ego, neighbors[:, :0], mask[:, :0]).mean
        all_masked = actor(ego, torch.full_like(neighbors, float("nan")),
                           torch.zeros_like(mask)).mean
        self.assertTrue(torch.isfinite(no_neighbors).all())
        torch.testing.assert_close(all_masked, no_neighbors)
        all_masked.square().sum().backward()
        self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all()
                            for p in actor.parameters()))

    def test_squashed_probability_and_extreme_latents_are_finite(self):
        distribution = torch.distributions.Normal(torch.zeros(2, 3), torch.ones(2, 3))
        latent = torch.tensor([[-1.0, 0.5, 2.0], [-0.2, 1.3, 0.0]])
        expected = (distribution.log_prob(latent)
                    - torch.log1p(-latent.tanh().square())).sum(-1)
        torch.testing.assert_close(LocalActor.log_prob(distribution, latent), expected)
        extreme = torch.tensor([[-50.0, 50.0, 30.0], [100.0, -100.0, 0.0]])
        self.assertTrue(torch.isfinite(LocalActor.log_prob(distribution, extreme)).all())

    def test_gae_terminal_does_not_bootstrap_or_leak_reset_rewards(self):
        advantage, returns = compute_gae(
            [[1.0], [2.0], [100.0]], [[0.5], [0.6], [0.0]],
            [[0.6], [99.0], [0.0]], [False, True, True], [False] * 3,
            gamma=0.9, lam=0.8)
        # terminal delta = 2 - .6; first delta = 1 + .9*.6 - .5
        np.testing.assert_allclose(advantage[:, 0], [1.04 + 0.72 * 1.4, 1.4, 100.0])
        np.testing.assert_allclose(returns, advantage + np.array([[0.5], [0.6], [0.0]]))

    def test_gae_collector_cut_bootstraps_but_cuts_trace(self):
        advantage, _ = compute_gae(
            [[1.0], [100.0]], [[0.5], [0.0]], [[2.0], [0.0]],
            [False, True], [True, False], gamma=0.9, lam=0.95)
        np.testing.assert_allclose(advantage[:, 0], [2.3, 100.0])
        final_cut, _ = compute_gae([[1.0]], [[0.5]], [[2.0]], [False], [False], gamma=0.9)
        np.testing.assert_allclose(final_cut, [[2.3]])
        with self.assertRaises(FloatingPointError):
            compute_gae([[float("nan")]], [[0.0]], [[0.0]], [True], [False])

    def test_empty_and_fully_inactive_rollouts_do_not_update_weights(self):
        agent = MAPPOAgent(hidden=16)
        self.assertEqual(agent.learn(), {})
        before = copy.deepcopy(agent.actor.state_dict())
        critic_before = copy.deepcopy(agent.critic.state_dict())
        add_transition(agent, active=np.zeros(3, dtype=bool))
        result = agent.learn()
        self.assertEqual(result["optimizer_updates"], 0)
        self.assertEqual(result["active_samples"], 0)
        self.assertTrue(result["skipped_no_active"])
        self.assertEqual(agent.buffer, [])
        for key, value in agent.actor.state_dict().items():
            torch.testing.assert_close(value, before[key], atol=0, rtol=0)
        for key, value in agent.critic.state_dict().items():
            torch.testing.assert_close(value, critic_before[key], atol=0, rtol=0)

    def test_action_causing_arrival_keeps_pre_step_credit_and_mask_is_copied(self):
        agent = MAPPOAgent(hidden=16, epochs=1, ent=0.0, gamma=0.0, batch=3)
        active_before_step = np.array([True, False, False])
        before = copy.deepcopy(agent.actor.state_dict())
        add_transition(agent, active=active_before_step)
        active_before_step[0] = False  # environment now reports all parked
        self.assertTrue(agent.buffer[0]["active"][0])
        result = agent.learn()
        self.assertEqual(result["active_samples"], 1)
        self.assertEqual(result["optimizer_updates"], 1)
        self.assertTrue(any(not torch.equal(value, before[key])
                            for key, value in agent.actor.state_dict().items()))

    def test_inactive_minibatches_are_safe(self):
        agent = MAPPOAgent(hidden=16, epochs=1, ent=0.0, batch=1, target_kl=None)
        add_transition(agent, active=np.array([True, False, False]))
        result = agent.learn()
        self.assertEqual(result["optimizer_updates"], 3)
        self.assertTrue(all(np.isfinite(value) for value in result.values()))

    def test_critic_gradient_magnitude_cannot_rescale_actor_update(self):
        first = MAPPOAgent(hidden=16, epochs=1, batch=3, ent=0.0, target_kl=None)
        second = MAPPOAgent(hidden=16, epochs=1, batch=3, ent=0.0,
                            target_kl=None, vcoef=1e12)
        second.actor.load_state_dict(first.actor.state_dict())
        second.critic.load_state_dict(first.critic.state_dict())
        transition = add_transition(first)
        second.remember(**transition)
        torch.manual_seed(199)
        first_result = first.learn()
        torch.manual_seed(199)
        second_result = second.learn()
        self.assertAlmostEqual(first_result["actor_grad_norm"], second_result["actor_grad_norm"])
        self.assertGreater(second_result["critic_grad_norm"], 1e6 * first_result["critic_grad_norm"])
        for key, value in first.actor.state_dict().items():
            torch.testing.assert_close(value, second.actor.state_dict()[key], atol=0, rtol=0)

    def test_checkpoint_roundtrip_metadata_variable_counts_and_optimizer_resume(self):
        agent = MAPPOAgent(hidden=16, epochs=1, batch=3, target_kl=None)
        add_transition(agent)
        agent.learn()
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "checkpoint.pt"
            agent.save(path, metadata={"source_hashes": {"env.py": "abc"}, "step": 7})
            restored, checkpoint = MAPPOAgent.from_checkpoint(path, load_optimizer=True)
            self.assertEqual(checkpoint["schema_version"], 3)
            self.assertEqual(checkpoint["model_spec"], dict(MODEL_SPEC, pooling="attention"))
            self.assertEqual(checkpoint["source_hashes"], {"env.py": "abc"})
            self.assertEqual(checkpoint["saved_device"], "cpu")
            self.assertEqual(len(agent.optimizer.state), len(restored.optimizer.state))
            self.assertIn("torch_rng_state", checkpoint)
            for n, k in ((1, 0), (3, 4), (8, 6)):
                observation = sample_observation(n, k)
                np.testing.assert_array_equal(agent.predict(observation), restored.predict(observation))
                state = np.zeros((n, CRITIC_DIM), dtype=np.float32)
                np.testing.assert_array_equal(agent.estimate_values(state), restored.estimate_values(state))
            transition = add_transition(agent, seed=19)
            restored.remember(**transition)
            torch.manual_seed(101)
            agent.learn()
            torch.manual_seed(101)
            restored.learn()
            for first, second in zip(agent.parameters, restored.parameters):
                torch.testing.assert_close(first, second, atol=0, rtol=0)
            checkpoint["schema_version"] = 1
            torch.save(checkpoint, path)
            with self.assertRaisesRegex(ValueError, "schema_version=3"):
                MAPPOAgent.from_checkpoint(path)

    def test_rollout_may_change_car_count_only_after_update(self):
        agent = MAPPOAgent(hidden=16, epochs=1)
        add_transition(agent)
        with self.assertRaisesRegex(ValueError, "between rollouts"):
            add_transition(agent, sample_observation(4, 2))
        agent.learn()
        add_transition(agent, sample_observation(4, 0))
        self.assertEqual(agent.learn()["active_samples"], 4)


if __name__ == "__main__":
    unittest.main()
