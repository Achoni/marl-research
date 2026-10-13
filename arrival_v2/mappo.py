"""Shared local attention actor and a permutation-equivariant centralized critic.

The critic is used only while collecting/updating training rollouts. Execution
calls ``predict(local_observation)`` and never supplies the joint state.
"""
from __future__ import annotations

import math
import os
from pathlib import Path
import tempfile

import numpy as np
import torch
from torch import nn
from torch.distributions import Normal
import torch.nn.functional as F

from .config import EGO_DIM, NEIGHBOR_DIM, CRITIC_DIM

SCHEMA_VERSION = 2
MODEL_SPEC = dict(architecture="local_attention_mean_pool_critic_v2",
                  ego_dim=EGO_DIM, neighbor_dim=NEIGHBOR_DIM,
                  critic_dim=CRITIC_DIM, action_dim=3)


def compute_gae(rewards, values, next_values, terminated, truncated,
                gamma=0.99, lam=0.95):
    """Bootstrap collector cuts, never bootstrap actual task endings.

    Arrays have shape ``[time, cars]``. A task timeout is a termination: its
    deadline and timeout margin are part of the task, not a collector limit.
    Truncation bootstraps the final next state but cuts the lambda trace so an
    unrelated reset episode cannot leak into it. The final stored transition
    also bootstraps even when its collector-cut flag is omitted.
    """
    rewards = np.asarray(rewards, dtype=np.float32)
    values = np.asarray(values, dtype=np.float32)
    next_values = np.asarray(next_values, dtype=np.float32)
    terminated = np.asarray(terminated, dtype=bool)
    truncated = np.asarray(truncated, dtype=bool)

    if rewards.ndim != 2 or rewards.shape != values.shape or rewards.shape != next_values.shape:
        raise ValueError("GAE rewards, values and next_values must share [time, cars] shape")
    if terminated.shape != (len(rewards),) or truncated.shape != (len(rewards),):
        raise ValueError("GAE terminal flags must have shape [time]")
    if not 0 <= gamma <= 1 or not 0 <= lam <= 1:
        raise ValueError("GAE gamma and lambda must lie in [0, 1]")
    if not all(np.isfinite(array).all() for array in (rewards, values, next_values)):
        raise FloatingPointError("Non-finite reward/value supplied to GAE")
    
    advantage = np.zeros_like(rewards)
    trace = np.zeros(rewards.shape[1:], dtype=np.float32)
    for t in reversed(range(len(rewards))):
        bootstrap = 0.0 if terminated[t] else 1.0
        continuation = 0.0 if terminated[t] or truncated[t] else 1.0
        delta = rewards[t] + gamma * bootstrap * next_values[t] - values[t]
        trace = delta + gamma * lam * continuation * trace
        advantage[t] = trace
    return advantage, advantage + values


def _mlp(inputs, hidden):
    return nn.Sequential(nn.Linear(inputs, hidden), nn.Tanh(),
                         nn.Linear(hidden, hidden), nn.Tanh())


def _initialize(module):
    if isinstance(module, nn.Linear):
        nn.init.orthogonal_(module.weight, math.sqrt(2.0))
        if module.bias is not None:
            nn.init.zeros_(module.bias)


class LocalActor(nn.Module):
    """Ego-conditioned attention over an unordered, masked neighbor set.

    Input leading dimensions may include a batch and/or cars; K may be zero.
    Parameters do not depend on the number of cars or neighbor slots.
    """
    def __init__(self, hidden=128):
        super().__init__()
        self.hidden = hidden
        self.ego_encoder = _mlp(EGO_DIM, hidden)
        self.neighbor_encoder = _mlp(NEIGHBOR_DIM, hidden)
        self.query = nn.Linear(hidden, hidden, bias=False)
        self.key = nn.Linear(hidden, hidden, bias=False)
        self.head = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.Tanh(),
                                  nn.Linear(hidden, 3))
        self.log_std = nn.Parameter(torch.full((3,), -0.5))
        self.apply(_initialize)
        nn.init.orthogonal_(self.head[-1].weight, 0.01)

    def forward(self, ego, neighbors, mask):
        ego_embedding = self.ego_encoder(ego)
        mask = mask.bool()
        if neighbors.shape[-2] == 0:
            context = torch.zeros_like(ego_embedding)
        else:
            # Mask before encoding: padding values cannot enter the network,
            # including NaNs in invalid slots.
            neighbors = torch.where(mask.unsqueeze(-1), neighbors,
                                    torch.zeros_like(neighbors))
            embeddings = self.neighbor_encoder(neighbors)
            logits = (self.key(embeddings)
                      * self.query(ego_embedding).unsqueeze(-2)).sum(-1)
            logits = logits / math.sqrt(self.hidden)
            weights = torch.softmax(logits.masked_fill(~mask, -1e9), dim=-1)
            weights = weights * mask.to(logits.dtype)
            weights = weights / weights.sum(-1, keepdim=True).clamp_min(1e-8)
            context = (weights.unsqueeze(-1) * embeddings).sum(-2)
        mean = self.head(torch.cat((ego_embedding, context), dim=-1))
        return Normal(mean, self.log_std.clamp(-5.0, 1.0).exp())

    @staticmethod
    def log_prob(distribution, latent):
        """Probability of tanh(latent), with a stable squash Jacobian."""
        log_jacobian = 2.0 * (math.log(2.0) - latent
                              - F.softplus(-2.0 * latent))
        return (distribution.log_prob(latent) - log_jacobian).sum(-1)


class CentralizedCritic(nn.Module):
    """Encode all cars, mean-pool the team, then predict one value per car."""
    def __init__(self, hidden=128):
        super().__init__()
        self.encoder = _mlp(CRITIC_DIM, hidden)
        self.head = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.Tanh(),
                                  nn.Linear(hidden, 1))
        self.apply(_initialize)
        nn.init.orthogonal_(self.head[-1].weight, 1.0)

    def forward(self, states):
        encoded = self.encoder(states)
        team = encoded.mean(-2, keepdim=True).expand_as(encoded)
        return self.head(torch.cat((encoded, team), -1)).squeeze(-1)


class MAPPOAgent:
    def __init__(self, device="cpu", hidden=128, lr=3e-4, gamma=0.99,
                 lam=0.95, clip=0.2, epochs=4, batch=128, ent=0.005,
                 target_kl=0.03, max_grad=0.5, vcoef=0.5):
        self.device = torch.device(device)
        self.algorithm = dict(hidden=hidden, lr=lr, gamma=gamma, lam=lam,
                              clip=clip, epochs=epochs, batch=batch, ent=ent,
                              target_kl=target_kl, max_grad=max_grad, vcoef=vcoef)
        
        for name in ("hidden", "epochs", "batch"):
            if not isinstance(self.algorithm[name], int) or self.algorithm[name] < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("lr", "clip", "max_grad"):
            if not math.isfinite(self.algorithm[name]) or self.algorithm[name] <= 0:
                raise ValueError(f"{name} must be positive and finite")
        for name in ("ent", "vcoef"):
            if not math.isfinite(self.algorithm[name]) or self.algorithm[name] < 0:
                raise ValueError(f"{name} must be nonnegative and finite")
        if not 0 <= gamma <= 1 or not 0 <= lam <= 1:
            raise ValueError("gamma and lambda must lie in [0, 1]")
        if target_kl is not None and (not math.isfinite(target_kl) or target_kl <= 0):
            raise ValueError("target_kl must be positive and finite, or None")
        
        self.actor = LocalActor(hidden).to(self.device)
        self.critic = CentralizedCritic(hidden).to(self.device)
        self.actor_parameters = list(self.actor.parameters())
        self.critic_parameters = list(self.critic.parameters())
        self.parameters = self.actor_parameters + self.critic_parameters
        self.optimizer = torch.optim.Adam(self.parameters, lr=lr, eps=1e-5)
        self.buffer = []

    def _tensor(self, value, dtype=torch.float32):
        return torch.as_tensor(np.asarray(value), dtype=dtype, device=self.device)

    def _distribution(self, observation):
        return self.actor(self._tensor(observation["ego"]),
                          self._tensor(observation["neighbors"]),
                          self._tensor(observation["mask"], torch.bool))

    @torch.no_grad()
    def predict(self, observation, deterministic=True):
        """Local execution; does not call the critic or change parameters."""
        distribution = self._distribution(observation)
        latent = distribution.mean if deterministic else distribution.sample()
        return latent.tanh().cpu().numpy()
    
    @torch.no_grad()
    def estimate_values(self, state):
        return self.critic(self._tensor(state).unsqueeze(0))[0].cpu().numpy()

    @torch.no_grad()
    def choose(self, observation, state):
        distribution = self._distribution(observation)
        latent = distribution.sample()
        return dict(actions=latent.tanh().cpu().numpy(),
                    latent=latent.cpu().numpy(),
                    log_probs=self.actor.log_prob(distribution, latent).cpu().numpy(),
                    values=self.estimate_values(state))

    def remember(self, observation, state, latent, log_probs, values, rewards,
                 next_values, terminated, truncated, active):
        """Store one transition, with ``active`` captured BEFORE env.step().

        The action that causes arrival must remain active for its PPO update.
        This layer can validate and copy the mask, but cannot reconstruct a
        pre-step mask from a post-step observation; the collector owns it.
        """
        transition = dict(
            ego=np.asarray(observation["ego"], np.float32).copy(),
            neighbors=np.asarray(observation["neighbors"], np.float32).copy(),
            mask=np.asarray(observation["mask"], bool).copy(),
            state=np.asarray(state, np.float32).copy(),
            latent=np.asarray(latent, np.float32).copy(),
            log_probs=np.asarray(log_probs, np.float32).copy(),
            values=np.asarray(values, np.float32).copy(),
            rewards=np.asarray(rewards, np.float32).copy(),
            next_values=np.asarray(next_values, np.float32).copy(),
            terminated=bool(terminated), truncated=bool(truncated),
            active=np.asarray(active, bool).copy())
        n_cars = transition["state"].shape[0]
        k = transition["neighbors"].shape[-2]
        shapes = dict(state=(n_cars, CRITIC_DIM), ego=(n_cars, EGO_DIM),
                      neighbors=(n_cars, k, NEIGHBOR_DIM), mask=(n_cars, k),
                      latent=(n_cars, 3), log_probs=(n_cars,), values=(n_cars,),
                      rewards=(n_cars,), next_values=(n_cars,), active=(n_cars,))
        
        for key, shape in shapes.items():
            if transition[key].shape != shape:
                raise ValueError(f"{key} must have shape {shape}, got {transition[key].shape}")
            if key not in ("neighbors", "mask", "active") and not np.isfinite(transition[key]).all():
                raise FloatingPointError(f"Non-finite {key} in rollout transition")
        if n_cars < 1:
            raise ValueError("A rollout needs at least one car")
        if not np.isfinite(transition["neighbors"][transition["mask"]]).all():
            raise FloatingPointError("Non-finite visible neighbor in rollout transition")
        if self.buffer and any(transition[key].shape != self.buffer[0][key].shape
                               for key in ("state", "neighbors")):
            raise ValueError("Car/neighbor counts may change between rollouts, not inside one buffer")
        

        self.buffer.append(transition)

    def learn(self):
        if not self.buffer:
            return {}
        cfg, data = self.algorithm, self.buffer
        n_cars = data[0]["state"].shape[0]
        advantage, returns = compute_gae(
            np.stack([x["rewards"] for x in data]),
            np.stack([x["values"] for x in data]),
            np.stack([x["next_values"] for x in data]),
            [x["terminated"] for x in data], [x["truncated"] for x in data],
            cfg["gamma"], cfg["lam"])
        active = np.concatenate([x["active"] for x in data])
        if not active.any():
            # There is no decision to reinforce. Do not produce NaN advantage
            # statistics or move an otherwise idle optimizer on padded data.
            rollout_steps = len(data)
            self.buffer.clear()
            return dict(actor_loss=0.0, value_loss=0.0, entropy=0.0,
                        approx_kl=0.0, clip_fraction=0.0, actor_grad_norm=0.0,
                        critic_grad_norm=0.0, optimizer_updates=0, early_stop=False,
                        active_samples=0, rollout_steps=rollout_steps,
                        skipped_no_active=True)
        advantage = advantage.reshape(-1)
        valid = advantage[active]
        # Centering a single sample (or an exactly constant tiny rollout) would
        # erase all policy credit. Full rollouts normally have nonzero spread.
        if len(valid) > 1 and float(valid.std()) > 1e-8:
            advantage[active] = (valid - valid.mean()) / float(valid.std())
        advantage[~active] = 0.0
        tensors = {key: self._tensor(np.concatenate([x[key] for x in data]),
                                    torch.bool if key == "mask" else torch.float32)
                   for key in ("ego", "neighbors", "mask", "latent", "log_probs")}
        states = self._tensor(np.stack([x["state"] for x in data]))
        advantage = self._tensor(advantage)
        returns = self._tensor(returns.reshape(-1))
        active = self._tensor(active, torch.bool)
        count = len(advantage)
        step_ids = torch.arange(len(data), device=self.device).repeat_interleave(n_cars)
        car_ids = torch.arange(n_cars, device=self.device).repeat(len(data))
        totals = dict(actor_loss=0.0, value_loss=0.0, entropy=0.0,
                      approx_kl=0.0, clip_fraction=0.0,
                      actor_grad_norm=0.0, critic_grad_norm=0.0)
        updates, stopped = 0, False
        for _ in range(cfg["epochs"]):
            for indices in torch.randperm(count, device=self.device).split(cfg["batch"]):
                distribution = self.actor(tensors["ego"][indices],
                                          tensors["neighbors"][indices],
                                          tensors["mask"][indices])
                log_prob = self.actor.log_prob(distribution, tensors["latent"][indices])
                use = active[indices]
                actor_loss = entropy = kl = clip_fraction = log_prob.new_zeros(())
                if use.any():
                    log_ratio = log_prob[use] - tensors["log_probs"][indices][use]
                    ratio = log_ratio.exp()
                    kl = ((ratio - 1.0) - log_ratio).mean()
                    if not torch.isfinite(kl):
                        raise FloatingPointError("Non-finite policy KL during PPO update")
                    if cfg["target_kl"] is not None and kl.item() > cfg["target_kl"]:
                        stopped = True
                        break
                    adv = advantage[indices][use]
                    actor_loss = -torch.minimum(ratio * adv,
                        ratio.clamp(1.0 - cfg["clip"], 1.0 + cfg["clip"]) * adv).mean()
                    entropy = -self.actor.log_prob(distribution, distribution.rsample())[use].mean()
                    clip_fraction = ((ratio - 1.0).abs() > cfg["clip"]).float().mean()
                values = self.critic(states[step_ids[indices]])[
                    torch.arange(len(indices), device=self.device), car_ids[indices]]
                value_loss = 0.5 * F.mse_loss(values, returns[indices])
                loss = actor_loss + cfg["vcoef"] * value_loss - cfg["ent"] * entropy
                if not torch.isfinite(loss):
                    raise FloatingPointError("Non-finite PPO objective")
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                # A large critic error must not shrink an unrelated actor
                # gradient through a shared norm. Networks share an optimizer,
                # but have independent clipping budgets and Adam moments.
                actor_grad = nn.utils.clip_grad_norm_(
                    self.actor_parameters, cfg["max_grad"], error_if_nonfinite=True)
                critic_grad = nn.utils.clip_grad_norm_(
                    self.critic_parameters, cfg["max_grad"], error_if_nonfinite=True)
                self.optimizer.step()
                updates += 1
                for key, value in (("actor_loss", actor_loss), ("value_loss", value_loss),
                                   ("entropy", entropy), ("approx_kl", kl),
                                   ("clip_fraction", clip_fraction),
                                   ("actor_grad_norm", actor_grad),
                                   ("critic_grad_norm", critic_grad)):
                    totals[key] += float(value.detach().cpu())
            if stopped:
                break
        rollout_steps = len(data)
        self.buffer.clear()
        return {key: value / max(updates, 1) for key, value in totals.items()} | {
            "optimizer_updates": updates, "early_stop": stopped,
            "active_samples": int(active.sum().item()), "rollout_steps": rollout_steps,
            "skipped_no_active": False}

    def save(self, path, metadata=None):
        """Atomically replace a checkpoint; metadata contains plain JSON types."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint = dict(metadata or {})
        checkpoint.update(schema_version=SCHEMA_VERSION, model_spec=dict(MODEL_SPEC),
                          saved_device=str(self.device), algorithm=dict(self.algorithm),
                          actor=self.actor.state_dict(), critic=self.critic.state_dict(),
                          optimizer=self.optimizer.state_dict(),
                          torch_rng_state=torch.get_rng_state())
        if torch.cuda.is_available():
            checkpoint["cuda_rng_states"] = torch.cuda.get_rng_state_all()
        descriptor, temporary = tempfile.mkstemp(prefix="." + path.name, suffix=".tmp",
                                                 dir=path.parent)
        os.close(descriptor)
        try:
            torch.save(checkpoint, temporary)
            os.replace(temporary, path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)

    @classmethod
    def from_checkpoint(cls, path, device="cpu", load_optimizer=False):
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        if checkpoint.get("schema_version") != SCHEMA_VERSION:
            raise ValueError("arrival_v2 requires schema_version=2; checkpoints from earlier "
                             "pipelines have incompatible observations and cannot be resumed")
        if checkpoint.get("model_spec") != MODEL_SPEC:
            raise ValueError("Checkpoint model/observation schema does not match arrival_v2")
        agent = cls(device=device, **checkpoint["algorithm"])
        agent.actor.load_state_dict(checkpoint["actor"])
        agent.critic.load_state_dict(checkpoint["critic"])
        if load_optimizer:
            # Adam.load_state_dict maps its moment buffers to each parameter's
            # device, and retains the proper CPU/device placement of step
            # counters. Loading the file on CPU therefore supports CPU/CUDA/MPS
            # resume without forcing all state tensors onto one device.
            agent.optimizer.load_state_dict(checkpoint["optimizer"])
        return agent, checkpoint
