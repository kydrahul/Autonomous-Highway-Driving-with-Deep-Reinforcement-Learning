"""
C51 — Categorical/Distributional DQN (Rainbow's Missing Component)
====================================================================
This completes the full 6/6 Rainbow by adding the Distributional RL
component from Bellemare et al. (2017): "A Distributional Perspective
on Reinforcement Learning" (C51).

Key idea:
  Instead of learning E[Q(s,a)], learn the full return DISTRIBUTION Z(s,a).
  Z is represented as a categorical distribution over 51 fixed atoms:
    Z ~ {z_0, z_1, ..., z_50} with z_i = V_min + i*(V_max-V_min)/50
  The network outputs logits → softmax → probability p_i for each atom.
  Target: Project the Bellman-updated distribution onto the fixed support.

This is combined with Double DQN + Dueling + PER + NoisyNets + n-step
to form the complete Rainbow agent.

Usage:
    python scripts/training/train_c51_rainbow.py [--steps 500000]

Outputs:
    models/c51_rainbow/c51_rainbow_final.zip
"""

import os
import sys
import math
import argparse
import numpy as np
import gymnasium as gym
import highway_env  # noqa: F401
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
from typing import Dict, List, Optional, Tuple, Union

from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor, FlattenExtractor
from stable_baselines3.common.type_aliases import GymEnv, Schedule
from stable_baselines3.common.utils import get_linear_fn, is_vectorized_observation
from stable_baselines3 import DQN

# Import NoisyLinear, SumTree, PrioritizedReplayBuffer from main Rainbow script
sys.path.insert(0, "scripts/training")
from train_rainbow_dqn import (  # noqa: E402
    NoisyLinear, SumTree, PrioritizedReplayBuffer, LeftLaneRewardWrapper
)

MODEL_DIR = "models/c51_rainbow"
LOG_DIR   = "logs/c51_rainbow"
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOG_DIR,   exist_ok=True)

ENV_CONFIG = {
    "observation": {
        "type": "Kinematics",
        "vehicles_count": 15,
        "features": ["x", "y", "vx", "vy", "cos_h", "sin_h"],
        "normalize": True,
        "absolute": False,
    },
    "action": {"type": "DiscreteMetaAction"},
    "lanes_count": 4,
    "vehicles_count": 50,
    "duration": 40,
    "initial_spacing": 2,
    "collision_reward": -1,
    "right_lane_reward": 0.1,
    "high_speed_reward": 0.4,
    "reward_speed_range": [20, 30],
    "normalize_reward": True,
    "simulation_frequency": 5,
    "policy_frequency": 1,
    "other_vehicles_type": "highway_env.vehicle.behavior.IDMVehicle",
}

# ── C51 Hyperparameters ──────────────────────────────────────────────────────────
N_ATOMS  = 51        # Number of categorical atoms
V_MIN    = -10.0     # Min support value
V_MAX    = 40.0      # Max support value (max possible episode reward ≈ 40)
DELTA_Z  = (V_MAX - V_MIN) / (N_ATOMS - 1)
SUPPORT  = th.linspace(V_MIN, V_MAX, N_ATOMS)  # Will be moved to device at runtime


# ════════════════════════════════════════════════════════════════════════════════
# C51 Q-Network: Distributional + Dueling + Noisy
# ════════════════════════════════════════════════════════════════════════════════
class C51QNetwork(nn.Module):
    """
    Full C51 network:
      - Shared NoisyLinear layers (replaces ε-greedy exploration)
      - Dueling streams: Value distribution V(s) and Advantage distribution A(s,a)
      - Output: logits of shape (batch, n_actions, n_atoms)
      - Call .get_q_values() to get E[Z(s,a)] for action selection
    """

    def __init__(
        self,
        observation_space,
        action_space,
        features_extractor: BaseFeaturesExtractor,
        features_dim: int,
        net_arch: Optional[List[int]] = None,
        n_atoms: int = N_ATOMS,
        activation_fn: type = nn.ReLU,
    ):
        super().__init__()
        self.features_extractor = features_extractor
        self.n_atoms   = n_atoms
        self.n_actions = int(action_space.n)
        net_arch       = net_arch or [256, 256]

        # ── Shared NoisyLinear backbone ─────────────────────────────────────
        layers = []
        in_dim = features_dim
        for h in net_arch:
            layers += [NoisyLinear(in_dim, h), activation_fn()]
            in_dim   = h
        self.shared = nn.Sequential(*layers)

        # ── Dueling: value stream outputs n_atoms scalars ───────────────────
        self.value_stream = nn.Sequential(
            NoisyLinear(in_dim, 128), activation_fn(),
            NoisyLinear(128, n_atoms),
        )

        # ── Dueling: advantage stream outputs n_actions × n_atoms ───────────
        self.advantage_stream = nn.Sequential(
            NoisyLinear(in_dim, 128), activation_fn(),
            NoisyLinear(128, self.n_actions * n_atoms),
        )

    def reset_noise(self):
        for m in self.modules():
            if isinstance(m, NoisyLinear):
                m.reset_noise()

    def forward(self, obs: th.Tensor) -> th.Tensor:
        """
        Returns log-probabilities: shape (batch, n_actions, n_atoms).
        Uses log_softmax for numerical stability.
        """
        features = self.features_extractor(obs)
        shared   = self.shared(features)                          # (B, H)

        value     = self.value_stream(shared)                     # (B, n_atoms)
        advantage = self.advantage_stream(shared)                 # (B, n_actions*n_atoms)
        advantage = advantage.view(-1, self.n_actions, self.n_atoms)
        value     = value.unsqueeze(1)                            # (B, 1, n_atoms)

        # Dueling combination in the atom dimension
        q_atoms = value + advantage - advantage.mean(dim=1, keepdim=True)  # (B, n_actions, n_atoms)
        return F.log_softmax(q_atoms, dim=2)  # log-probs per atom

    def get_q_values(self, obs: th.Tensor, support: th.Tensor) -> th.Tensor:
        """Expected Q-values: E[Z] = sum_i(p_i * z_i). Shape: (B, n_actions)."""
        log_probs = self.forward(obs)                  # (B, n_actions, n_atoms)
        probs     = log_probs.exp()
        return (probs * support.unsqueeze(0).unsqueeze(0)).sum(dim=2)  # (B, n_actions)


# ════════════════════════════════════════════════════════════════════════════════
# Full C51 Rainbow Agent (standalone, does not subclass SB3 DQN)
# ════════════════════════════════════════════════════════════════════════════════
class C51RainbowAgent:
    """
    Standalone C51 Rainbow = C51 + Double DQN + Dueling + NoisyNets + PER + n-step.
    Manages its own training loop for maximum flexibility.
    """

    def __init__(
        self,
        env: gym.Env,
        lr: float = 5e-4,
        buffer_size: int = 100_000,
        batch_size: int = 64,
        gamma: float = 0.99,
        n_step: int = 3,
        target_update_freq: int = 1000,
        train_freq: int = 4,
        learning_starts: int = 1000,
        net_arch: Optional[List[int]] = None,
        alpha: float = 0.6,
        beta_start: float = 0.4,
        total_steps: int = 500_000,
        device: str = "auto",
        seed: int = 42,
        log_dir: str = LOG_DIR,
    ):
        self.env           = env
        self.gamma         = gamma
        self.n_step        = n_step
        self.batch_size    = batch_size
        self.train_freq    = train_freq
        self.learning_starts = learning_starts
        self.target_update_freq = target_update_freq
        self.total_steps   = total_steps
        self.log_dir       = log_dir
        os.makedirs(log_dir, exist_ok=True)

        self.device = th.device(
            "cuda" if device == "auto" and th.cuda.is_available() else
            "cpu" if device == "auto" else device
        )

        # Support on device
        self.support = SUPPORT.to(self.device)

        # Feature extractor (flatten obs)
        obs_space     = env.observation_space
        features_dim  = int(np.prod(obs_space.shape))
        feat_extractor = FlattenExtractor(obs_space)

        # Networks
        self.online = C51QNetwork(
            obs_space, env.action_space, feat_extractor, features_dim, net_arch
        ).to(self.device)
        self.target = C51QNetwork(
            obs_space, env.action_space,
            FlattenExtractor(obs_space), features_dim, net_arch
        ).to(self.device)
        self.target.load_state_dict(self.online.state_dict())
        self.target.eval()

        self.optimizer = th.optim.Adam(self.online.parameters(), lr=lr)

        # PER Buffer with n-step
        self.replay = PrioritizedReplayBuffer(
            buffer_size, obs_space, env.action_space,
            device=str(self.device), alpha=alpha,
            beta_start=beta_start, beta_frames=total_steps,
            n_step=n_step, gamma=gamma,
        )

        self._step    = 0
        self._episode = 0
        self._rng     = np.random.default_rng(seed)

    def _project_distribution(
        self,
        rewards: th.Tensor,  # (B,)
        next_log_probs: th.Tensor,  # (B, n_actions, n_atoms) from target
        dones: th.Tensor,    # (B,)
        next_actions: th.Tensor,  # (B,) — Double DQN: online selects action
    ) -> th.Tensor:
        """
        Project the Bellman-updated distribution onto the fixed support atoms.
        Returns target probabilities: shape (B, n_atoms).

        Algorithm (Bellemare et al. 2017, Algorithm 1):
          For each atom z_j of the return distribution:
            Compute projected atom: tz_j = clip(r + γ·z_j, V_min, V_max)
            Find the two nearest atoms: b_j = (tz_j - V_min) / Δz
            Split probability mass proportionally between floor(b_j) and ceil(b_j)
        """
        B      = rewards.size(0)
        target = th.zeros(B, N_ATOMS, device=self.device)

        # Get probability of next_actions from target network
        next_probs = next_log_probs.exp()                          # (B, n_actions, N_ATOMS)
        # Select the action chosen by online network (Double DQN)
        next_action_probs = next_probs[
            th.arange(B), next_actions
        ]  # (B, N_ATOMS)

        for j in range(N_ATOMS):
            z_j = self.support[j]
            # Projected atom (Bellman update)
            tz_j = (rewards + (1.0 - dones) * self.gamma * z_j).clamp(V_MIN, V_MAX)
            # Relative position in support
            b_j  = (tz_j - V_MIN) / DELTA_Z
            l    = b_j.floor().long().clamp(0, N_ATOMS - 1)
            u    = b_j.ceil().long().clamp(0, N_ATOMS - 1)

            p_j  = next_action_probs[:, j]

            # Distribute probability mass
            target.scatter_add_(1, l.unsqueeze(1), (p_j * (u.float() - b_j)).unsqueeze(1))
            target.scatter_add_(1, u.unsqueeze(1), (p_j * (b_j - l.float())).unsqueeze(1))

        return target  # (B, N_ATOMS) — target probabilities

    def _update(self):
        """One gradient step with C51 cross-entropy loss + PER IS weights."""
        if self.replay.sum_tree.n_entries < self.batch_size:
            return

        samples, is_weights, indices = self.replay.sample(self.batch_size)

        obs       = th.tensor(samples.observations.float(), device=self.device)
        next_obs  = th.tensor(samples.next_observations.float(), device=self.device)
        actions   = samples.actions.long().squeeze(1)
        rewards   = samples.rewards.squeeze(1)
        dones     = samples.dones.squeeze(1)

        with th.no_grad():
            # Double DQN: online selects, target evaluates
            next_q_online  = self.online.get_q_values(next_obs, self.support)
            next_actions   = next_q_online.argmax(dim=1)
            next_log_probs = self.target(next_obs)  # (B, n_actions, N_ATOMS)

            target_probs = self._project_distribution(rewards, next_log_probs, dones, next_actions)

        # Log-probabilities for selected actions
        log_probs = self.online(obs)                              # (B, n_actions, N_ATOMS)
        log_probs_selected = log_probs[th.arange(len(actions)), actions]  # (B, N_ATOMS)

        # C51 loss: cross-entropy between target distribution and predicted log-probs
        elementwise_loss = -(target_probs * log_probs_selected).sum(dim=1)  # (B,)

        # PER importance-sampling weighted loss
        loss = (is_weights.squeeze(1) * elementwise_loss).mean()

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(), 10.0)
        self.optimizer.step()

        # Update priorities with TD errors (approximate as elementwise loss)
        td_errors = elementwise_loss.detach().cpu().numpy()
        self.replay.update_priorities(indices, td_errors)

        # Reset NoisyNet noise
        self.online.reset_noise()
        self.target.reset_noise()

        return float(loss.item())

    def learn(self, checkpoint_dir: str = MODEL_DIR, checkpoint_freq: int = 100_000):
        """Main training loop."""
        obs, _ = self.env.reset()
        ep_reward = 0.0
        ep_rewards = []

        print(f"\n{'='*60}")
        print(f"  Training: C51 Rainbow (6/6 components)")
        print(f"  Device  : {self.device}")
        print(f"  Steps   : {self.total_steps:,}")
        print(f"  Atoms   : {N_ATOMS}  |  V_min={V_MIN}  V_max={V_MAX}")
        print(f"{'='*60}\n")

        for step in range(self.total_steps):
            self._step = step

            # Action selection: use E[Z] from online network (no ε-greedy)
            obs_t = th.tensor(obs, dtype=th.float32, device=self.device).unsqueeze(0)
            with th.no_grad():
                q_vals  = self.online.get_q_values(obs_t, self.support)
                action  = int(q_vals.argmax(dim=1).item())

            next_obs, reward, done, truncated, info = self.env.step(action)
            ep_reward += reward

            self.replay.add(
                obs.reshape(1, -1), next_obs.reshape(1, -1),
                np.array([action]), np.array([reward]),
                np.array([done or truncated]), [{}],
            )

            obs = next_obs

            if done or truncated:
                ep_rewards.append(ep_reward)
                self._episode += 1
                if self._episode % 50 == 0:
                    mean_r = np.mean(ep_rewards[-50:])
                    print(f"  Step {step:>8,} | Ep {self._episode:>5} | "
                          f"Mean Reward (last 50): {mean_r:>7.2f} | "
                          f"β={self.replay.beta:.3f}")
                obs, _    = self.env.reset()
                ep_reward = 0.0

            # Training update
            if step >= self.learning_starts and step % self.train_freq == 0:
                self._update()

            # Target network update
            if step % self.target_update_freq == 0:
                self.target.load_state_dict(self.online.state_dict())

            # Checkpoint
            if step > 0 and step % checkpoint_freq == 0:
                ckpt_path = os.path.join(checkpoint_dir, f"c51_rainbow_{step}_steps.pt")
                th.save({
                    "step": step,
                    "model_state": self.online.state_dict(),
                    "optimizer_state": self.optimizer.state_dict(),
                }, ckpt_path)
                print(f"  💾  Checkpoint saved: {ckpt_path}")

    def save(self, path: str):
        """Save full model state."""
        th.save({
            "step": self._step,
            "model_state": self.online.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "n_atoms": N_ATOMS,
            "v_min": V_MIN,
            "v_max": V_MAX,
        }, path)
        print(f"  ✅  Model saved: {path}")

    def load(self, path: str):
        """Load model state."""
        ckpt = th.load(path, map_location=self.device)
        self.online.load_state_dict(ckpt["model_state"])
        self.target.load_state_dict(ckpt["model_state"])
        self.optimizer.load_state_dict(ckpt["optimizer_state"])
        self._step = ckpt.get("step", 0)
        print(f"  ✅  Loaded checkpoint: {path} (step={self._step:,})")


def make_env():
    env = gym.make("highway-v0", config=ENV_CONFIG)
    env = LeftLaneRewardWrapper(env)
    env = Monitor(env)
    return env


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps",  type=int, default=500_000)
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint .pt file to resume from")
    args = parser.parse_args()

    env   = make_env()
    agent = C51RainbowAgent(
        env=env,
        lr=5e-4,
        buffer_size=100_000,
        batch_size=64,
        gamma=0.99,
        n_step=3,
        target_update_freq=1000,
        train_freq=4,
        learning_starts=1000,
        net_arch=[256, 256],
        alpha=0.6,
        beta_start=0.4,
        total_steps=args.steps,
        device="auto",
        seed=42,
        log_dir=LOG_DIR,
    )

    if args.resume:
        agent.load(args.resume)

    agent.learn(checkpoint_dir=MODEL_DIR, checkpoint_freq=100_000)
    agent.save(os.path.join(MODEL_DIR, "c51_rainbow_final.pt"))

    print(f"\n✅ C51 Rainbow training complete → {MODEL_DIR}/c51_rainbow_final.pt")
    env.close()
