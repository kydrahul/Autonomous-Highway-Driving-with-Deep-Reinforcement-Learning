"""
Rainbow DQN (5/6 components — implemented inside SB3)
=======================================================
Algorithm : Rainbow DQN (Hessel et al. 2017) — Partial
Purpose   : Best DQN variant, combines all major improvements
Device    : CUDA (auto-detected)
Steps     : 500,000

Components implemented:
  ✅ Double DQN          → online net selects, target net evaluates
  ✅ Dueling DQN         → V(s) + A(s,a) − mean(A) architecture
  ✅ Prioritized Replay  → SumTree, priority ∝ |TD error|^α
  ✅ Multi-step Returns  → n=3 step bootstrapped targets
  ✅ Noisy Networks      → NoisyLinear replaces epsilon-greedy
  ❌ Distributional C51  → skipped (requires full output layer redesign)

How it's built inside SB3:
  - NoisyLinear + Dueling → custom QNetwork subclass
  - PER + n-step          → custom ReplayBuffer subclass
  - Double DQN            → override train()
  - No epsilon-greedy     → exploration_fraction=0, eps_final=0
"""

import os
import math
import numpy as np
import gymnasium as gym
import highway_env  # noqa: F401
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
from typing import Dict, List, Optional, Tuple, Type, Union

from stable_baselines3 import DQN
from stable_baselines3.dqn.policies import DQNPolicy, QNetwork
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.type_aliases import ReplayBufferSamples

# ── Paths ───────────────────────────────────────────────────────────────────────
MODEL_DIR = "models/rainbow_dqn"
LOG_DIR   = "logs/rainbow_dqn"
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOG_DIR,   exist_ok=True)

# ── Standardized Environment Config ─────────────────────────────────────────────
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

# ── Left Lane Reward Wrapper ─────────────────────────────────────────────────────
class LeftLaneRewardWrapper(gym.Wrapper):
    """Adds a bonus reward for staying in the left-most lane."""
    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)
        current_lane = self.unwrapped.vehicle.lane_index[2]
        total_lanes  = self.unwrapped.config["lanes_count"]
        left_reward  = (total_lanes - 1 - current_lane) / (total_lanes - 1)
        reward      += 0.1 * left_reward
        return obs, reward, done, truncated, info


# ══════════════════════════════════════════════════════════════════════════════════
# COMPONENT 1 — Noisy Networks
# Replaces standard Linear layers with NoisyLinear.
# Noise parameters are LEARNED, replacing epsilon-greedy exploration.
# ══════════════════════════════════════════════════════════════════════════════════
class NoisyLinear(nn.Module):
    """
    NoisyLinear layer (Fortunato et al. 2017).
    weight = weight_mu + weight_sigma ⊙ weight_epsilon
    bias   = bias_mu   + bias_sigma   ⊙ bias_epsilon

    Uses factorised Gaussian noise for efficiency.
    reset_noise() must be called before each forward pass during training.
    """

    def __init__(self, in_features: int, out_features: int, sigma_init: float = 0.5):
        super().__init__()
        self.in_features  = in_features
        self.out_features = out_features

        # Learnable parameters — mean and sigma for weight and bias
        self.weight_mu    = nn.Parameter(th.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(th.empty(out_features, in_features))
        self.bias_mu      = nn.Parameter(th.empty(out_features))
        self.bias_sigma   = nn.Parameter(th.empty(out_features))

        # Noise buffers (not parameters — not optimised)
        self.register_buffer("weight_epsilon", th.empty(out_features, in_features))
        self.register_buffer("bias_epsilon",   th.empty(out_features))

        self.sigma_init = sigma_init
        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        """Initialise mu with uniform and sigma with a fixed constant."""
        mu_range = 1.0 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.sigma_init / math.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.sigma_init / math.sqrt(self.out_features))

    @staticmethod
    def _scale_noise(size: int) -> th.Tensor:
        """Factorised noise: f(x) = sgn(x) · √|x|"""
        x = th.randn(size)
        return x.sign().mul_(x.abs().sqrt_())

    def reset_noise(self):
        """Sample new noise (call once per training step, before forward)."""
        eps_in  = self._scale_noise(self.in_features)
        eps_out = self._scale_noise(self.out_features)
        self.weight_epsilon.copy_(eps_out.outer(eps_in))
        self.bias_epsilon.copy_(eps_out)

    def forward(self, x: th.Tensor) -> th.Tensor:
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias   = self.bias_mu   + self.bias_sigma   * self.bias_epsilon
        else:
            # At evaluation time, use only the mean (no noise)
            weight = self.weight_mu
            bias   = self.bias_mu
        return F.linear(x, weight, bias)


# ══════════════════════════════════════════════════════════════════════════════════
# COMPONENT 2+3 — Dueling Architecture + Noisy Networks combined
# ══════════════════════════════════════════════════════════════════════════════════
class RainbowQNetwork(QNetwork):
    """
    Rainbow Q-Network:
      - Shared layers use NoisyLinear (learned exploration)
      - Splits into Value stream + Advantage stream (Dueling)
      - Q(s,a) = V(s) + A(s,a) − mean(A(s,a))
    """

    def __init__(
        self,
        observation_space,
        action_space,
        features_extractor: BaseFeaturesExtractor,
        features_dim: int,
        net_arch: Optional[List[int]] = None,
        activation_fn: Type[nn.Module] = nn.ReLU,
        normalize_images: bool = True,
    ):
        super().__init__(
            observation_space, action_space, features_extractor,
            features_dim, net_arch, activation_fn, normalize_images,
        )

        action_dim = int(action_space.n)
        net_arch   = net_arch if net_arch is not None else [256, 256]

        # ── Shared NoisyLinear layers ────────────────────────────────────────
        shared_layers = []
        in_dim = features_dim
        for size in net_arch:
            shared_layers.append(NoisyLinear(in_dim, size))
            shared_layers.append(activation_fn())
            in_dim = size
        self.q_net = nn.Sequential(*shared_layers)  # override parent's q_net

        # ── Dueling streams with NoisyLinear ────────────────────────────────
        self.value_stream = nn.Sequential(
            NoisyLinear(in_dim, 128),
            activation_fn(),
            NoisyLinear(128, 1),
        )
        self.advantage_stream = nn.Sequential(
            NoisyLinear(in_dim, 128),
            activation_fn(),
            NoisyLinear(128, action_dim),
        )

    def reset_noise(self):
        """Reset noise in all NoisyLinear layers."""
        for module in self.modules():
            if isinstance(module, NoisyLinear):
                module.reset_noise()

    def forward(self, obs: th.Tensor) -> th.Tensor:
        features  = self.extract_features(obs, self.features_extractor)
        shared    = self.q_net(features)
        value     = self.value_stream(shared)
        advantage = self.advantage_stream(shared)
        return value + advantage - advantage.mean(dim=1, keepdim=True)


# ── Rainbow Policy ───────────────────────────────────────────────────────────────
class RainbowPolicy(DQNPolicy):
    """DQN policy using RainbowQNetwork (Noisy + Dueling)."""

    def make_q_net(self) -> RainbowQNetwork:
        net_args = self._update_features_extractor(self.net_args, features_extractor=None)
        return RainbowQNetwork(**net_args).to(self.device)


# ══════════════════════════════════════════════════════════════════════════════════
# COMPONENT 4 — SumTree for Prioritized Experience Replay
# ══════════════════════════════════════════════════════════════════════════════════
class SumTree:
    """
    Binary tree where:
      - Leaves store individual priorities p_i
      - Internal nodes store sum of child subtrees
      - Root = total priority sum

    Enables O(log n) priority sampling.
    """

    def __init__(self, capacity: int):
        self.capacity  = capacity
        self.tree      = np.zeros(2 * capacity - 1, dtype=np.float64)
        self.write_ptr = 0
        self.n_entries = 0

    def _propagate(self, idx: int, delta: float):
        """Propagate priority change up to root."""
        parent = (idx - 1) // 2
        self.tree[parent] += delta
        if parent != 0:
            self._propagate(parent, delta)

    def _retrieve(self, idx: int, s: float) -> int:
        """Traverse tree to find leaf for cumulative sum s."""
        left  = 2 * idx + 1
        right = left + 1
        if left >= len(self.tree):
            return idx
        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    @property
    def total(self) -> float:
        return float(self.tree[0])

    @property
    def min_priority(self) -> float:
        """Minimum leaf priority (for IS weight normalisation)."""
        leaves = self.tree[self.capacity - 1 : self.capacity - 1 + self.n_entries]
        return float(np.min(leaves)) if self.n_entries > 0 else 1.0

    def add(self, priority: float) -> int:
        """Store priority at current write pointer, return data index."""
        leaf_idx  = self.write_ptr + self.capacity - 1
        data_idx  = self.write_ptr
        self.update(leaf_idx, priority)
        self.write_ptr = (self.write_ptr + 1) % self.capacity
        self.n_entries = min(self.n_entries + 1, self.capacity)
        return data_idx

    def update(self, leaf_idx: int, priority: float):
        """Update priority at leaf_idx and propagate change."""
        delta = priority - self.tree[leaf_idx]
        self.tree[leaf_idx] = priority
        self._propagate(leaf_idx, delta)

    def get(self, s: float) -> Tuple[int, float, int]:
        """
        Sample by cumulative priority value s.
        Returns (leaf_idx, priority, data_idx).
        """
        leaf_idx = self._retrieve(0, s)
        data_idx = leaf_idx - self.capacity + 1
        priority = self.tree[leaf_idx]
        return leaf_idx, priority, data_idx


# ══════════════════════════════════════════════════════════════════════════════════
# COMPONENT 5 — Prioritized Replay Buffer with Multi-step Returns
# ══════════════════════════════════════════════════════════════════════════════════
class PrioritizedReplayBuffer(ReplayBuffer):
    """
    Prioritized Experience Replay (PER) with n-step returns.

    PER: transitions are sampled with probability proportional to |TD error|^α
    Importance sampling (IS) weights correct for the non-uniform sampling bias.

    n-step: instead of storing (s_t, a_t, r_t, s_{t+1}),
            stores (s_t, a_t, G_t, s_{t+n}) where
            G_t = r_t + γ·r_{t+1} + γ²·r_{t+2} + γ³·Q(s_{t+n})
    """

    def __init__(
        self,
        buffer_size: int,
        observation_space,
        action_space,
        device: Union[th.device, str] = "auto",
        n_envs: int = 1,
        optimize_memory_usage: bool = False,
        handle_timeout_termination: bool = True,
        alpha: float = 0.6,
        beta_start: float = 0.4,
        beta_frames: int = 500_000,
        n_step: int = 3,
        gamma: float = 0.99,
    ):
        super().__init__(
            buffer_size, observation_space, action_space,
            device, n_envs, optimize_memory_usage, handle_timeout_termination,
        )
        self.alpha        = alpha           # priority exponent
        self.beta         = beta_start      # IS weight exponent (anneals to 1.0)
        self.beta_start   = beta_start
        self.beta_frames  = beta_frames
        self.max_priority = 1.0
        self.epsilon      = 1e-6            # small constant to avoid zero priority
        self.n_step       = n_step
        self.gamma        = gamma
        self.n_step_buffer: deque = deque(maxlen=n_step)
        self._frame       = 0               # used for beta annealing

        # SumTree tracks priorities for all buffer positions
        self.sum_tree = SumTree(buffer_size)

    # ── n-step helper ────────────────────────────────────────────────────────────
    def _get_n_step_transition(self) -> Tuple:
        """
        Compute n-step return G_t from the n-step buffer.
        If an intermediate step is terminal (done=True), truncate there.

        Returns: (obs_0, action_0, n_step_reward, next_obs_n, done_n)
        """
        obs_0, action_0, _, _, _ = self.n_step_buffer[0]
        n_step_reward = 0.0

        for i, (_, _, reward, next_obs, done) in enumerate(self.n_step_buffer):
            n_step_reward += (self.gamma ** i) * float(np.asarray(reward).flat[0])
            if done:
                return obs_0, action_0, n_step_reward, next_obs, True

        _, _, _, next_obs_n, done_n = self.n_step_buffer[-1]
        return obs_0, action_0, n_step_reward, next_obs_n, bool(np.asarray(done_n).flat[0])

    def add(
        self,
        obs: np.ndarray,
        next_obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        infos: List[Dict],
    ) -> None:
        """
        Buffer the transition in n_step_buffer.
        Once we have n transitions, compute n-step return and add to replay buffer.
        On done=True, flush all remaining transitions.
        """
        _reward = float(np.asarray(reward).flat[0])
        _done   = bool(np.asarray(done).flat[0])
        self.n_step_buffer.append((
            obs.copy(), action.copy(), _reward,
            next_obs.copy(), _done,
        ))

        # Only add to replay buffer when we have n steps buffered
        if len(self.n_step_buffer) == self.n_step:
            self._flush_oldest()

        # On episode end, flush remaining transitions (shorter n-step windows)
        if _done:
            while len(self.n_step_buffer) > 0:
                self._flush_oldest()

    def _flush_oldest(self):
        """Compute n-step return for oldest transition and add to replay buffer."""
        if len(self.n_step_buffer) == 0:
            return

        obs_0, action_0, n_step_reward, next_obs_n, done_n = self._get_n_step_transition()

        # Record position before super().add() increments it
        stored_pos = self.pos

        # Add to parent ReplayBuffer using numpy scalars to match expected shapes
        super().add(
            obs_0,
            next_obs_n,
            action_0,
            np.array([n_step_reward]),
            np.array([done_n]),
            [{}],
        )

        # Update SumTree with max priority (new transitions get highest priority)
        priority = self.max_priority ** self.alpha
        self.sum_tree.update(stored_pos + self.sum_tree.capacity - 1, priority)

        # Keep SumTree.n_entries in sync with ReplayBuffer (update() skips this)
        self.sum_tree.n_entries = self.buffer_size if self.full else self.pos

        # Remove oldest from n-step buffer
        self.n_step_buffer.popleft()

    def sample(self, batch_size: int, env=None) -> Tuple[ReplayBufferSamples, th.Tensor, np.ndarray]:
        """
        Priority-based sampling. Returns (samples, IS_weights, indices).
        IS weights correct for bias introduced by non-uniform sampling.
        """
        assert self.sum_tree.n_entries >= batch_size, (
            f"Not enough samples: {self.sum_tree.n_entries} < {batch_size}"
        )

        # Anneal beta from beta_start → 1.0 over beta_frames
        self._frame += 1
        self.beta = min(1.0, self.beta_start + self._frame * (1.0 - self.beta_start) / self.beta_frames)

        indices    = np.empty(batch_size, dtype=np.int64)
        leaf_indices = np.empty(batch_size, dtype=np.int64)
        priorities = np.empty(batch_size, dtype=np.float64)

        total     = self.sum_tree.total
        segment   = total / batch_size

        for i in range(batch_size):
            s                    = np.random.uniform(segment * i, segment * (i + 1))
            leaf_idx, p, data_idx = self.sum_tree.get(s)
            # Clamp to valid range
            data_idx              = max(0, min(data_idx, self.buffer_size - 1))
            indices[i]            = data_idx
            leaf_indices[i]       = leaf_idx
            priorities[i]         = p

        # ── Importance Sampling weights ──────────────────────────────────────
        # w_i = (N · P(i))^(−β) normalised by max weight
        n = self.sum_tree.n_entries
        p_min = self.sum_tree.min_priority / total    # minimum sampling probability
        max_weight = (n * p_min) ** (-self.beta)

        weights = np.empty(batch_size, dtype=np.float32)
        for i, p in enumerate(priorities):
            p_sample    = p / total
            weight      = (n * p_sample) ** (-self.beta)
            weights[i]  = weight / max_weight           # normalise

        # Store leaf_indices for later priority update
        self._last_leaf_indices = leaf_indices

        # Fetch the actual transitions from internal arrays
        samples = self._get_samples(indices, env=env)
        weights_tensor = th.tensor(weights, dtype=th.float32, device=self.device).unsqueeze(1)
        return samples, weights_tensor, indices

    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray):
        """Update priorities based on new TD errors (call after each train step)."""
        priorities = (np.abs(td_errors) + self.epsilon) ** self.alpha
        for idx, leaf_idx, priority in zip(indices, self._last_leaf_indices, priorities):
            self.sum_tree.update(int(leaf_idx), float(priority))
            self.max_priority = max(self.max_priority, float(priority))


# ══════════════════════════════════════════════════════════════════════════════════
# Rainbow DQN — Combines all components in one training loop
# ══════════════════════════════════════════════════════════════════════════════════
class RainbowDQN(DQN):
    """
    Rainbow DQN = Double DQN + Dueling + PER + Multi-step + NoisyNets

    Overrides train() to:
      1. Sample with priorities (PER)
      2. Apply importance-sampling weights to loss
      3. Use Double DQN target computation
      4. Update priorities with new TD errors
      5. Reset noise after each update
    """

    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)

        losses = []
        for _ in range(gradient_steps):
            # ── PER: Sample with priorities ───────────────────────────────────
            replay_data, is_weights, indices = self.replay_buffer.sample(
                batch_size, env=self._vec_normalize_env
            )

            with th.no_grad():
                # ── Double DQN target ──────────────────────────────────────────
                # Online network selects next action
                next_actions = self.q_net(replay_data.next_observations).argmax(dim=1, keepdim=True)
                # Target network evaluates that action
                next_q_values = th.gather(
                    self.q_net_target(replay_data.next_observations),
                    dim=1,
                    index=next_actions,
                )
                target_q_values = (
                    replay_data.rewards
                    + (1 - replay_data.dones) * self.gamma * next_q_values
                )

            current_q_values = th.gather(
                self.q_net(replay_data.observations),
                dim=1,
                index=replay_data.actions.long(),
            )

            # ── TD errors for priority update ─────────────────────────────────
            td_errors = (current_q_values - target_q_values).detach().abs().squeeze(1).cpu().numpy()

            # ── Weighted Huber loss (IS-corrected) ────────────────────────────
            elementwise_loss = F.smooth_l1_loss(current_q_values, target_q_values, reduction="none")
            loss = (is_weights * elementwise_loss).mean()

            self.policy.optimizer.zero_grad()
            loss.backward()
            th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.policy.optimizer.step()

            # ── Update priorities ──────────────────────────────────────────────
            self.replay_buffer.update_priorities(indices, td_errors)

            # ── Reset noise for next step ──────────────────────────────────────
            self.q_net.reset_noise()
            self.q_net_target.reset_noise()

            losses.append(loss.item())

        self._n_updates += gradient_steps
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/loss",      float(np.mean(losses)))
        self.logger.record("train/beta",      self.replay_buffer.beta)


def make_env():
    env = gym.make("highway-v0", config=ENV_CONFIG)
    env = LeftLaneRewardWrapper(env)
    env = Monitor(env)
    return env


def find_latest_checkpoint(model_dir: str, prefix: str):
    import re, glob
    files = glob.glob(os.path.join(model_dir, f"{prefix}_*_steps.zip"))
    best_path, best_steps = None, 0
    for f in files:
        m = re.search(rf"{prefix}_(\d+)_steps\.zip", f)
        if m:
            steps = int(m.group(1))
            if steps > best_steps:
                best_steps, best_path = steps, f
    return best_path, best_steps


# ── Main ─────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    TOTAL_STEPS = 500_000
    env = make_env()

    ckpt_path, done_steps = find_latest_checkpoint(MODEL_DIR, "rainbow_dqn")

    if ckpt_path and done_steps >= TOTAL_STEPS:
        print(f"✅ Rainbow DQN already fully trained ({done_steps} steps). Nothing to do.")
        env.close()
        exit(0)
    elif ckpt_path:
        print(f"▶  Resuming Rainbow DQN from {ckpt_path} ({done_steps:,} / {TOTAL_STEPS:,} steps done)")
        model = RainbowDQN.load(
            ckpt_path, env=env, device="auto",
            custom_objects={
                "replay_buffer_class": PrioritizedReplayBuffer,
                "replay_buffer_kwargs": dict(
                    alpha=0.6, beta_start=0.4, beta_frames=500_000, n_step=3, gamma=0.99,
                ),
            },
        )
        remaining   = TOTAL_STEPS - done_steps
        reset_steps = False
    else:
        print("▶  Starting Rainbow DQN from scratch")
        model = RainbowDQN(
            policy=RainbowPolicy,
            env=env,
            replay_buffer_class=PrioritizedReplayBuffer,
            replay_buffer_kwargs=dict(
                alpha=0.6,
                beta_start=0.4,
                beta_frames=500_000,
                n_step=3,
                gamma=0.99,
            ),
            learning_rate=5e-4,
            buffer_size=100_000,
            batch_size=64,
            gamma=0.99,
            tau=1.0,
            target_update_interval=1000,
            train_freq=4,
            gradient_steps=1,
            learning_starts=1000,
            exploration_fraction=0.0,
            exploration_initial_eps=0.0,
            exploration_final_eps=0.0,
            policy_kwargs=dict(net_arch=[256, 256]),
            tensorboard_log=LOG_DIR,
            device="auto",
            verbose=1,
            seed=42,
        )
        remaining   = TOTAL_STEPS
        reset_steps = True

    checkpoint_cb = CheckpointCallback(
        save_freq=100_000,
        save_path=MODEL_DIR,
        name_prefix="rainbow_dqn",
        save_replay_buffer=False,
        verbose=1,
    )

    print("=" * 60)
    print("  Training: Rainbow DQN (5/6 components)")
    print("  Components: Double + Dueling + PER + n-step + NoisyNets")
    print(f"  Device        : {model.device}")
    print(f"  Remaining steps: {remaining:,}")
    print(f"  Logs          : {LOG_DIR}")
    print(f"  Models        : {MODEL_DIR}")
    print("=" * 60)

    model.learn(
        total_timesteps=remaining,
        callback=checkpoint_cb,
        tb_log_name="RainbowDQN",
        reset_num_timesteps=reset_steps,
        progress_bar=True,
    )

    model.save(f"{MODEL_DIR}/rainbow_dqn_final")
    print(f"\n✅ Rainbow DQN training complete → {MODEL_DIR}/rainbow_dqn_final.zip")
    env.close()
