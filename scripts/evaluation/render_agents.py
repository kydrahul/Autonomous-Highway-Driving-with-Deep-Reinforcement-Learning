"""
Render Agents
==============
Loads a trained model and renders it live in a pygame window so you can
watch the agent drive.

Usage:
    # Watch all 5 models, 3 episodes each (default)
    python scripts/evaluation/render_agents.py

    # Watch only a specific model
    python scripts/evaluation/render_agents.py --model "Rainbow DQN"

    # Change number of episodes
    python scripts/evaluation/render_agents.py --model DQN --episodes 5

Controls:
    - Window closes automatically after each episode ends
    - Close the pygame window at any time to skip to the next model
"""

import argparse
import math
import os
import sys

import gymnasium as gym
import highway_env  # noqa: F401
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Type

from stable_baselines3 import DQN, PPO
from stable_baselines3.dqn.policies import DQNPolicy, QNetwork
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

# ── Environment Config ───────────────────────────────────────────────────────────
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
    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)
        current_lane = self.unwrapped.vehicle.lane_index[2]
        total_lanes  = self.unwrapped.config["lanes_count"]
        left_reward  = (total_lanes - 1 - current_lane) / (total_lanes - 1)
        reward      += 0.1 * left_reward
        return obs, reward, done, truncated, info


# ── Custom network classes (needed to load Dueling / Rainbow) ────────────────────
class DuelingQNetwork(QNetwork):
    def __init__(self, observation_space, action_space,
                 features_extractor: BaseFeaturesExtractor, features_dim: int,
                 net_arch=None, activation_fn=nn.ReLU, normalize_images=True):
        super().__init__(observation_space, action_space, features_extractor,
                         features_dim, net_arch, activation_fn, normalize_images)
        action_dim = int(action_space.n)
        net_arch   = net_arch if net_arch is not None else [256, 256]
        shared_layers, in_dim = [], features_dim
        for size in net_arch:
            shared_layers += [nn.Linear(in_dim, size), activation_fn()]
            in_dim = size
        self.q_net          = nn.Sequential(*shared_layers)
        self.value_stream   = nn.Sequential(nn.Linear(in_dim, 128), activation_fn(), nn.Linear(128, 1))
        self.advantage_stream = nn.Sequential(nn.Linear(in_dim, 128), activation_fn(), nn.Linear(128, action_dim))

    def forward(self, obs):
        shared    = self.q_net(self.extract_features(obs, self.features_extractor))
        value     = self.value_stream(shared)
        advantage = self.advantage_stream(shared)
        return value + advantage - advantage.mean(dim=1, keepdim=True)


class DuelingDQNPolicy(DQNPolicy):
    def make_q_net(self):
        return DuelingQNetwork(**self._update_features_extractor(self.net_args, features_extractor=None)).to(self.device)


class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, sigma_init=0.5):
        super().__init__()
        self.in_features  = in_features
        self.out_features = out_features
        self.weight_mu    = nn.Parameter(th.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(th.empty(out_features, in_features))
        self.register_buffer("weight_epsilon", th.empty(out_features, in_features))
        self.bias_mu      = nn.Parameter(th.empty(out_features))
        self.bias_sigma   = nn.Parameter(th.empty(out_features))
        self.register_buffer("bias_epsilon", th.empty(out_features))
        bound = 1.0 / math.sqrt(in_features)
        self.weight_mu.data.uniform_(-bound, bound)
        self.weight_sigma.data.fill_(sigma_init / math.sqrt(in_features))
        self.bias_mu.data.uniform_(-bound, bound)
        self.bias_sigma.data.fill_(sigma_init / math.sqrt(out_features))
        self.reset_noise()

    def _scale_noise(self, size):
        x = th.randn(size, device=self.weight_mu.device)
        return x.sign() * x.abs().sqrt()

    def reset_noise(self):
        eps_in  = self._scale_noise(self.in_features)
        eps_out = self._scale_noise(self.out_features)
        self.weight_epsilon.copy_(eps_out.outer(eps_in))
        self.bias_epsilon.copy_(eps_out)

    def forward(self, x):
        if self.training:
            w = self.weight_mu + self.weight_sigma * self.weight_epsilon
            b = self.bias_mu   + self.bias_sigma   * self.bias_epsilon
        else:
            w = self.weight_mu
            b = self.bias_mu
        return F.linear(x, w, b)


class RainbowQNetwork(QNetwork):
    def __init__(self, observation_space, action_space,
                 features_extractor: BaseFeaturesExtractor, features_dim: int,
                 net_arch=None, activation_fn=nn.ReLU, normalize_images=True):
        super().__init__(observation_space, action_space, features_extractor,
                         features_dim, net_arch, activation_fn, normalize_images)
        action_dim = int(action_space.n)
        net_arch   = net_arch if net_arch is not None else [256, 256]
        shared_layers, in_dim = [], features_dim
        for size in net_arch:
            shared_layers += [NoisyLinear(in_dim, size), activation_fn()]
            in_dim = size
        self.q_net          = nn.Sequential(*shared_layers)
        self.value_stream   = nn.Sequential(NoisyLinear(in_dim, 128), activation_fn(), NoisyLinear(128, 1))
        self.advantage_stream = nn.Sequential(NoisyLinear(in_dim, 128), activation_fn(), NoisyLinear(128, action_dim))

    def forward(self, obs):
        shared    = self.q_net(self.extract_features(obs, self.features_extractor))
        value     = self.value_stream(shared)
        advantage = self.advantage_stream(shared)
        return value + advantage - advantage.mean(dim=1, keepdim=True)

    def reset_noise(self):
        for m in self.modules():
            if isinstance(m, NoisyLinear):
                m.reset_noise()


class RainbowPolicy(DQNPolicy):
    def make_q_net(self):
        return RainbowQNetwork(**self._update_features_extractor(self.net_args, features_extractor=None)).to(self.device)


# ── Model Registry ───────────────────────────────────────────────────────────────
MODELS = {
    "DQN": {
        "path":   "models/dqn/dqn_final",
        "cls":    DQN,
        "kwargs": {},
    },
    "Double DQN": {
        "path":   "models/double_dqn/double_dqn_final",
        "cls":    DQN,
        "kwargs": {},
    },
    "Dueling DQN": {
        "path":   "models/dueling_dqn/dueling_dqn_final",
        "cls":    DQN,
        "kwargs": {"custom_objects": {"policy_class": DuelingDQNPolicy}},
    },
    "Rainbow DQN": {
        "path":   "models/rainbow_dqn/rainbow_dqn_final",
        "cls":    DQN,
        "kwargs": {"custom_objects": {"policy_class": RainbowPolicy}},
    },
    "PPO": {
        "path":   "models/ppo/ppo_final",
        "cls":    PPO,
        "kwargs": {},
    },
}


# ── Render one model ─────────────────────────────────────────────────────────────
def run_render(model_name: str, n_episodes: int = 3):
    config = MODELS[model_name]
    model_path = config["path"] + ".zip"

    if not os.path.exists(model_path):
        print(f"  ⚠️  Model not found: {model_path} — skipping")
        return

    # Create env with human render mode
    env = gym.make("highway-v0", render_mode="human", config=ENV_CONFIG)
    env = LeftLaneRewardWrapper(env)

    model = config["cls"].load(config["path"], env=env, **config["kwargs"])

    print(f"\n{'─'*55}")
    print(f"  Watching: {model_name}  ({n_episodes} episodes)")
    print(f"  Close the window to skip to next model")
    print(f"{'─'*55}")

    for ep in range(1, n_episodes + 1):
        obs, _ = env.reset()
        done       = False
        ep_reward  = 0.0
        ep_steps   = 0
        collision  = False

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            try:
                env.render()
            except Exception:
                pass

            ep_reward += reward
            ep_steps  += 1
            done       = terminated or truncated

            if info.get("crashed", False):
                collision = True

        status = "💥 CRASH" if collision else "✅ Safe"
        print(f"  Episode {ep:2d}: reward={ep_reward:6.2f}  steps={ep_steps:3d}  {status}")

    env.close()


# ── Main ─────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Watch trained highway-env agents")
    parser.add_argument(
        "--model", type=str, default=None,
        help="Model to render: 'DQN', 'Double DQN', 'Dueling DQN', 'Rainbow DQN', 'PPO'. "
             "Omit to cycle through all."
    )
    parser.add_argument(
        "--episodes", type=int, default=3,
        help="Number of episodes to render per model (default: 3)"
    )
    args = parser.parse_args()

    if args.model:
        if args.model not in MODELS:
            print(f"Unknown model '{args.model}'. Choose from: {list(MODELS.keys())}")
            sys.exit(1)
        run_render(args.model, args.episodes)
    else:
        print("=" * 55)
        print("  Highway-Env Agent Visualizer")
        print(f"  Rendering all 5 models × {args.episodes} episodes each")
        print("=" * 55)
        for name in MODELS:
            run_render(name, args.episodes)
        print("\n✅ All models rendered.")
