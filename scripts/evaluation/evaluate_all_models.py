"""
Evaluate All Models
====================
Loads the final trained model for each algorithm and runs 50 evaluation
episodes. Records per-episode metrics and saves a JSON summary.

Metrics collected:
  - Episode reward
  - Collision (True/False)
  - Average speed
  - Episode length
  - Lanes visited

Run:
    python scripts/evaluation/evaluate_all_models.py
"""

import os
import sys
import json
import numpy as np
import gymnasium as gym
import highway_env  # noqa: F401
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import List, Optional, Tuple, Type

from stable_baselines3 import DQN, PPO
from stable_baselines3.dqn.policies import DQNPolicy, QNetwork
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.monitor import Monitor

# ── Paths ───────────────────────────────────────────────────────────────────────
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

N_EVAL_EPISODES = 50

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
    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)
        current_lane = self.unwrapped.vehicle.lane_index[2]
        total_lanes  = self.unwrapped.config["lanes_count"]
        left_reward  = (total_lanes - 1 - current_lane) / (total_lanes - 1)
        reward      += 0.1 * left_reward
        return obs, reward, done, truncated, info


def make_eval_env():
    env = gym.make("highway-v0", config=ENV_CONFIG)
    env = LeftLaneRewardWrapper(env)
    return env


# ── Custom Classes — needed to load Dueling and Rainbow models ──────────────────

class DuelingQNetwork(QNetwork):
    def __init__(self, observation_space, action_space,
                 features_extractor: BaseFeaturesExtractor, features_dim: int,
                 net_arch=None, activation_fn=nn.ReLU, normalize_images=True):
        super().__init__(observation_space, action_space, features_extractor,
                         features_dim, net_arch, activation_fn, normalize_images)
        action_dim = int(action_space.n)
        net_arch   = net_arch if net_arch is not None else [256, 256]
        shared_layers = []
        in_dim = features_dim
        for size in net_arch:
            shared_layers.append(nn.Linear(in_dim, size))
            shared_layers.append(activation_fn())
            in_dim = size
        self.q_net = nn.Sequential(*shared_layers)
        self.value_stream = nn.Sequential(
            nn.Linear(in_dim, 128), activation_fn(), nn.Linear(128, 1))
        self.advantage_stream = nn.Sequential(
            nn.Linear(in_dim, 128), activation_fn(), nn.Linear(128, action_dim))

    def forward(self, obs):
        features  = self.extract_features(obs, self.features_extractor)
        shared    = self.q_net(features)
        value     = self.value_stream(shared)
        advantage = self.advantage_stream(shared)
        return value + advantage - advantage.mean(dim=1, keepdim=True)


class DuelingDQNPolicy(DQNPolicy):
    def make_q_net(self):
        net_args = self._update_features_extractor(self.net_args, features_extractor=None)
        return DuelingQNetwork(**net_args).to(self.device)


class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, sigma_init=0.5):
        super().__init__()
        self.in_features  = in_features
        self.out_features = out_features
        self.weight_mu    = nn.Parameter(th.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(th.empty(out_features, in_features))
        self.bias_mu      = nn.Parameter(th.empty(out_features))
        self.bias_sigma   = nn.Parameter(th.empty(out_features))
        self.register_buffer("weight_epsilon", th.empty(out_features, in_features))
        self.register_buffer("bias_epsilon",   th.empty(out_features))
        self.sigma_init   = sigma_init
        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        mu_range = 1.0 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.sigma_init / math.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.sigma_init / math.sqrt(self.out_features))

    @staticmethod
    def _scale_noise(size):
        x = th.randn(size)
        return x.sign().mul_(x.abs().sqrt_())

    def reset_noise(self):
        eps_in  = self._scale_noise(self.in_features)
        eps_out = self._scale_noise(self.out_features)
        self.weight_epsilon.copy_(eps_out.outer(eps_in))
        self.bias_epsilon.copy_(eps_out)

    def forward(self, x):
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias   = self.bias_mu   + self.bias_sigma   * self.bias_epsilon
        else:
            weight = self.weight_mu
            bias   = self.bias_mu
        return F.linear(x, weight, bias)


class RainbowQNetwork(QNetwork):
    def __init__(self, observation_space, action_space,
                 features_extractor: BaseFeaturesExtractor, features_dim: int,
                 net_arch=None, activation_fn=nn.ReLU, normalize_images=True):
        super().__init__(observation_space, action_space, features_extractor,
                         features_dim, net_arch, activation_fn, normalize_images)
        action_dim = int(action_space.n)
        net_arch   = net_arch if net_arch is not None else [256, 256]
        shared_layers = []
        in_dim = features_dim
        for size in net_arch:
            shared_layers.append(NoisyLinear(in_dim, size))
            shared_layers.append(activation_fn())
            in_dim = size
        self.q_net = nn.Sequential(*shared_layers)
        self.value_stream = nn.Sequential(
            NoisyLinear(in_dim, 128), activation_fn(), NoisyLinear(128, 1))
        self.advantage_stream = nn.Sequential(
            NoisyLinear(in_dim, 128), activation_fn(), NoisyLinear(128, action_dim))

    def reset_noise(self):
        for module in self.modules():
            if isinstance(module, NoisyLinear):
                module.reset_noise()

    def forward(self, obs):
        features  = self.extract_features(obs, self.features_extractor)
        shared    = self.q_net(features)
        value     = self.value_stream(shared)
        advantage = self.advantage_stream(shared)
        return value + advantage - advantage.mean(dim=1, keepdim=True)


class RainbowPolicy(DQNPolicy):
    def make_q_net(self):
        net_args = self._update_features_extractor(self.net_args, features_extractor=None)
        return RainbowQNetwork(**net_args).to(self.device)


# ── Model Registry ───────────────────────────────────────────────────────────────
MODELS = {
    "DQN": {
        "path":    "models/dqn/dqn_final",
        "cls":     DQN,
        "kwargs":  {},
    },
    "Double DQN": {
        "path":    "models/double_dqn/double_dqn_final",
        "cls":     DQN,
        "kwargs":  {},
    },
    "Dueling DQN": {
        "path":    "models/dueling_dqn/dueling_dqn_final",
        "cls":     DQN,
        "kwargs":  {"custom_objects": {"policy_class": DuelingDQNPolicy}},
    },
    "Rainbow DQN": {
        "path":    "models/rainbow_dqn/rainbow_dqn_final",
        "cls":     DQN,
        "kwargs":  {"custom_objects": {"policy_class": RainbowPolicy}},
    },
    "PPO": {
        "path":    "models/ppo/ppo_final",
        "cls":     PPO,
        "kwargs":  {},
    },
}


# ── Evaluation ───────────────────────────────────────────────────────────────────
def evaluate_model(model, env, n_episodes: int = 50) -> dict:
    """Run n_episodes and collect per-episode metrics."""
    rewards       = []
    collisions    = []
    speeds        = []
    episode_lens  = []
    lanes_used    = []

    for ep in range(n_episodes):
        obs, _       = env.reset()
        done         = False
        ep_reward    = 0.0
        ep_speeds    = []
        ep_lanes     = set()
        ep_len       = 0
        ep_collision = False

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done       = terminated or truncated

            ep_reward += reward
            ep_len    += 1

            # Collect speed from env vehicle
            try:
                speed = env.unwrapped.vehicle.speed
            except AttributeError:
                speed = 0.0
            ep_speeds.append(speed)

            # Track lane
            try:
                lane = env.unwrapped.vehicle.lane_index[2]
                ep_lanes.add(lane)
            except AttributeError:
                pass

            # Check collision
            if terminated and info.get("crashed", False):
                ep_collision = True

        rewards.append(ep_reward)
        collisions.append(ep_collision)
        speeds.append(float(np.mean(ep_speeds)) if ep_speeds else 0.0)
        episode_lens.append(ep_len)
        lanes_used.append(len(ep_lanes))

        print(f"  Episode {ep+1:3d}/{n_episodes} | "
              f"Reward: {ep_reward:7.3f} | "
              f"{'CRASH' if ep_collision else '  OK '} | "
              f"Speed: {speeds[-1]:5.1f}")

    success_rate   = 1.0 - np.mean(collisions)
    collision_rate = float(np.mean(collisions))

    return {
        "mean_reward":      float(np.mean(rewards)),
        "std_reward":       float(np.std(rewards)),
        "min_reward":       float(np.min(rewards)),
        "max_reward":       float(np.max(rewards)),
        "success_rate":     float(success_rate),
        "collision_rate":   float(collision_rate),
        "mean_speed":       float(np.mean(speeds)),
        "std_speed":        float(np.std(speeds)),
        "mean_ep_length":   float(np.mean(episode_lens)),
        "mean_lanes_used":  float(np.mean(lanes_used)),
        "episodes":         n_episodes,
        "raw_rewards":      [float(r) for r in rewards],
        "raw_collisions":   [bool(c) for c in collisions],
        "raw_speeds":       [float(s) for s in speeds],
    }


# ── Main ─────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    all_results = {}

    print("=" * 65)
    print("  Highway-Env RL Model Evaluation")
    print(f"  Episodes per model: {N_EVAL_EPISODES}")
    print("=" * 65)

    for name, config in MODELS.items():
        model_path = config["path"] + ".zip"
        if not os.path.exists(model_path):
            print(f"\n⚠️  [{name}] Model not found: {model_path} — skipping")
            continue

        print(f"\n{'─'*65}")
        print(f"  Evaluating: {name}")
        print(f"{'─'*65}")

        env = make_eval_env()

        try:
            model = config["cls"].load(config["path"], env=env, **config["kwargs"])
            model.set_env(env)

            results = evaluate_model(model, env, N_EVAL_EPISODES)
            all_results[name] = results

            print(f"\n  ┌─ {name} Summary ─────────────────────────────────────")
            print(f"  │  Mean Reward    : {results['mean_reward']:.4f} ± {results['std_reward']:.4f}")
            print(f"  │  Success Rate   : {results['success_rate']*100:.1f}%")
            print(f"  │  Collision Rate : {results['collision_rate']*100:.1f}%")
            print(f"  │  Mean Speed     : {results['mean_speed']:.2f}")
            print(f"  │  Mean Ep Length : {results['mean_ep_length']:.1f}")
            print(f"  └──────────────────────────────────────────────────────")

        except Exception as e:
            print(f"  ❌ Error evaluating {name}: {e}")

        finally:
            env.close()

    # ── Save results ──────────────────────────────────────────────────────────────
    output_path = os.path.join(RESULTS_DIR, "metrics.json")
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n✅ Results saved to {output_path}")

    # ── Print final comparison table ──────────────────────────────────────────────
    if all_results:
        print("\n" + "=" * 65)
        print(f"  {'Model':<15} {'Mean Reward':>12} {'Success%':>10} {'Speed':>8} {'CrashRate':>10}")
        print("  " + "─" * 60)
        for name, r in sorted(all_results.items(), key=lambda x: x[1]["mean_reward"], reverse=True):
            print(f"  {name:<15} {r['mean_reward']:>12.4f} {r['success_rate']*100:>9.1f}% "
                  f"{r['mean_speed']:>8.2f} {r['collision_rate']*100:>9.1f}%")
        print("=" * 65)
