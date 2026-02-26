"""
Dueling DQN
============
Algorithm : Dueling DQN (Wang et al. 2016)
Purpose   : Separate state-value V(s) from action-advantage A(s,a)
Key Change: Custom network splits into Value stream + Advantage stream

Architecture:
    Input (90,)   [15 vehicles × 6 features]
         ↓
    Shared Net    Linear(90→256) → ReLU → Linear(256→256) → ReLU
         ↓                    ↓
    Value Stream          Advantage Stream
    Linear(256→128)       Linear(256→128)
    ReLU                  ReLU
    Linear(128→1)         Linear(128→n_actions)
    V(s)                  A(s,a)
         ↓                    ↓
    Q(s,a) = V(s) + A(s,a) − mean(A(s,a))
                              ↑
                   Subtracting mean ensures identifiability

Network   : Custom Dueling MLP
Device    : CUDA (auto-detected)
Steps     : 500,000
"""

import os
import gymnasium as gym
import highway_env  # noqa: F401
import torch as th
import torch.nn as nn
from stable_baselines3 import DQN
from stable_baselines3.dqn.policies import DQNPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.dqn.policies import QNetwork
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from typing import List, Optional, Tuple, Type
import re
import glob

# ── Paths ───────────────────────────────────────────────────────────────────────
MODEL_DIR = "models/dueling_dqn"
LOG_DIR   = "logs/dueling_dqn"
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


# ── Dueling Q-Network ────────────────────────────────────────────────────────────
class DuelingQNetwork(QNetwork):
    """
    Replaces the standard Q-network with a Dueling architecture.
    Inherits from SB3's QNetwork and overrides the internal structure.

    Key insight: separating V(s) from A(s,a) allows the agent to learn
    state values without needing to evaluate every action — more efficient
    especially when most actions don't matter (e.g., driving straight).
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

        # ── Rebuild self.q_net as SHARED hidden layers only ─────────────────
        # Parent builds q_net as full MLP → action_dim.
        # We override it to output last hidden dim, then split.
        shared_layers = []
        in_dim = features_dim
        for size in net_arch:
            shared_layers.append(nn.Linear(in_dim, size))
            shared_layers.append(activation_fn())
            in_dim = size
        self.q_net = nn.Sequential(*shared_layers)   # ← override parent's q_net

        # ── Value stream: outputs a single scalar V(s) ──────────────────────
        self.value_stream = nn.Sequential(
            nn.Linear(in_dim, 128),
            activation_fn(),
            nn.Linear(128, 1),
        )

        # ── Advantage stream: outputs A(s,a) for every action ───────────────
        self.advantage_stream = nn.Sequential(
            nn.Linear(in_dim, 128),
            activation_fn(),
            nn.Linear(128, action_dim),
        )

    def forward(self, obs: th.Tensor) -> th.Tensor:
        """
        Dueling forward pass.
        Returns Q(s,a) = V(s) + A(s,a) − mean_a(A(s,a))
        """
        features  = self.extract_features(obs, self.features_extractor)
        shared    = self.q_net(features)
        value     = self.value_stream(shared)                   # (batch, 1)
        advantage = self.advantage_stream(shared)               # (batch, n_actions)
        # Combine: subtract mean for identifiability
        q_values  = value + advantage - advantage.mean(dim=1, keepdim=True)
        return q_values                                         # (batch, n_actions)


# ── Dueling DQN Policy ───────────────────────────────────────────────────────────
class DuelingDQNPolicy(DQNPolicy):
    """
    Custom DQN policy that creates DuelingQNetwork instead of standard QNetwork.
    Plug-and-play with SB3's DQN — only make_q_net() is overridden.
    """

    def make_q_net(self) -> DuelingQNetwork:
        net_args = self._update_features_extractor(self.net_args, features_extractor=None)
        return DuelingQNetwork(**net_args).to(self.device)


def make_env():
    env = gym.make("highway-v0", config=ENV_CONFIG)
    env = LeftLaneRewardWrapper(env)
    env = Monitor(env)
    return env


def find_latest_checkpoint(model_dir: str, prefix: str) -> Tuple[Optional[str], int]:
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

    ckpt_path, done_steps = find_latest_checkpoint(MODEL_DIR, "dueling_dqn")

    if ckpt_path and done_steps >= TOTAL_STEPS:
        print(f"✅ Dueling DQN already fully trained ({done_steps} steps). Nothing to do.")
        env.close()
        exit(0)
    elif ckpt_path:
        print(f"▶  Resuming Dueling DQN from {ckpt_path} ({done_steps:,} / {TOTAL_STEPS:,} steps done)")
        model = DQN.load(ckpt_path, env=env, device="auto",
                         custom_objects={"policy_class": DuelingDQNPolicy})
        remaining   = TOTAL_STEPS - done_steps
        reset_steps = False
    else:
        print("▶  Starting Dueling DQN from scratch")
        model = DQN(
            policy=DuelingDQNPolicy,
            env=env,
            learning_rate=5e-4,
            buffer_size=100_000,
            batch_size=64,
            gamma=0.99,
            tau=1.0,
            target_update_interval=1000,
            train_freq=4,
            gradient_steps=1,
            exploration_fraction=0.20,
            exploration_initial_eps=1.0,
            exploration_final_eps=0.05,
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
        name_prefix="dueling_dqn",
        save_replay_buffer=False,
        verbose=1,
    )

    print("=" * 60)
    print("  Training: Dueling DQN")
    print(f"  Device        : {model.device}")
    print(f"  Remaining steps: {remaining:,}")
    print(f"  Logs          : {LOG_DIR}")
    print(f"  Models        : {MODEL_DIR}")
    print("=" * 60)

    model.learn(
        total_timesteps=remaining,
        callback=checkpoint_cb,
        tb_log_name="DuelingDQN",
        reset_num_timesteps=reset_steps,
        progress_bar=True,
    )

    model.save(f"{MODEL_DIR}/dueling_dqn_final")
    print(f"\n✅ Dueling DQN training complete → {MODEL_DIR}/dueling_dqn_final.zip")
    env.close()
