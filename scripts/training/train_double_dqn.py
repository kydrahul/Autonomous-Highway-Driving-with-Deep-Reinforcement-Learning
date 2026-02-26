"""
Double DQN
===========
Algorithm : Double DQN (van Hasselt et al. 2016)
Purpose   : Fix Q-value overestimation of basic DQN
Key Change: Online network SELECTS action,
            Target network EVALUATES it.
Network   : MLP [256, 256]
Device    : CUDA (auto-detected)
Steps     : 500,000

Standard DQN target:
    y = r + γ · max_a  Q_target(s', a)          ← same net selects & evaluates → overestimates

Double DQN target:
    y = r + γ · Q_target(s', argmax_a Q_online(s', a))  ← online selects, target evaluates
"""

import os
import re
import glob
import numpy as np
from typing import Optional, Tuple
import gymnasium as gym
import highway_env  # noqa: F401
import torch as th
import torch.nn.functional as F
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor

# ── Paths ───────────────────────────────────────────────────────────────────────
MODEL_DIR = "models/double_dqn"
LOG_DIR   = "logs/double_dqn"
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


# ── Double DQN — Override train() to use double Q-value target ──────────────────
class DoubleDQN(DQN):
    """
    Double DQN implementation.

    Only difference from DQN: next action is selected by the ONLINE network
    but evaluated by the TARGET network, reducing overestimation.
    """

    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)

        losses = []
        for _ in range(gradient_steps):
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)

            with th.no_grad():
                # ── Double DQN Target ───────────────────────────────────────
                # Step 1: Online network selects the best next action
                next_actions = self.q_net(replay_data.next_observations).argmax(dim=1, keepdim=True)
                # Step 2: Target network evaluates that action's Q-value
                next_q_values = th.gather(
                    self.q_net_target(replay_data.next_observations),
                    dim=1,
                    index=next_actions,
                )
                target_q_values = (
                    replay_data.rewards
                    + (1 - replay_data.dones) * self.gamma * next_q_values
                )

            # Current Q estimates for taken actions
            current_q_values = th.gather(
                self.q_net(replay_data.observations),
                dim=1,
                index=replay_data.actions.long(),
            )

            loss = F.smooth_l1_loss(current_q_values, target_q_values)

            self.policy.optimizer.zero_grad()
            loss.backward()
            th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.policy.optimizer.step()

            losses.append(loss.item())

        self._n_updates += gradient_steps
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/loss", float(np.mean(losses)))


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

    ckpt_path, done_steps = find_latest_checkpoint(MODEL_DIR, "double_dqn")

    if ckpt_path and done_steps >= TOTAL_STEPS:
        print(f"✅ Double DQN already fully trained ({done_steps} steps). Nothing to do.")
        env.close()
        exit(0)
    elif ckpt_path:
        print(f"▶  Resuming Double DQN from {ckpt_path} ({done_steps:,} / {TOTAL_STEPS:,} steps done)")
        model = DoubleDQN.load(ckpt_path, env=env, device="auto")
        remaining   = TOTAL_STEPS - done_steps
        reset_steps = False
    else:
        print("▶  Starting Double DQN from scratch")
        model = DoubleDQN(
            policy="MlpPolicy",
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
        name_prefix="double_dqn",
        save_replay_buffer=False,
        verbose=1,
    )

    print("=" * 60)
    print("  Training: Double DQN")
    print(f"  Device        : {model.device}")
    print(f"  Remaining steps: {remaining:,}")
    print(f"  Logs          : {LOG_DIR}")
    print(f"  Models        : {MODEL_DIR}")
    print("=" * 60)

    model.learn(
        total_timesteps=remaining,
        callback=checkpoint_cb,
        tb_log_name="DoubleDQN",
        reset_num_timesteps=reset_steps,
        progress_bar=True,
    )

    model.save(f"{MODEL_DIR}/double_dqn_final")
    print(f"\n✅ Double DQN training complete → {MODEL_DIR}/double_dqn_final.zip")
    env.close()
