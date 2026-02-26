"""
Basic DQN — Baseline Model
===========================
Algorithm : DQN (Mnih et al. 2015)
Purpose   : Establish baseline performance for comparison
Network   : MLP [256, 256]
Device    : CUDA (auto-detected)
Steps     : 500,000
"""

import os
import re
import glob
from typing import Optional, Tuple
import gymnasium as gym
import highway_env  # noqa: F401
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor

# ── Paths ───────────────────────────────────────────────────────────────────────
MODEL_DIR = "models/dqn"
LOG_DIR   = "logs/dqn"
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
    """
    Adds a bonus reward for staying in the left-most lane.
    Overrides highway-env's default right-lane preference.

    Lane reward:
        lane 0 (leftmost)  → +0.10
        lane 1             → +0.067
        lane 2             → +0.033
        lane 3 (rightmost) → +0.00
    """
    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)
        current_lane = self.unwrapped.vehicle.lane_index[2]
        total_lanes  = self.unwrapped.config["lanes_count"]
        left_reward  = (total_lanes - 1 - current_lane) / (total_lanes - 1)
        reward      += 0.1 * left_reward
        return obs, reward, done, truncated, info


def make_env():
    env = gym.make("highway-v0", config=ENV_CONFIG)
    env = LeftLaneRewardWrapper(env)
    env = Monitor(env)
    return env


def find_latest_checkpoint(model_dir: str, prefix: str) -> Tuple[Optional[str], int]:
    """Return (path, steps) of the most advanced <prefix>_N_steps.zip, or (None, 0)."""
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

    ckpt_path, done_steps = find_latest_checkpoint(MODEL_DIR, "dqn")

    if ckpt_path and done_steps >= TOTAL_STEPS:
        print(f"✅ DQN already fully trained ({done_steps} steps). Nothing to do.")
        env.close()
        exit(0)
    elif ckpt_path:
        print(f"▶  Resuming DQN from {ckpt_path} ({done_steps:,} / {TOTAL_STEPS:,} steps done)")
        model = DQN.load(ckpt_path, env=env, device="auto")
        remaining   = TOTAL_STEPS - done_steps
        reset_steps = False
    else:
        print("▶  Starting DQN from scratch")
        model = DQN(
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
        name_prefix="dqn",
        save_replay_buffer=False,
        verbose=1,
    )

    print("=" * 60)
    print("  Training: Basic DQN (Baseline)")
    print(f"  Device        : {model.device}")
    print(f"  Remaining steps: {remaining:,}")
    print(f"  Logs          : {LOG_DIR}")
    print(f"  Models        : {MODEL_DIR}")
    print("=" * 60)

    model.learn(
        total_timesteps=remaining,
        callback=checkpoint_cb,
        tb_log_name="DQN",
        reset_num_timesteps=reset_steps,
        progress_bar=True,
    )

    model.save(f"{MODEL_DIR}/dqn_final")
    print(f"\n✅ DQN training complete → {MODEL_DIR}/dqn_final.zip")
    env.close()
