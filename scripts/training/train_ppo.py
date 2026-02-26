"""
PPO — Proximal Policy Optimization
====================================
Algorithm : PPO (Schulman et al. 2017)
Purpose   : Policy gradient comparison vs. value-based DQN family
Key Diff  : On-policy, no replay buffer, directly optimises policy

Objective:
    L = E[ min(r_t · A_t,  clip(r_t, 1−ε, 1+ε) · A_t) ]
    where r_t = π(a|s) / π_old(a|s)  (probability ratio)
          A_t = GAE advantage estimate

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
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor

# ── Paths ───────────────────────────────────────────────────────────────────────
MODEL_DIR = "models/ppo"
LOG_DIR   = "logs/ppo"
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

    ckpt_path, done_steps = find_latest_checkpoint(MODEL_DIR, "ppo")

    if ckpt_path and done_steps >= TOTAL_STEPS:
        print(f"✅ PPO already fully trained ({done_steps} steps). Nothing to do.")
        env.close()
        exit(0)
    elif ckpt_path:
        print(f"▶  Resuming PPO from {ckpt_path} ({done_steps:,} / {TOTAL_STEPS:,} steps done)")
        model = PPO.load(ckpt_path, env=env, device="auto")
        remaining   = TOTAL_STEPS - done_steps
        reset_steps = False
    else:
        print("▶  Starting PPO from scratch")
        model = PPO(
            policy="MlpPolicy",
            env=env,
            learning_rate=3e-4,
            n_steps=512,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            vf_coef=0.5,
            max_grad_norm=0.5,
            policy_kwargs=dict(
                net_arch=dict(pi=[256, 256], vf=[256, 256]),
            ),
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
        name_prefix="ppo",
        verbose=1,
    )

    print("=" * 60)
    print("  Training: PPO (Policy Gradient)")
    print(f"  Device        : {model.device}")
    print(f"  Remaining steps: {remaining:,}")
    print(f"  Logs          : {LOG_DIR}")
    print(f"  Models        : {MODEL_DIR}")
    print("=" * 60)

    model.learn(
        total_timesteps=remaining,
        callback=checkpoint_cb,
        tb_log_name="PPO",
        reset_num_timesteps=reset_steps,
        progress_bar=True,
    )

    model.save(f"{MODEL_DIR}/ppo_final")
    print(f"\n✅ PPO training complete → {MODEL_DIR}/ppo_final.zip")
    env.close()
