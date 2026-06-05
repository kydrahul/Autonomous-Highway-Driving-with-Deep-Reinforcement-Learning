"""
A2C — Advantage Actor-Critic
====================================
Algorithm : A2C (Mnih et al. 2016)
Purpose   : Simplest on-policy baseline — tests if PPO's clipping is what matters
            or if ANY on-policy method beats off-policy DQN in dense traffic
Key Diff  : On-policy like PPO, but NO clipping, NO multiple epochs per batch

Why A2C matters for this paper:
  If A2C ≈ PPO >> DQN:  On-policy structure is the key (not PPO's clipping)
  If PPO >> A2C ≈ DQN:  PPO's clipping specifically helps in safety-critical envs
  If PPO > A2C > DQN:   On-policy helps, AND clipping helps further

Objective:
    L = E[ A_t · log π(a_t|s_t) ]   (vanilla policy gradient with advantage)
    No clipping, no ratio — just direct gradient ascent on expected return.

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
from stable_baselines3 import A2C
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor

# ── Paths ───────────────────────────────────────────────────────────────────────
MODEL_DIR = "models/a2c"
LOG_DIR   = "logs/a2c"
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

    ckpt_path, done_steps = find_latest_checkpoint(MODEL_DIR, "a2c")

    if ckpt_path and done_steps >= TOTAL_STEPS:
        print(f"✅ A2C already fully trained ({done_steps} steps). Nothing to do.")
        env.close()
        exit(0)
    elif ckpt_path:
        print(f"▶  Resuming A2C from {ckpt_path} ({done_steps:,} / {TOTAL_STEPS:,} steps done)")
        model = A2C.load(ckpt_path, env=env, device="auto")
        remaining   = TOTAL_STEPS - done_steps
        reset_steps = False
    else:
        print("▶  Starting A2C from scratch")
        model = A2C(
            policy="MlpPolicy",
            env=env,
            learning_rate=7e-4,
            n_steps=5,           # A2C uses shorter rollouts than PPO
            gamma=0.99,
            gae_lambda=0.95,
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
        name_prefix="a2c",
        verbose=1,
    )

    print("=" * 60)
    print("  Training: A2C (Advantage Actor-Critic)")
    print(f"  Device        : {model.device}")
    print(f"  Remaining steps: {remaining:,}")
    print(f"  Logs          : {LOG_DIR}")
    print(f"  Models        : {MODEL_DIR}")
    print("=" * 60)

    model.learn(
        total_timesteps=remaining,
        callback=checkpoint_cb,
        tb_log_name="A2C",
        reset_num_timesteps=reset_steps,
        progress_bar=True,
    )

    model.save(f"{MODEL_DIR}/a2c_final")
    print(f"\n✅ A2C training complete → {MODEL_DIR}/a2c_final.zip")
    env.close()
