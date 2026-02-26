"""
PPO — Live Visualisation
=========================
Loads the trained PPO model and renders it in a pygame window.
Slowed down with configurable FPS so you can follow the agent easily.

PPO (Proximal Policy Optimisation) is an on-policy actor-critic method — unlike
the DQN family it does not use a replay buffer and learns a stochastic policy
that is then clipped to prevent too-large updates.

Usage:
    python scripts/evaluation/watch_ppo.py
    python scripts/evaluation/watch_ppo.py --episodes 5
    python scripts/evaluation/watch_ppo.py --fps 3       # even slower
"""

import argparse
import os
import time

import gymnasium as gym
import highway_env  # noqa: F401

from stable_baselines3 import PPO

# ── Environment Config ────────────────────────────────────────────────────────────
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
    "screen_width":  1200,
    "screen_height":  200,
    "centering_position": [0.3, 0.5],
    "scaling": 7.0,
}

MODEL_PATH = "models/ppo/ppo_final"

ACTION_NAMES = {0: "LANE LEFT", 1: "IDLE", 2: "LANE RIGHT", 3: "FASTER", 4: "SLOWER"}


# ── Left Lane Wrapper ─────────────────────────────────────────────────────────────
class LeftLaneRewardWrapper(gym.Wrapper):
    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)
        current_lane = self.unwrapped.vehicle.lane_index[2]
        total_lanes  = self.unwrapped.config["lanes_count"]
        reward      += 0.1 * (total_lanes - 1 - current_lane) / (total_lanes - 1)
        return obs, reward, done, truncated, info


# ── Main ──────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Watch PPO drive")
    parser.add_argument("--episodes", type=int, default=10,
                        help="Number of episodes to watch (default: 10)")
    parser.add_argument("--fps", type=float, default=6.0,
                        help="Render FPS — lower = slower (default: 6)")
    args = parser.parse_args()

    frame_time = 1.0 / args.fps

    if not os.path.exists(MODEL_PATH + ".zip"):
        print(f"Model not found: {MODEL_PATH}.zip")
        return

    env = gym.make("highway-v0", render_mode="human", config=ENV_CONFIG)
    env = LeftLaneRewardWrapper(env)

    model = PPO.load(MODEL_PATH, env=env)
    model.policy.set_training_mode(False)

    print("=" * 55)
    print("  PPO — Live Demo")
    print(f"  Episodes : {args.episodes}")
    print(f"  FPS      : {args.fps}  (slower = easier to watch)")
    print(f"  Screen   : {ENV_CONFIG['screen_width']}×{ENV_CONFIG['screen_height']}")
    print("=" * 55)
    print()

    total_reward = 0.0
    total_safe   = 0

    for ep in range(1, args.episodes + 1):
        obs, _ = env.reset()
        done      = False
        ep_reward = 0.0
        ep_steps  = 0
        collision = False
        last_action_name = "─"

        print(f"  Episode {ep} starting…")

        while not done:
            t_start = time.perf_counter()

            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            env.render()

            ep_reward        += reward
            ep_steps         += 1
            done              = terminated or truncated
            last_action_name  = ACTION_NAMES.get(int(action), str(action))

            if info.get("crashed", False):
                collision = True

            elapsed = time.perf_counter() - t_start
            sleep   = frame_time - elapsed
            if sleep > 0:
                time.sleep(sleep)

        status = "CRASHED" if collision else "Safe"
        print(f"  Episode {ep:2d}: reward={ep_reward:6.2f}  "
              f"steps={ep_steps:3d}  last_action={last_action_name:<11}  {status}")

        total_reward += ep_reward
        if not collision:
            total_safe += 1

    env.close()

    print()
    print("─" * 55)
    print(f"  Summary over {args.episodes} episodes:")
    print(f"    Mean reward : {total_reward / args.episodes:.2f}")
    print(f"    Safe runs   : {total_safe} / {args.episodes}")
    print("─" * 55)


if __name__ == "__main__":
    main()
