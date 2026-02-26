"""
train_overtaker_quick.py  —  GPU-accelerated Aggressive Overtaker
=================================================================
Reward design:
  ✅ +bonus  for every LANE LEFT overtake
  ✅ +bonus  for maintaining speed > 34 m/s
  ✅ +bonus  for being faster than average nearby traffic
  ❌ -heavy  collision penalty
  ◻  No lane preference (right_lane_reward = 0)
  ◻  No speed ceiling — agent can go as fast as the env allows

Usage:
    .venv\\Scripts\\python.exe scripts\\training\\train_overtaker_quick.py
    .venv\\Scripts\\python.exe scripts\\training\\train_overtaker_quick.py --steps 30000
    .venv\\Scripts\\python.exe scripts\\training\\train_overtaker_quick.py --watch-only
"""

import argparse
import os
import time

import gymnasium as gym
import highway_env  # noqa: F401
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

# ── device ────────────────────────────────────────────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ── Env config ────────────────────────────────────────────────────────────────
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
    "vehicles_count": 30,
    "duration": 40,
    "initial_spacing": 2,
    # reward shaping via wrapper instead:
    "collision_reward": -3,       # base heavy penalty
    "right_lane_reward": 0.0,     # ← no lane preference
    "high_speed_reward": 0.4,
    "reward_speed_range": [34, 50],  # ← min 34 m/s, no real ceiling
    "normalize_reward": True,
    "simulation_frequency": 5,
    "policy_frequency": 1,
    "speed_limit": 60,            # ← no artificial cap; 60 is effectively unlimited
    "other_vehicles_type": "highway_env.vehicle.behavior.IDMVehicle",
    # render settings (used in watch mode)
    "screen_width":  1400,
    "screen_height": 250,
    "centering_position": [0.25, 0.5],
    "scaling": 5.5,
}

MODEL_SAVE = "models/overtaker/overtaker_quick"

ACTION_EMOJI = {
    0: "⬅️  overtaking!",
    1: "── holding lane",
    2: "➡️  moving right",
    3: "⬆️  accelerating",
    4: "⬇️  braking",
}

MIN_SPEED = 34.0   # below this the agent gets no speed bonus


# ── Custom Reward Wrapper ─────────────────────────────────────────────────────
class OvertakerRewardWrapper(gym.Wrapper):
    """
    Stacks on top of the base env reward:
      + 0.4  per LANE-LEFT action while faster than the average nearby car
      + 0.3  for every step the ego is going > MIN_SPEED (34 m/s)
      + 0.05 × (ego_speed - avg_nearby_speed)  continuous relative-speed bonus
      - 2.0  extra collision penalty (base env already gives -3)
    """

    def step(self, action):
        obs, base_reward, terminated, truncated, info = self.env.step(action)
        action = int(action)

        ego  = self.unwrapped.vehicle
        road = self.unwrapped.road

        # Nearby vehicles within 80 m
        try:
            nearby = [v for v in road.vehicles
                      if v is not ego and abs(v.position[0] - ego.position[0]) < 80]
            avg_nearby = float(np.mean([v.speed for v in nearby])) if nearby else ego.speed
        except Exception:
            avg_nearby = ego.speed

        # 1. Overtake bonus
        overtake_bonus = 0.4 if (action == 0 and ego.speed >= avg_nearby) else 0.0

        # 2. Min-speed bonus (reward going fast)
        speed_bonus = 0.3 if ego.speed >= MIN_SPEED else 0.0

        # 3. Relative speed bonus
        rel_bonus = max(0.0, ego.speed - avg_nearby) * 0.05

        # 4. Extra collision penalty
        crash_extra = -2.0 if info.get("crashed", False) else 0.0

        reward = base_reward + overtake_bonus + speed_bonus + rel_bonus + crash_extra
        return obs, reward, terminated, truncated, info


# ── Factory ───────────────────────────────────────────────────────────────────
def _make(render=False):
    mode = "human" if render else None
    cfg  = dict(ENV_CONFIG)
    e = gym.make("highway-v0", render_mode=mode, config=cfg)
    return OvertakerRewardWrapper(e)


# ── Training ──────────────────────────────────────────────────────────────────
def train(steps: int):
    os.makedirs("models/overtaker", exist_ok=True)

    print(f"\n🖥️  Device : {DEVICE.upper()}")
    print(f"🏋️  Training Aggressive Overtaker  —  {steps:,} steps\n")

    # Use 4 parallel envs on GPU run, 1 on CPU
    n_envs = 4 if DEVICE == "cuda" else 1
    vec_env = DummyVecEnv([lambda: _make() for _ in range(n_envs)])

    model = PPO(
        "MlpPolicy",
        vec_env,
        device=DEVICE,
        learning_rate=3e-4,
        n_steps=512,
        batch_size=128,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.03,   # higher entropy → more exploration → more lane changes
        verbose=1,
    )

    t0 = time.time()
    model.learn(total_timesteps=steps)
    elapsed = time.time() - t0

    model.save(MODEL_SAVE)
    print(f"\n✅  Done in {elapsed:.0f}s  —  saved to {MODEL_SAVE}.zip\n")
    return model


# ── Watch ─────────────────────────────────────────────────────────────────────
def watch(model, episodes=10, fps=5.0):
    frame_time = 1.0 / fps

    env = _make(render=True)

    print("\n╔══════════════════════════════════════════════════════╗")
    print("║  🚗  Aggressive Overtaker — Live Demo                ║")
    print("║  Min speed: 34 m/s | No lane bias | Avoid crashes   ║")
    print("╚══════════════════════════════════════════════════════╝\n")

    total_safe      = 0
    total_overtakes = 0

    for ep in range(1, episodes + 1):
        obs, _ = env.reset()

        # Slow down traffic to ~22 m/s so the agent (targeting 34+) MUST overtake
        try:
            ego = env.unwrapped.vehicle
            for v in env.unwrapped.road.vehicles:
                if v is not ego:
                    v.target_speed = 22.0
                    v.speed        = min(v.speed, 23.0)
        except Exception:
            pass

        done        = False
        ep_reward   = 0.0
        ep_steps    = 0
        collision   = False
        ep_overtakes = 0

        while not done:
            t0 = time.perf_counter()

            action, _ = model.predict(obs, deterministic=False)
            obs, reward, terminated, truncated, info = env.step(action)
            env.render()

            ep_reward += reward
            ep_steps  += 1
            done       = terminated or truncated
            action     = int(action)

            if action == 0:
                ep_overtakes += 1
            if info.get("crashed", False):
                collision = True

            try:
                lane = env.unwrapped.vehicle.lane_index[2]
                spd  = env.unwrapped.vehicle.speed
            except Exception:
                lane, spd = "?", 0

            status = "💥 CRASHED" if collision else "✅ Safe"
            emoji  = ACTION_EMOJI.get(action, "?")
            print(
                f"\r  Ep {ep:2d} | Step {ep_steps:3d} | {emoji:<25} | "
                f"Lane {lane} | {spd:5.1f} m/s | {status}   ",
                end="", flush=True,
            )

            elapsed = time.perf_counter() - t0
            slp = frame_time - elapsed
            if slp > 0:
                time.sleep(slp)

        print()
        status = "💥 CRASHED" if collision else "✅ Safe"
        print(f"  └─ Ep {ep:2d} | Reward: {ep_reward:.1f} | "
              f"Steps: {ep_steps} | Overtakes: {ep_overtakes} | {status}\n")

        total_overtakes += ep_overtakes
        if not collision:
            total_safe += 1

    env.close()
    print("═" * 60)
    print(f"  📊 Final Summary ({episodes} episodes)")
    print(f"  Safe runs      : {total_safe} / {episodes}  ({100*total_safe//episodes}%)")
    print(f"  Total overtakes: {total_overtakes}  (avg {total_overtakes/episodes:.1f}/ep)")
    print("═" * 60)


# ── Entry ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps",      type=int,   default=20000)
    parser.add_argument("--episodes",   type=int,   default=10)
    parser.add_argument("--fps",        type=float, default=5.0)
    parser.add_argument("--watch-only", action="store_true",
                        help="Skip training, load existing model")
    args = parser.parse_args()

    if args.watch_only:
        if not os.path.exists(MODEL_SAVE + ".zip"):
            print(f"❌  No model at {MODEL_SAVE}.zip — run without --watch-only first.")
        else:
            m = PPO.load(MODEL_SAVE, device=DEVICE)
            watch(m, args.episodes, args.fps)
    else:
        m = train(args.steps)
        watch(m, args.episodes, args.fps)
