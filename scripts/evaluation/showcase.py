"""
showcase.py — Highlight Reel Demo (v2)

showcase.py — Highlight Reel Demo
===================================
Runs PPO in a tuned environment designed to SHOW OFF:
  1. Smooth, safe cruising in the left lane
  2. Visible overtaking manoeuvres

Tricks used to make it look great on camera:
  - Fewer vehicles (20) → more gaps → agent can overtake instead of braking
  - initial_spacing = 1  → vehicles start bunched → more overtaking opportunities
  - scaling = 5.5        → wider field of view → you see more road
  - centering_position   → ego car is left-centred so you see what's ahead
  - FPS 5 (default)      → slow enough to follow every lane change
  - Prints action live   → easy to add text overlay while recording

Usage:
    .venv\\Scripts\\python.exe scripts\\evaluation\\showcase.py
    .venv\\Scripts\\python.exe scripts\\evaluation\\showcase.py --fps 3 --episodes 5
    .venv\\Scripts\\python.exe scripts\\evaluation\\showcase.py --model ppo       (default)
    .venv\\Scripts\\python.exe scripts\\evaluation\\showcase.py --model rainbow
"""

import argparse
import random
import os
import sys
import time

import gymnasium as gym
import highway_env  # noqa: F401

from stable_baselines3 import DQN, PPO

# ── Model registry ────────────────────────────────────────────────────────────
MODELS = {
    "ppo":         ("PPO",        "models/ppo/ppo_final",             PPO),
    "rainbow":     ("Rainbow DQN","models/rainbow_dqn/rainbow_final", DQN),
    "dueling":     ("Dueling DQN","models/dueling_dqn/dueling_final", DQN),
    "double":      ("Double DQN", "models/double_dqn/double_final",   DQN),
    "dqn":         ("DQN",        "models/dqn/dqn_final",             DQN),
}

ACTION_NAMES = {
    0: "◀  LANE LEFT",
    1: "── IDLE",
    2: "LANE RIGHT ▶",
    3: "▲  FASTER",
    4: "▼  SLOWER",
}

ACTION_EMOJI = {
    0: "⬅️  overtaking!",
    1: "➡️  holding lane",
    2: "➡️  moving right",
    3: "⬆️  accelerating",
    4: "⬇️  braking",
}

# ── Showcase environment config ────────────────────────────────────────────────
SHOWCASE_CONFIG = {
    "observation": {
        "type": "Kinematics",
        "vehicles_count": 15,
        "features": ["x", "y", "vx", "vy", "cos_h", "sin_h"],
        "normalize": True,
        "absolute": False,
    },
    "action": {"type": "DiscreteMetaAction"},
    "lanes_count": 4,
    "vehicles_count": 30,          # ← enough to create overtaking but not gridlock
    "duration": 40,
    "initial_spacing": 2,          # ← spaced out like DQN training → no gridlock
    "collision_reward": -1,
    "right_lane_reward": 0.1,      # ← small right-lane nudge, no left-lane bias
    "high_speed_reward": 0.4,
    "reward_speed_range": [35, 40],  # ← agent targets 40 m/s
    "normalize_reward": True,
    "simulation_frequency": 5,
    "policy_frequency": 1,
    "speed_limit": 40,               # ← allow 40 m/s (default cap is 30)
    "other_vehicles_type": "highway_env.vehicle.behavior.IDMVehicle",
    "screen_width":  1400,
    "screen_height": 250,
    "centering_position": [0.25, 0.5],
    "scaling": 5.5,
}


# LeftLaneRewardWrapper removed — it was biasing the agent into lane 0 and causing crashes


# ── Helpers ───────────────────────────────────────────────────────────────────
def bar(value, total, width=30):
    filled = int(round(width * value / total))
    return "█" * filled + "░" * (width - filled)


def print_live(ep, step, action, reward, lane, crashed):
    action_str = ACTION_EMOJI.get(action, "?")
    lane_str   = f"Lane {lane}"
    status     = "💥 CRASHED" if crashed else "✅ Safe"
    print(
        f"\r  Ep {ep:2d} | Step {step:3d} | {action_str:<25} | "
        f"{lane_str} | R={reward:5.1f} | {status}   ",
        end="", flush=True
    )


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Highway RL Showcase")
    parser.add_argument("--model",    default="ppo",
                        choices=list(MODELS.keys()),
                        help="Which model to watch (default: ppo)")
    parser.add_argument("--episodes", type=int,   default=8,
                        help="Episodes to run (default: 8)")
    parser.add_argument("--fps",      type=float, default=5.0,
                        help="Render FPS — lower = slower (default: 5)")
    args = parser.parse_args()

    label, model_path, ModelClass = MODELS[args.model]
    frame_time = 1.0 / args.fps

    # Check model exists
    if not os.path.exists(model_path + ".zip"):
        # Try alternative path patterns
        alt = model_path.replace("_final", "_checkpoint_300000_steps")
        if not os.path.exists(alt + ".zip"):
            # Scan models folder for any matching file
            base = model_path.split("/")[1]  # e.g. "ppo"
            model_dir = f"models/{base}"
            if os.path.isdir(model_dir):
                zips = sorted([f for f in os.listdir(model_dir) if f.endswith(".zip")])
                if zips:
                    model_path = os.path.join(model_dir, zips[-1][:-4])
                    print(f"  Using latest checkpoint: {zips[-1]}")
                else:
                    print(f"❌  No model found in {model_dir}/")
                    sys.exit(1)
            else:
                print(f"❌  Model not found: {model_path}.zip")
                print(f"    Available model folders: {os.listdir('models')}")
                sys.exit(1)

    env = gym.make("highway-v0", render_mode="human", config=SHOWCASE_CONFIG)
    # No wrapper — agent drives with its original trained policy, no lane bias

    model = ModelClass.load(model_path, env=env)
    model.policy.set_training_mode(False)

    print()
    print("╔══════════════════════════════════════════════════════╗")
    print(f"║  🚗  Highway RL Showcase — {label:<25} ║")
    print(f"║  Episodes : {args.episodes:<5}  FPS : {args.fps:<6}  Model : {args.model:<10}  ║")
    print("╠══════════════════════════════════════════════════════╣")
    print("║  Watch for:                                          ║")
    print("║  ⬅️  Lane Left  = overtaking manoeuvre               ║")
    print("║  ⬆️  FASTER     = accelerating past slow traffic     ║")
    print("║  ✅ reaching step 40 = full episode survived         ║")
    print("╚══════════════════════════════════════════════════════╝")
    print()

    total_reward = 0.0
    total_safe   = 0
    overtakes    = 0

    for ep in range(1, args.episodes + 1):
        obs, _ = env.reset()

        # ── Force slow traffic so the agent MUST overtake ────────────────────
        try:
            ego = env.unwrapped.vehicle
            for v in env.unwrapped.road.vehicles:
                if v is not ego:
                    traffic_spd = random.uniform(20.0, 22.0)  # slower traffic → safer overtakes
                    v.target_speed = traffic_spd
                    v.speed = min(v.speed, traffic_spd)
        except Exception:
            pass  # silently skip if vehicle API differs
        # ─────────────────────────────────────────────────────────────────────

        done        = False
        ep_reward   = 0.0
        ep_steps    = 0
        collision   = False
        ep_overtakes = 0
        prev_lane   = None

        while not done:
            t0 = time.perf_counter()

            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            env.render()

            ep_reward += reward
            ep_steps  += 1
            done       = terminated or truncated
            action     = int(action)

            # Count overtakes (every LANE LEFT action)
            if action == 0:
                ep_overtakes += 1

            if info.get("crashed", False):
                collision = True

            # Current lane
            try:
                lane = env.unwrapped.vehicle.lane_index[2]
            except Exception:
                lane = "?"

            print_live(ep, ep_steps, action, ep_reward, lane, collision)

            elapsed = time.perf_counter() - t0
            sleep = frame_time - elapsed
            if sleep > 0:
                time.sleep(sleep)

        print()  # newline after live line

        status = "💥 CRASHED" if collision else "✅ Safe"
        print(f"  └─ Episode {ep:2d} done | Reward: {ep_reward:.1f} | "
              f"Steps: {ep_steps} | Overtakes: {ep_overtakes} | {status}")
        print()

        total_reward += ep_reward
        overtakes    += ep_overtakes
        if not collision:
            total_safe += 1

    env.close()

    print("═" * 56)
    print(f"  📊 Final Summary ({args.episodes} episodes)")
    print(f"  Mean reward   : {total_reward / args.episodes:.2f}")
    print(f"  Safe runs     : {total_safe} / {args.episodes}  "
          f"({100*total_safe//args.episodes}%)")
    print(f"  Total overtakes detected : {overtakes}")
    print("═" * 56)


if __name__ == "__main__":
    main()
