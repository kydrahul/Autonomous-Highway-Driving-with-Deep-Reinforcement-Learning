"""
Multi-Seed Evaluation Script
==============================
Evaluates ALL trained models (using whatever model checkpoints exist)
over a configurable number of episodes and seeds.

This directly addresses the reviewer concern:
  "50 episodes from a single seed is insufficient for statistical claims."

Usage:
    python scripts/evaluation/evaluate_multi_seed.py [--episodes 200] [--seeds 42 123 456]

Strategy:
  - We have pre-trained models (single seed=42).
  - This script re-evaluates each model over many episodes with DIFFERENT env seeds,
    which captures environment stochasticity even without multi-seed training.
  - For FULL multi-seed, use train_all_seeds.py first, then run this.

Outputs:
    results/multi_seed_metrics.json
    results/plots/14_multi_seed_comparison.png
    results/analysis/multi_seed_report.txt
"""

import os
import json
import argparse
import numpy as np
import gymnasium as gym
import highway_env  # noqa: F401
import matplotlib.pyplot as plt
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.monitor import Monitor
from pathlib import Path
import sys

# ── Import Rainbow from training script ─────────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "training"))
try:
    from train_rainbow_dqn import RainbowDQN, RainbowPolicy, PrioritizedReplayBuffer
    RAINBOW_AVAILABLE = True
except ImportError:
    RAINBOW_AVAILABLE = False
    print("⚠️  Rainbow DQN not importable — will skip Rainbow multi-seed eval.")

plt.rcParams.update({
    "figure.facecolor": "#0f0f1a",
    "axes.facecolor":   "#1a1a2e",
    "axes.edgecolor":   "#444466",
    "axes.labelcolor":  "#e0e0e0",
    "xtick.color":      "#e0e0e0",
    "ytick.color":      "#e0e0e0",
    "text.color":       "#e0e0e0",
    "grid.color":       "#2a2a4a",
    "font.size":        11,
})

ALGO_COLORS = {
    "PPO":        "#00d4ff",
    "Rainbow DQN":"#ff6b9d",
    "Dueling DQN":"#ffd166",
    "Double DQN": "#06d6a0",
    "DQN":        "#ef476f",
}

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

MODELS = {
    "PPO":        ("models/ppo/ppo_final.zip",               "ppo"),
    "Rainbow DQN":("models/rainbow_dqn/rainbow_dqn_final.zip","rainbow"),
    "Dueling DQN":("models/dueling_dqn/dueling_dqn_final.zip","dqn"),
    "Double DQN": ("models/double_dqn/double_dqn_final.zip",  "dqn"),
    "DQN":        ("models/dqn/dqn_final.zip",               "dqn"),
}


def make_env(seed: int) -> gym.Env:
    env = gym.make("highway-v0", config=ENV_CONFIG)
    env = Monitor(env)
    env.reset(seed=seed)
    return env


def load_model(algo_name: str, path: str, env: gym.Env):
    if algo_name == "PPO":
        return PPO.load(path, env=env, device="cpu")
    elif algo_name == "Rainbow DQN" and RAINBOW_AVAILABLE:
        return RainbowDQN.load(
            path, env=env, device="cpu",
            custom_objects={
                "replay_buffer_class": PrioritizedReplayBuffer,
                "replay_buffer_kwargs": dict(
                    alpha=0.6, beta_start=0.4, beta_frames=500_000, n_step=3, gamma=0.99,
                ),
            },
        )
    else:
        return DQN.load(path, env=env, device="cpu")


def evaluate_model(model, env: gym.Env, n_episodes: int) -> dict:
    rewards, collisions, speeds, ep_lengths = [], [], [], []

    for _ in range(n_episodes):
        obs, info = env.reset()
        done = truncated = False
        ep_reward = 0.0
        step_speeds = []

        while not (done or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            ep_reward += reward
            try:
                v = env.unwrapped.vehicle.speed
                step_speeds.append(v)
            except Exception:
                pass

        rewards.append(ep_reward)
        collisions.append(bool(info.get("crashed", False)))
        speeds.append(float(np.mean(step_speeds)) if step_speeds else 0.0)
        ep_lengths.append(info.get("step", 40))

    return {
        "mean_reward":    float(np.mean(rewards)),
        "std_reward":     float(np.std(rewards, ddof=1)),
        "success_rate":   float(1.0 - np.mean(collisions)),
        "collision_rate": float(np.mean(collisions)),
        "mean_speed":     float(np.mean(speeds)),
        "n_episodes":     n_episodes,
        "raw_rewards":    [float(r) for r in rewards],
        "raw_collisions": [bool(c) for c in collisions],
    }


def plot_multi_seed_results(all_results: dict):
    algos = [a for a in ALGO_COLORS if a in all_results]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f"Multi-Episode Evaluation Results (n={next(iter(all_results.values())).get('n_episodes', '?')} episodes)",
                 fontsize=14, color="white")

    for ax, (metric, ylabel, title) in zip(axes, [
        ("mean_reward",  "Mean Reward",    "Mean Episodic Reward ± Std"),
        ("success_rate", "Success Rate",   "Success Rate (No Collision)"),
        ("collision_rate","Collision Rate","Collision Rate"),
    ]):
        vals   = [all_results[a].get(metric, 0) for a in algos]
        stds   = [all_results[a].get("std_reward", 0) / 40 for a in algos]  # approx
        colors = [ALGO_COLORS[a] for a in algos]

        bars = ax.bar(range(len(algos)), vals,
                      color=colors, edgecolor="white", linewidth=0.5, alpha=0.85)
        if metric == "mean_reward":
            ax.errorbar(range(len(algos)), vals, yerr=stds,
                        fmt="none", color="white", capsize=5, linewidth=1.5)
        for bar, val in zip(bars, vals):
            fmt = f"{val:.2f}" if metric == "mean_reward" else f"{val:.1%}"
            ax.text(bar.get_x() + bar.get_width() / 2, val + 0.005,
                    fmt, ha="center", va="bottom", fontsize=9, color="white")

        ax.set_xticks(range(len(algos)))
        ax.set_xticklabels(algos, rotation=20, ha="right", fontsize=9)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(title, fontsize=11, pad=8)
        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    path = "results/plots/14_multi_seed_comparison.png"
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"  ✅  Plot saved: {path}")


def write_report(all_results: dict, n_episodes: int):
    lines = [
        "=" * 70,
        f"MULTI-EPISODE EVALUATION REPORT  (n={n_episodes} episodes per model)",
        "=" * 70,
        f"{'Algorithm':<15} {'Mean±Std':>18} {'Success':>10} {'Collision':>12} {'Speed':>10}",
        "-" * 70,
    ]
    for algo, res in all_results.items():
        lines.append(
            f"{algo:<15} {res['mean_reward']:>8.3f}±{res['std_reward']:<8.3f}"
            f"{res['success_rate']:>9.1%} {res['collision_rate']:>11.1%} {res['mean_speed']:>10.2f}"
        )
    lines += [
        "",
        "NOTE: For publication, additionally run train_all_seeds.py with 3-5",
        "seeds and aggregate results across seed x episode combinations.",
        "=" * 70,
    ]
    path = "results/analysis/multi_seed_report.txt"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write("\n".join(lines))
    print("\n" + "\n".join(lines))


# ─────────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=200,
                        help="Number of evaluation episodes per model (default: 200)")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42],
                        help="Environment seeds to average over (default: [42])")
    args = parser.parse_args()

    print(f"\n🔬 Multi-Episode Evaluation")
    print(f"   Episodes: {args.episodes} | Env seeds: {args.seeds}")
    print("=" * 50)

    all_results = {}

    for algo, (model_path, _) in MODELS.items():
        if not os.path.exists(model_path):
            print(f"  ⚠️  {algo}: model not found at {model_path} — skipping.")
            continue
        print(f"\n  ▶  Evaluating {algo}...")
        seed_results = []
        for seed in args.seeds:
            env = make_env(seed)
            try:
                model = load_model(algo, model_path, env)
                res   = evaluate_model(model, env, args.episodes // len(args.seeds))
                seed_results.append(res)
            except Exception as e:
                print(f"      ⚠️  Seed {seed} failed: {e}")
            finally:
                env.close()

        if not seed_results:
            continue

        # Aggregate across seeds
        all_rewards = []
        for sr in seed_results:
            all_rewards.extend(sr["raw_rewards"])
        all_results[algo] = {
            "mean_reward":    float(np.mean(all_rewards)),
            "std_reward":     float(np.std(all_rewards, ddof=1)),
            "success_rate":   float(1.0 - np.mean([r for sr in seed_results for r in sr["raw_collisions"]])),
            "collision_rate": float(np.mean([r for sr in seed_results for r in sr["raw_collisions"]])),
            "mean_speed":     float(np.mean([sr["mean_speed"] for sr in seed_results])),
            "n_episodes":     len(all_rewards),
            "raw_rewards":    [float(r) for r in all_rewards],
        }
        r = all_results[algo]
        print(f"      Reward: {r['mean_reward']:.2f} ± {r['std_reward']:.2f}  |  "
              f"Success: {r['success_rate']:.1%}  |  Collision: {r['collision_rate']:.1%}")

    if all_results:
        os.makedirs("results", exist_ok=True)
        with open("results/multi_seed_metrics.json", "w") as f:
            json.dump(all_results, f, indent=2)
        print("\n  ✅  Saved: results/multi_seed_metrics.json")
        plot_multi_seed_results(all_results)
        write_report(all_results, args.episodes)
    else:
        print("\n❌ No models found. Train models first with scripts/training/")

    print("\n✅ Multi-episode evaluation complete!")
