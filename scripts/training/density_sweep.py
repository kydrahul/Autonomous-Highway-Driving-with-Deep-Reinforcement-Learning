"""
density_sweep.py
================
Train PPO, Rainbow DQN, and DQN across traffic densities {10,20,30,40,50}.
Each algorithm × density × seed combination is one run.

Usage:
    python scripts/training/density_sweep.py
    python scripts/training/density_sweep.py --algos ppo rainbow --densities 10 30 50 --seeds 42 123
    python scripts/training/density_sweep.py --steps 300000 --eval-episodes 50

Outputs:
    models/density_sweep/<algo>_density<N>_seed<S>/  (saved model)
    results/density_sweep/<algo>_density<N>_seed<S>_results.json
    results/plots/density_sweep.png
"""

import os
import json
import argparse
import numpy as np
import gymnasium as gym
import highway_env  # noqa: F401 — registers highway-v0
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from stable_baselines3 import DQN, PPO
from stable_baselines3.common.vec_env import DummyVecEnv

# ── add project root to path so custom Rainbow can be imported ─────────────────
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
try:
    from scripts.training.train_rainbow_dqn import (
        RainbowDQN, RainbowPolicy, PrioritizedReplayBuffer
    )
except ImportError:
    RainbowDQN = None
    print("[WARNING] RainbowDQN not found - will skip rainbow in sweep")

# ──────────────────────────────────────────────────────────────────────────────
DENSITIES = [10, 20, 30, 40, 50]
SEEDS     = [42, 123, 456]
ALGOS     = ["ppo", "rainbow", "dqn"]
STEPS     = 300_000
EVAL_EPS  = 50

os.makedirs("models/density_sweep", exist_ok=True)
os.makedirs("results/density_sweep", exist_ok=True)
os.makedirs("results/plots",         exist_ok=True)


# ── Left Lane Reward Wrapper (must match other training scripts) ──────────────
class LeftLaneRewardWrapper(gym.Wrapper):
    """Adds a bonus reward for staying in the left-most lane."""
    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)
        current_lane = self.unwrapped.vehicle.lane_index[2]
        total_lanes  = self.unwrapped.config["lanes_count"]
        left_reward  = (total_lanes - 1 - current_lane) / (total_lanes - 1)
        reward      += 0.1 * left_reward
        return obs, reward, done, truncated, info


def make_env(density: int, seed: int):
    """Return a factory for a highway-v0 env with the given vehicle count."""
    def _init():
        env = gym.make("highway-v0", render_mode=None)
        env.unwrapped.config.update({
            "vehicles_count": density,
            "lanes_count": 4,
            "observation": {
                "type": "Kinematics",
                "vehicles_count": 15,
                "features": ["x", "y", "vx", "vy", "cos_h", "sin_h"],
                "normalize": True,
                "absolute": False,
            },
            "action": {"type": "DiscreteMetaAction"},
            "initial_spacing": 2,
            "collision_reward": -1,
            "right_lane_reward": 0.1,
            "high_speed_reward": 0.4,
            "reward_speed_range": [20, 30],
            "normalize_reward": True,
            "duration": 40,
            "simulation_frequency": 5,
            "policy_frequency": 1,
            "other_vehicles_type": "highway_env.vehicle.behavior.IDMVehicle",
        })
        env = LeftLaneRewardWrapper(env)
        env.reset(seed=seed)
        return env
    return _init


def evaluate(model, density: int, seed: int, n_eps: int = EVAL_EPS):
    """Run n_eps stochastic evaluation episodes, return dict of metrics."""
    env = make_env(density, seed + 1000)()
    rewards, lengths, crashed = [], [], []
    obs, _ = env.reset()
    ep_r = ep_len = 0
    ep_count = 0

    while ep_count < n_eps:
        action, _ = model.predict(obs, deterministic=False)
        obs, r, terminated, truncated, info = env.step(int(action))
        ep_r   += r
        ep_len += 1
        if terminated or truncated:
            rewards.append(ep_r)
            lengths.append(ep_len)
            crashed.append(1 if info.get("crashed", False) else 0)
            obs, _ = env.reset()
            ep_r = ep_len = 0
            ep_count += 1

    env.close()
    return {
        "mean_reward":     float(np.mean(rewards)),
        "std_reward":      float(np.std(rewards)),
        "success_rate":    float(1.0 - np.mean(crashed)),
        "collision_rate":  float(np.mean(crashed)),
        "mean_ep_length":  float(np.mean(lengths)),
        "n_episodes":      n_eps,
    }


def train_one(algo: str, density: int, seed: int, steps: int):
    """Train a single (algo, density, seed) combination. Returns eval metrics."""
    tag = f"{algo}_density{density}_seed{seed}"
    result_path = f"results/density_sweep/{tag}_results.json"
    model_dir   = f"models/density_sweep/{tag}"

    if os.path.exists(result_path):
        print(f"  [skip] {tag} — results already exist")
        with open(result_path) as f:
            return json.load(f)

    print(f"\n{'='*60}")
    print(f"  Training: {tag}  ({steps:,} steps)")
    print(f"{'='*60}")

    vec_env = DummyVecEnv([make_env(density, seed)])

    if algo == "ppo":
        model = PPO(
            "MlpPolicy", vec_env, seed=seed, verbose=0,
            learning_rate=3e-4, n_steps=512, batch_size=64,
            n_epochs=10, gamma=0.99, gae_lambda=0.95,
            clip_range=0.2, ent_coef=0.01,
            policy_kwargs={"net_arch": [256, 256]},
        )
    elif algo == "rainbow" and RainbowDQN is not None:
        model = RainbowDQN(
            policy=RainbowPolicy,
            env=vec_env, seed=seed, verbose=0,
            replay_buffer_class=PrioritizedReplayBuffer,
            replay_buffer_kwargs=dict(
                alpha=0.6, beta_start=0.4, beta_frames=steps,
                n_step=3, gamma=0.99,
            ),
            learning_rate=5e-4, batch_size=64, gamma=0.99,
            buffer_size=100_000,
            tau=1.0, target_update_interval=1000,
            train_freq=4, gradient_steps=1,
            learning_starts=1000,
            exploration_fraction=0.0,
            exploration_initial_eps=0.0,
            exploration_final_eps=0.0,
            policy_kwargs={"net_arch": [256, 256]},
        )
    else:  # plain DQN fallback or if rainbow not available
        model = DQN(
            "MlpPolicy", vec_env, seed=seed, verbose=0,
            learning_rate=5e-4, batch_size=64, gamma=0.99,
            buffer_size=100_000, exploration_fraction=0.1,
            exploration_final_eps=0.05,
            policy_kwargs={"net_arch": [256, 256]},
        )

    model.learn(total_timesteps=steps)
    os.makedirs(model_dir, exist_ok=True)
    model.save(os.path.join(model_dir, "model"))

    metrics = evaluate(model, density, seed)
    metrics.update({"algo": algo, "density": density, "seed": seed, "steps": steps})
    with open(result_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"  => success={metrics['success_rate']:.1%}  "
          f"reward={metrics['mean_reward']:.2f}  "
          f"collision={metrics['collision_rate']:.1%}")
    return metrics


def plot_density_sweep(results: list[dict], out_path="results/plots/density_sweep.png"):
    """Hero figure: success rate vs vehicle density for each algorithm."""
    import matplotlib.pyplot as plt
    from collections import defaultdict

    # Group by (algo, density) → list of success rates across seeds
    grouped: dict = defaultdict(list)
    for r in results:
        grouped[(r["algo"], r["density"])].append(r["success_rate"])

    colors = {"ppo": "#2196F3", "rainbow": "#FF9800", "dqn": "#F44336"}
    labels = {"ppo": "PPO (on-policy)", "rainbow": "Rainbow DQN", "dqn": "DQN"}
    markers = {"ppo": "o", "rainbow": "s", "dqn": "^"}

    fig, ax = plt.subplots(figsize=(7, 4.5))
    fig.patch.set_facecolor("#0D1117")
    ax.set_facecolor("#161B22")

    for algo in ["ppo", "rainbow", "dqn"]:
        densities_sorted = sorted(set(d for (a, d) in grouped if a == algo))
        means = [np.mean(grouped[(algo, d)]) for d in densities_sorted]
        stds  = [np.std(grouped[(algo, d)])  for d in densities_sorted]
        ax.plot(densities_sorted, means, color=colors[algo],
                marker=markers[algo], linewidth=2.5, markersize=8,
                label=labels[algo], zorder=3)
        ax.fill_between(densities_sorted,
                         [m - s for m, s in zip(means, stds)],
                         [m + s for m, s in zip(means, stds)],
                         color=colors[algo], alpha=0.15, zorder=2)

    # Annotate crossover threshold if detectable
    ppo_means    = {d: np.mean(grouped[("ppo",     d)]) for d in DENSITIES}
    rainbow_means = {d: np.mean(grouped[("rainbow", d)]) for d in DENSITIES}
    for d in DENSITIES:
        if d in ppo_means and d in rainbow_means:
            if ppo_means[d] > rainbow_means[d] + 0.05:
                ax.axvline(d, color="white", linestyle="--", alpha=0.4, linewidth=1)
                ax.text(d + 0.5, 0.05, f"Threshold\n~{d} vehicles",
                        color="white", fontsize=7, alpha=0.7)
                break

    ax.set_xlabel("Number of surrounding vehicles", color="white", fontsize=11)
    ax.set_ylabel("Success rate (no collision)", color="white", fontsize=11)
    ax.set_title("Distribution Shift Phase Transition:\nOn-Policy vs Off-Policy DRL in Dense Traffic",
                 color="white", fontsize=12, pad=10)
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#30363D")
    ax.legend(facecolor="#161B22", edgecolor="#30363D", labelcolor="white", fontsize=10)
    ax.set_xlim(8, 52)
    ax.set_ylim(-0.02, 1.05)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    ax.grid(color="#30363D", linestyle="--", linewidth=0.5)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"[plot] saved -> {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Density sweep experiment")
    parser.add_argument("--algos",     nargs="+", default=ALGOS,     choices=["ppo","rainbow","dqn"])
    parser.add_argument("--densities", nargs="+", default=DENSITIES, type=int)
    parser.add_argument("--seeds",     nargs="+", default=SEEDS,     type=int)
    parser.add_argument("--steps",     default=STEPS,                type=int)
    parser.add_argument("--eval-episodes", default=EVAL_EPS,         type=int)
    args = parser.parse_args()

    all_results = []
    total = len(args.algos) * len(args.densities) * len(args.seeds)
    done  = 0

    for algo in args.algos:
        for density in sorted(args.densities):
            for seed in args.seeds:
                done += 1
                print(f"\n[{done}/{total}] algo={algo}  density={density}  seed={seed}")
                r = train_one(algo, density, seed, args.steps)
                all_results.append(r)

    # Save aggregate
    with open("results/density_sweep/all_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print("\n[done] All results saved to results/density_sweep/all_results.json")

    # Plot
    plot_density_sweep(all_results)
    print("[done] Density sweep complete!")


if __name__ == "__main__":
    main()
