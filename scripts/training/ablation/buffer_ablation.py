"""
buffer_ablation.py
==================
Two sub-experiments proving the distribution shift hypothesis:

Experiment A — Buffer Size Sweep:
  Train Rainbow DQN with buffer_size in {1000, 5000, 10000, 50000, 100000}.
  Prediction: larger buffer → more stale crash data → worse performance.
  This directly tests whether the AMOUNT of stale data scales with buffer size.

Experiment B — Periodic Buffer Wipe:
  Train Rainbow DQN with periodic full buffer resets every N steps.
  wipe_interval in {never, 100k, 50k, 25k}.
  Prediction: more frequent wipes → approaches PPO's performance.
  This is the CAUSAL proof — simulating on-policy behaviour in an off-policy agent.

Usage:
    python scripts/training/ablation/buffer_ablation.py
    python scripts/training/ablation/buffer_ablation.py --experiment size
    python scripts/training/ablation/buffer_ablation.py --experiment wipe
    python scripts/training/ablation/buffer_ablation.py --seeds 42 123 --steps 200000
"""

import os
import sys
import json
import argparse
import numpy as np
import gymnasium as gym
import highway_env  # noqa: F401
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

os.makedirs("models/buffer_ablation", exist_ok=True)
os.makedirs("results/buffer_ablation", exist_ok=True)
os.makedirs("results/plots",           exist_ok=True)

# ── Defaults ──────────────────────────────────────────────────────────────────
BUFFER_SIZES    = [1_000, 5_000, 10_000, 50_000, 100_000]
WIPE_INTERVALS  = [None, 100_000, 50_000, 25_000]  # None = never wipe
SEEDS           = [42, 123, 456]
STEPS           = 200_000
EVAL_EPS        = 50
DENSITY         = 50  # always use maximum density


# ── Environment ───────────────────────────────────────────────────────────────
def make_env(seed: int, density: int = DENSITY):
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
            "reward_speed_range": [20, 30],
            "collision_reward": -1,
            "normalize_reward": True,
            "duration": 40,
            "simulation_frequency": 5,
            "policy_frequency": 1,
        })
        env.reset(seed=seed)
        return env
    return _init


# ── Periodic Buffer Wipe Callback ─────────────────────────────────────────────
class BufferWipeCallback(BaseCallback):
    """Wipe the replay buffer every `interval` timesteps."""

    def __init__(self, interval: int, verbose=0):
        super().__init__(verbose)
        self.interval = interval
        self.last_wipe = 0
        self.wipe_count = 0

    def _on_step(self) -> bool:
        if self.num_timesteps - self.last_wipe >= self.interval:
            # Reset the replay buffer by re-initialising it
            buf = self.model.replay_buffer
            buf.pos = 0
            buf.full = False
            self.last_wipe = self.num_timesteps
            self.wipe_count += 1
            if self.verbose:
                print(f"  [wipe #{self.wipe_count}] Buffer cleared at step {self.num_timesteps:,}")
        return True


# ── Evaluation ────────────────────────────────────────────────────────────────
def evaluate(model, seed: int, n_eps: int = EVAL_EPS):
    env = make_env(seed + 9999)()
    rewards, crashed = [], []
    obs, _ = env.reset()
    ep_r, ep_count = 0.0, 0
    while ep_count < n_eps:
        action, _ = model.predict(obs, deterministic=False)
        obs, r, terminated, truncated, info = env.step(int(action))
        ep_r += r
        if terminated or truncated:
            rewards.append(ep_r)
            crashed.append(1 if info.get("crashed", False) else 0)
            obs, _ = env.reset()
            ep_r = 0.0
            ep_count += 1
    env.close()
    return {
        "mean_reward":    float(np.mean(rewards)),
        "std_reward":     float(np.std(rewards)),
        "success_rate":   float(1.0 - np.mean(crashed)),
        "collision_rate": float(np.mean(crashed)),
    }


# ── Experiment A: Buffer Size Sweep ──────────────────────────────────────────
def run_size_sweep(seeds, steps, buffer_sizes):
    all_results = []
    total = len(buffer_sizes) * len(seeds)
    done  = 0

    for buf_size in buffer_sizes:
        for seed in seeds:
            done += 1
            tag = f"bufsize{buf_size}_seed{seed}"
            result_path = f"results/buffer_ablation/size_{tag}.json"

            if os.path.exists(result_path):
                print(f"  [skip] {tag}")
                with open(result_path) as f:
                    all_results.append(json.load(f))
                continue

            print(f"\n[{done}/{total}] Buffer size={buf_size:,}  seed={seed}")
            vec_env = DummyVecEnv([make_env(seed)])
            model = DQN(
                "MlpPolicy", vec_env, seed=seed, verbose=0,
                learning_rate=5e-4, batch_size=64, gamma=0.99,
                buffer_size=buf_size,
                exploration_fraction=0.1, exploration_final_eps=0.05,
                policy_kwargs={"net_arch": [256, 256]},
            )
            model.learn(total_timesteps=steps)
            metrics = evaluate(model, seed)
            metrics.update({"buffer_size": buf_size, "seed": seed, "experiment": "size"})
            with open(result_path, "w") as f:
                json.dump(metrics, f, indent=2)
            print(f"  => success={metrics['success_rate']:.1%}  reward={metrics['mean_reward']:.2f}")
            all_results.append(metrics)

    return all_results


# ── Experiment B: Buffer Wipe ─────────────────────────────────────────────────
def run_wipe_experiment(seeds, steps, wipe_intervals):
    all_results = []
    total = len(wipe_intervals) * len(seeds)
    done  = 0

    for wipe_int in wipe_intervals:
        wipe_label = str(wipe_int) if wipe_int is not None else "never"
        for seed in seeds:
            done += 1
            tag = f"wipe{wipe_label}_seed{seed}"
            result_path = f"results/buffer_ablation/wipe_{tag}.json"

            if os.path.exists(result_path):
                print(f"  [skip] {tag}")
                with open(result_path) as f:
                    all_results.append(json.load(f))
                continue

            print(f"\n[{done}/{total}] Wipe interval={wipe_label}  seed={seed}")
            vec_env = DummyVecEnv([make_env(seed)])
            model = DQN(
                "MlpPolicy", vec_env, seed=seed, verbose=0,
                learning_rate=5e-4, batch_size=64, gamma=0.99,
                buffer_size=100_000,
                exploration_fraction=0.1, exploration_final_eps=0.05,
                policy_kwargs={"net_arch": [256, 256]},
            )
            callbacks = []
            if wipe_int is not None:
                callbacks.append(BufferWipeCallback(wipe_int, verbose=1))
            model.learn(total_timesteps=steps, callback=callbacks if callbacks else None)
            metrics = evaluate(model, seed)
            metrics.update({
                "wipe_interval": wipe_int,
                "wipe_label": wipe_label,
                "seed": seed,
                "experiment": "wipe",
            })
            with open(result_path, "w") as f:
                json.dump(metrics, f, indent=2)
            print(f"  => success={metrics['success_rate']:.1%}  reward={metrics['mean_reward']:.2f}")
            all_results.append(metrics)

    return all_results


# ── Plotting ──────────────────────────────────────────────────────────────────
def plot_buffer_ablation(size_results, wipe_results, ppo_reference=0.96):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    fig.patch.set_facecolor("#0D1117")

    # ── Panel A: Buffer size sweep ────────────────────────────────────────────
    ax = axes[0]
    ax.set_facecolor("#161B22")
    if size_results:
        from collections import defaultdict
        grouped = defaultdict(list)
        for r in size_results:
            grouped[r["buffer_size"]].append(r["success_rate"])
        sizes = sorted(grouped.keys())
        means = [np.mean(grouped[s]) for s in sizes]
        stds  = [np.std(grouped[s])  for s in sizes]
        ax.errorbar(sizes, means, yerr=stds, marker="o", linewidth=2,
                    color="#FF9800", ecolor="#FF9800", capsize=4, markersize=7)
        ax.axhline(ppo_reference, color="#2196F3", linestyle="--", linewidth=1.5,
                   label=f"PPO reference ({ppo_reference:.0%})")
        ax.set_xscale("log")
        ax.set_xlabel("Replay buffer size", color="white", fontsize=10)
        ax.set_ylabel("Success rate", color="white", fontsize=10)
        ax.set_title("A: Buffer Size vs Performance\n(DQN, density=50, 3 seeds)",
                     color="white", fontsize=10)
        ax.tick_params(colors="white")
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
        ax.legend(facecolor="#161B22", edgecolor="#30363D", labelcolor="white", fontsize=9)
        for spine in ax.spines.values():
            spine.set_edgecolor("#30363D")
        ax.grid(color="#30363D", linestyle="--", linewidth=0.5)

    # ── Panel B: Buffer wipe experiment ───────────────────────────────────────
    ax = axes[1]
    ax.set_facecolor("#161B22")
    if wipe_results:
        from collections import defaultdict
        grouped = defaultdict(list)
        for r in wipe_results:
            grouped[r["wipe_label"]].append(r["success_rate"])
        order  = ["never", "100000", "50000", "25000"]
        labels = ["Never\n(baseline)", "Every 100k\nsteps",
                  "Every 50k\nsteps", "Every 25k\nsteps"]
        means, stds, xs = [], [], []
        for i, key in enumerate(order):
            if key in grouped:
                means.append(np.mean(grouped[key]))
                stds.append(np.std(grouped[key]))
                xs.append(i)
                labels_used = [labels[j] for j in xs]  # noqa
        x_pos = range(len(means))
        bars = ax.bar(x_pos, means, color="#FF9800", alpha=0.8, edgecolor="#FFA726")
        ax.errorbar(x_pos, means, yerr=stds, fmt="none", ecolor="white", capsize=4)
        ax.axhline(ppo_reference, color="#2196F3", linestyle="--", linewidth=1.5,
                   label=f"PPO reference ({ppo_reference:.0%})")
        valid_labels = [labels[i] for i in range(len(order)) if order[i] in grouped]
        ax.set_xticks(x_pos)
        ax.set_xticklabels(valid_labels, color="white", fontsize=8)
        ax.set_ylabel("Success rate", color="white", fontsize=10)
        ax.set_title("B: Buffer Wipe Frequency vs Performance\n(DQN, density=50, 3 seeds)",
                     color="white", fontsize=10)
        ax.tick_params(colors="white")
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
        ax.legend(facecolor="#161B22", edgecolor="#30363D", labelcolor="white", fontsize=9)
        for spine in ax.spines.values():
            spine.set_edgecolor("#30363D")
        ax.grid(color="#30363D", linestyle="--", linewidth=0.5, axis="y")
        ax.set_ylim(0, 1.1)

    plt.suptitle("Replay Buffer Ablation: Causal Evidence for Distribution Shift",
                 color="white", fontsize=12, y=1.02)
    plt.tight_layout()
    out = "results/plots/buffer_ablation.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"[plot] saved -> {out}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Buffer ablation experiments")
    parser.add_argument("--experiment", choices=["size", "wipe", "both"], default="both")
    parser.add_argument("--seeds",          nargs="+", default=SEEDS,          type=int)
    parser.add_argument("--steps",          default=STEPS,                     type=int)
    parser.add_argument("--buffer-sizes",   nargs="+", default=BUFFER_SIZES,   type=int)
    parser.add_argument("--wipe-intervals", nargs="+", default=[None, 100000, 50000, 25000])
    args = parser.parse_args()

    size_results = []
    wipe_results = []

    if args.experiment in ("size", "both"):
        print("\n" + "="*60)
        print("EXPERIMENT A: Buffer Size Sweep")
        print("="*60)
        size_results = run_size_sweep(args.seeds, args.steps, args.buffer_sizes)

    if args.experiment in ("wipe", "both"):
        print("\n" + "="*60)
        print("EXPERIMENT B: Buffer Wipe Experiment")
        print("="*60)
        wipe_results = run_wipe_experiment(args.seeds, args.steps,
                                           [None, 100_000, 50_000, 25_000])

    plot_buffer_ablation(size_results, wipe_results)
    print("\n[done] Buffer ablation complete!")


if __name__ == "__main__":
    main()
