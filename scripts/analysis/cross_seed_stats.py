"""
cross_seed_stats.py
===================
Compute proper cross-seed statistics for the multi-seed results.

Usage:
    python scripts/analysis/cross_seed_stats.py
    python scripts/analysis/cross_seed_stats.py --results-dir results/multi_seed
    python scripts/analysis/cross_seed_stats.py --latex   # output LaTeX table rows

Outputs:
    results/cross_seed_stats.json         (full stats)
    results/plots/multiseed_boxplot.png   (box plot with Wilcoxon annotations)
    Prints LaTeX table rows to stdout (copy into main.tex)
"""

import os
import sys
import json
import argparse
import numpy as np
from scipy import stats
from itertools import combinations
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

os.makedirs("results/plots", exist_ok=True)

ALGO_LABELS = {
    "ppo":     "PPO",
    "rainbow": "Rainbow DQN",
    "dueling": "Dueling DQN",
    "double":  "Double DQN",
    "dqn":     "DQN",
    "sac":     "SAC",
}

ALGO_ORDER = ["ppo", "sac", "rainbow", "dueling", "double", "dqn"]
COLORS     = {
    "ppo":     "#2196F3",
    "sac":     "#9C27B0",
    "rainbow": "#FF9800",
    "dueling": "#4CAF50",
    "double":  "#00BCD4",
    "dqn":     "#F44336",
}


def cohens_d(x: np.ndarray, y: np.ndarray) -> float:
    """Compute Cohen's d effect size between two groups."""
    nx, ny = len(x), len(y)
    pooled_std = np.sqrt(((nx - 1) * x.std()**2 + (ny - 1) * y.std()**2) / (nx + ny - 2))
    return (x.mean() - y.mean()) / (pooled_std + 1e-9)


def effect_label(d: float) -> str:
    d = abs(d)
    if d < 0.2:  return "negligible"
    if d < 0.5:  return "small"
    if d < 0.8:  return "medium"
    return "large"


def load_results(results_dir: str) -> dict:
    """
    Load all *_results.json files from results_dir.
    Returns dict: algo -> list of metric dicts (one per seed).
    """
    data = {}
    if not os.path.exists(results_dir):
        print(f"[WARNING] results_dir not found: {results_dir}")
        return data

    for fname in os.listdir(results_dir):
        if not fname.endswith("_results.json"):
            continue
        with open(os.path.join(results_dir, fname)) as f:
            r = json.load(f)
        algo = r.get("algo", fname.split("_")[0])
        if algo not in data:
            data[algo] = []
        data[algo].append(r)

    print(f"[load] Loaded {sum(len(v) for v in data.values())} results "
          f"across {len(data)} algorithms from {results_dir}")
    return data


def compute_stats(data: dict) -> dict:
    """
    For each algo, compute:
      - Per-seed means (the correct unit for cross-seed inference)
      - Mean ± std across seeds
      - 95% CI using t-distribution
      - Wilcoxon signed-rank test vs PPO (paired, if same seeds available)
      - Cohen's d vs PPO
    """
    stats_out = {}
    ppo_rewards = None

    if "ppo" in data:
        ppo_rewards = np.array([r["mean_reward"] for r in data["ppo"]])

    for algo, results in data.items():
        rewards    = np.array([r["mean_reward"]    for r in results])
        successes  = np.array([r["success_rate"]   for r in results])
        collisions = np.array([r["collision_rate"] for r in results])
        n          = len(rewards)

        # 95% CI using t-distribution (correct for small n=3)
        t_crit = stats.t.ppf(0.975, df=max(n - 1, 1))
        sem    = rewards.std() / np.sqrt(n) if n > 1 else 0.0

        entry = {
            "algo":        algo,
            "n_seeds":     n,
            "mean_reward": float(rewards.mean()),
            "std_reward":  float(rewards.std()),
            "ci95_reward": float(t_crit * sem),
            "mean_success":    float(successes.mean()),
            "std_success":     float(successes.std()),
            "mean_collision":  float(collisions.mean()),
            "std_collision":   float(collisions.std()),
        }

        # Statistical tests vs PPO
        if ppo_rewards is not None and algo != "ppo" and n >= 2:
            min_n = min(len(ppo_rewards), len(rewards))
            try:
                w_stat, p_val = stats.wilcoxon(
                    ppo_rewards[:min_n], rewards[:min_n],
                    alternative="greater",
                )
                entry["wilcoxon_p"]    = float(p_val)
                entry["wilcoxon_stat"] = float(w_stat)
            except Exception as e:
                entry["wilcoxon_p"]    = None
                entry["wilcoxon_note"] = str(e)

            d = cohens_d(ppo_rewards[:min_n], rewards[:min_n])
            entry["cohens_d"]      = float(d)
            entry["effect_size"]   = effect_label(d)

        stats_out[algo] = entry

    return stats_out


def sig_stars(p: float | None) -> str:
    """Convert p-value to significance stars."""
    if p is None:  return "n/a"
    if p < 0.001:  return "***"
    if p < 0.01:   return "**"
    if p < 0.05:   return "*"
    return "ns"


def print_latex_table(stats_out: dict):
    """Print a LaTeX-formatted results table for the paper."""
    print("\n% ── Multi-Seed Results Table (paste into main.tex) ─────────────────")
    print("\\begin{table}[htbp]")
    print("\\centering")
    print("\\caption{Multi-Seed Evaluation Results (3 seeds, 50 episodes each).")
    print("         $\\dagger$ = Wilcoxon signed-rank vs PPO (one-tailed, $H_1$: PPO $>$ algo).}")
    print("\\label{tab:multiseed}")
    print("\\begin{tabular}{@{}lcccc@{}}")
    print("\\toprule")
    print("\\textbf{Algorithm} & \\textbf{Reward} & \\textbf{Success} "
          "& \\textbf{Collision} & \\textbf{$p^\\dagger$} \\\\ \\midrule")

    for algo in ALGO_ORDER:
        if algo not in stats_out:
            continue
        s = stats_out[algo]
        reward_str  = f"{s['mean_reward']:.2f} $\\pm$ {s['std_reward']:.2f}"
        success_str = f"{s['mean_success']:.1%} $\\pm$ {s['std_success']:.1%}"
        coll_str    = f"{s['mean_collision']:.1%}"
        p_str       = sig_stars(s.get("wilcoxon_p")) if algo != "ppo" else "---"

        label = ALGO_LABELS.get(algo, algo.upper())
        bold_start = "\\textbf{" if algo == "ppo" else ""
        bold_end   = "}"         if algo == "ppo" else ""
        print(f"{bold_start}{label}{bold_end} & {reward_str} & {success_str} & {coll_str} & {p_str} \\\\")

    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\end{table}")
    print("% ─────────────────────────────────────────────────────────────────────\n")


def plot_multiseed(data: dict, stats_out: dict, out_path="results/plots/multiseed_boxplot.png"):
    """Box plot of per-seed mean rewards with Wilcoxon annotations."""
    algos_present = [a for a in ALGO_ORDER if a in data]
    if not algos_present:
        print("[plot] No data to plot")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.patch.set_facecolor("#0D1117")

    for ax_idx, metric in enumerate(["mean_reward", "success_rate"]):
        ax = axes[ax_idx]
        ax.set_facecolor("#161B22")

        boxes_data = [
            [r[metric] for r in data[a]]
            for a in algos_present
        ]
        colors_list = [COLORS.get(a, "#FFFFFF") for a in algos_present]
        labels_list = [ALGO_LABELS.get(a, a.upper()) for a in algos_present]

        bp = ax.boxplot(boxes_data, patch_artist=True, notch=False,
                        medianprops={"color": "white", "linewidth": 2})
        for patch, color in zip(bp["boxes"], colors_list):
            patch.set_facecolor(color)
            patch.set_alpha(0.8)
        for element in ["whiskers", "caps", "fliers"]:
            for item in bp[element]:
                item.set_color("#AAAAAA")

        # Annotate p-values vs PPO
        if "ppo" in algos_present and ax_idx == 0:
            ppo_idx = algos_present.index("ppo")
            for i, algo in enumerate(algos_present):
                if algo == "ppo":
                    continue
                p = stats_out.get(algo, {}).get("wilcoxon_p")
                stars = sig_stars(p)
                y_max = max(r[metric] for r in data[algo]) + 0.5
                ax.text(i + 1, y_max, stars, ha="center", va="bottom",
                        color="white", fontsize=9, fontweight="bold")

        ax.set_xticks(range(1, len(algos_present) + 1))
        ax.set_xticklabels(labels_list, rotation=20, ha="right", color="white", fontsize=8)
        ax.tick_params(colors="white")
        for spine in ax.spines.values():
            spine.set_edgecolor("#30363D")
        ax.grid(color="#30363D", linestyle="--", linewidth=0.5, axis="y")

        ylabel = "Mean episodic reward" if metric == "mean_reward" else "Success rate"
        ax.set_ylabel(ylabel, color="white", fontsize=10)
        title  = "Mean Reward per Algorithm (3 seeds)" if metric == "mean_reward" \
            else "Success Rate per Algorithm (3 seeds)"
        ax.set_title(title, color="white", fontsize=10)
        if metric == "success_rate":
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))

    plt.suptitle("Multi-Seed Statistical Comparison (* p<0.05, ** p<0.01, *** p<0.001 vs PPO)",
                 color="white", fontsize=11, y=1.01)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"[plot] saved -> {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Cross-seed statistical analysis")
    parser.add_argument("--results-dir", default="results/multi_seed")
    parser.add_argument("--latex",  action="store_true", help="Print LaTeX table")
    parser.add_argument("--output", default="results/cross_seed_stats.json")
    args = parser.parse_args()

    data = load_results(args.results_dir)

    if not data:
        print("[WARNING] No results found. Run train_all_seeds.py first.")
        print("          Expected files: results/multi_seed/<algo>_seed<N>_results.json")
        return

    stats_out = compute_stats(data)

    with open(args.output, "w") as f:
        json.dump(stats_out, f, indent=2)
    print(f"[saved] {args.output}")

    # Console summary
    print(f"\n{'='*65}")
    print(f"{'Algorithm':<15} {'Reward':>12} {'Success':>10} {'p vs PPO':>10} {'d':>8}")
    print(f"{'-'*65}")
    for algo in ALGO_ORDER:
        if algo not in stats_out:
            continue
        s = stats_out[algo]
        p_str = f"{s.get('wilcoxon_p', 0.0):.4f}" if algo != "ppo" else "  ---  "
        d_str = f"{s.get('cohens_d', 0.0):+.3f}" if algo != "ppo" else "  ---  "
        print(f"{ALGO_LABELS.get(algo, algo.upper()):<15} "
              f"{s['mean_reward']:>8.2f}±{s['std_reward']:.2f} "
              f"{s['mean_success']:>9.1%} "
              f"{p_str:>10} "
              f"{d_str:>8}")
    print(f"{'='*65}")

    if args.latex:
        print_latex_table(stats_out)

    plot_multiseed(data, stats_out)
    print("\n[done] Cross-seed analysis complete!")


if __name__ == "__main__":
    main()
