"""
Statistical Significance Tests
================================
Computes formal statistical tests between all algorithm pairs from the 50
evaluation episodes stored in results/metrics.json.

Tests performed:
  - Wilcoxon Signed-Rank Test (paired): PPO vs each other algorithm
  - Mann-Whitney U Test (unpaired): all pairs
  - 95% Bootstrap Confidence Intervals for mean reward
  - Cohen's d effect size

Outputs:
  results/analysis/statistical_tests.txt   — human-readable report
  results/analysis/significance_matrix.png — heatmap of p-values

Usage:
    python scripts/analysis/statistical_tests.py
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy import stats
from itertools import combinations

METRICS_PATH = "results/metrics.json"
OUT_DIR      = "results/analysis"
PLOT_DIR     = "results/plots"
os.makedirs(OUT_DIR,  exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)

plt.rcParams.update({
    "figure.facecolor": "#0f0f1a",
    "axes.facecolor":   "#1a1a2e",
    "axes.edgecolor":   "#444466",
    "axes.labelcolor":  "#e0e0e0",
    "xtick.color":      "#e0e0e0",
    "ytick.color":      "#e0e0e0",
    "text.color":       "#e0e0e0",
    "font.size":        10,
})

ALGO_ORDER = ["PPO", "Rainbow DQN", "Dueling DQN", "Double DQN", "DQN"]


# ─────────────────────────────────────────────────────────────────────────────────
def load_rewards(metrics: dict) -> dict[str, np.ndarray]:
    return {algo: np.array(metrics[algo]["raw_rewards"]) for algo in ALGO_ORDER if algo in metrics}


def bootstrap_ci(data: np.ndarray, n_boot: int = 10_000, ci: float = 0.95) -> tuple:
    """Return (mean, lower_ci, upper_ci) using percentile bootstrap."""
    rng     = np.random.default_rng(42)
    samples = rng.choice(data, size=(n_boot, len(data)), replace=True).mean(axis=1)
    alpha   = (1 - ci) / 2
    return float(data.mean()), float(np.percentile(samples, 100 * alpha)), float(np.percentile(samples, 100 * (1 - alpha)))


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    """Pooled Cohen's d effect size."""
    pooled_std = np.sqrt((np.var(a, ddof=1) + np.var(b, ddof=1)) / 2)
    return float((np.mean(a) - np.mean(b)) / pooled_std) if pooled_std > 0 else 0.0


def interpret_d(d: float) -> str:
    d = abs(d)
    if d < 0.2:  return "negligible"
    if d < 0.5:  return "small"
    if d < 0.8:  return "medium"
    return "large"


def interpret_p(p: float) -> str:
    if p < 0.001: return "***  (p<0.001)"
    if p < 0.01:  return "**   (p<0.01)"
    if p < 0.05:  return "*    (p<0.05)"
    return "n.s. (p≥0.05)"


# ─────────────────────────────────────────────────────────────────────────────────
def run_tests(rewards: dict[str, np.ndarray]) -> dict:
    """Run all statistical tests between all pairs."""
    results = {}
    algos = list(rewards.keys())
    for a, b in combinations(algos, 2):
        ra, rb = rewards[a], rewards[b]
        # Wilcoxon (for equal-length paired data) or Mann-Whitney
        if len(ra) == len(rb):
            stat_w, p_w = stats.wilcoxon(ra, rb, alternative="two-sided")
            test_name = "Wilcoxon"
        else:
            stat_w, p_w = stats.mannwhitneyu(ra, rb, alternative="two-sided")
            test_name = "Mann-Whitney U"
        stat_m, p_m = stats.mannwhitneyu(ra, rb, alternative="two-sided")
        d = cohens_d(ra, rb)
        results[(a, b)] = {
            "test":   test_name,
            "stat_w": stat_w,
            "p_w":    p_w,
            "stat_m": stat_m,
            "p_m":    p_m,
            "d":      d,
            "sig":    p_w < 0.05,
        }
    return results


def plot_significance_heatmap(rewards: dict, test_results: dict):
    algos = list(rewards.keys())
    n     = len(algos)
    pmat  = np.ones((n, n))

    for i, a in enumerate(algos):
        for j, b in enumerate(algos):
            if i == j:
                continue
            key = (a, b) if (a, b) in test_results else (b, a)
            if key in test_results:
                pmat[i, j] = test_results[key]["p_w"]

    fig, ax = plt.subplots(figsize=(8, 6))
    fig.patch.set_facecolor("#0f0f1a")

    cmap = plt.cm.RdYlGn_r
    norm = mcolors.LogNorm(vmin=1e-4, vmax=1.0)
    im   = ax.imshow(pmat, cmap=cmap, norm=norm, aspect="auto")

    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(algos, rotation=35, ha="right", fontsize=9)
    ax.set_yticklabels(algos, fontsize=9)
    ax.set_title("Pairwise Significance Heatmap (p-values, log scale)\n"
                 "Green = significant difference, Red = not significant",
                 color="white", fontsize=11, pad=12)

    for i in range(n):
        for j in range(n):
            if i != j:
                p = pmat[i, j]
                txt = f"{p:.3f}" if p >= 0.001 else "<0.001"
                stars = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
                ax.text(j, i, f"{txt}\n{stars}", ha="center", va="center",
                        color="white", fontsize=8, fontweight="bold" if stars else "normal")
            else:
                ax.text(j, i, "—", ha="center", va="center", color="#666688", fontsize=10)

    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("p-value (log scale)", color="#e0e0e0", fontsize=10)
    cbar.ax.yaxis.set_tick_params(color="#e0e0e0")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="#e0e0e0")

    plt.tight_layout()
    path = os.path.join(PLOT_DIR, "9_significance_heatmap.png")
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"  ✅  Saved: {path}")


def plot_confidence_intervals(rewards: dict):
    algos = list(rewards.keys())
    means, lows, highs = [], [], []
    for algo in algos:
        m, lo, hi = bootstrap_ci(rewards[algo])
        means.append(m); lows.append(m - lo); highs.append(hi - m)

    fig, ax = plt.subplots(figsize=(9, 5))
    fig.patch.set_facecolor("#0f0f1a")
    ax.set_facecolor("#1a1a2e")

    colors = ["#00d4ff", "#ff6b9d", "#ffd166", "#06d6a0", "#ef476f"]
    x = range(len(algos))
    ax.barh(x, means, xerr=[lows, highs], color=colors, alpha=0.8,
            edgecolor="white", linewidth=0.5, height=0.55,
            error_kw=dict(ecolor="white", lw=2, capsize=6, capthick=2))

    for i, (m, lo, hi) in enumerate(zip(means, lows, highs)):
        ax.text(m + hi + 0.3, i, f"{m:.2f} [{m-lo:.2f}, {m+hi:.2f}]",
                va="center", fontsize=9, color="white")

    ax.set_yticks(list(x))
    ax.set_yticklabels(algos, fontsize=10)
    ax.set_xlabel("Mean Episodic Reward", fontsize=12)
    ax.set_title("95% Bootstrap Confidence Intervals (10,000 resamples)", fontsize=13, pad=12)
    ax.grid(axis="x", alpha=0.3)
    ax.set_xlim(0, max(m + hi for m, hi in zip(means, highs)) + 4)

    plt.tight_layout()
    path = os.path.join(PLOT_DIR, "10_confidence_intervals.png")
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"  ✅  Saved: {path}")


def write_report(rewards: dict, test_results: dict):
    lines = [
        "=" * 75,
        "STATISTICAL SIGNIFICANCE REPORT",
        "Dataset: 50 evaluation episodes per algorithm | Seed: 42",
        "=" * 75,
        "",
        "─── 95% Bootstrap Confidence Intervals (10,000 resamples) ───────────────",
        f"{'Algorithm':<15} {'Mean':>8} {'95% CI':>20} {'Std':>8}",
        "-" * 55,
    ]
    for algo in ALGO_ORDER:
        if algo not in rewards: continue
        m, lo, hi = bootstrap_ci(rewards[algo])
        std = float(np.std(rewards[algo], ddof=1))
        lines.append(f"{algo:<15} {m:>8.3f} [{lo:>7.3f}, {hi:>7.3f}]   {std:>8.3f}")

    lines += [
        "",
        "─── Pairwise Tests (Wilcoxon or Mann-Whitney U) ─────────────────────────",
        f"{'Pair':<35} {'p-value':>10} {'Sig':>15} {'Cohen d':>10} {'Effect':>12}",
        "-" * 75,
    ]
    for (a, b), res in test_results.items():
        pair  = f"{a} vs {b}"
        lines.append(
            f"{pair:<35} {res['p_w']:>10.4f} {interpret_p(res['p_w']):>15} "
            f"{res['d']:>10.3f} {interpret_d(res['d']):>12}"
        )

    lines += [
        "",
        "─── PPO vs All Others (Focus) ────────────────────────────────────────────",
    ]
    for algo in ALGO_ORDER[1:]:
        key = ("PPO", algo) if ("PPO", algo) in test_results else (algo, "PPO")
        if key not in test_results: continue
        res = test_results[key]
        sig = "SIGNIFICANT ✓" if res["sig"] else "not significant"
        lines.append(f"  PPO vs {algo:<15}: p={res['p_w']:.4f} {sig}, d={res['d']:.3f} ({interpret_d(res['d'])})")

    lines += [
        "",
        "─── Interpretation ───────────────────────────────────────────────────────",
        "  • p < 0.05 means we can reject H₀ (no difference) with 95% confidence.",
        "  • Cohen's d: |d|<0.2=negligible, 0.2–0.5=small, 0.5–0.8=medium, >0.8=large",
        "  • NOTE: These tests use n=50 per algorithm (single seed). Results should",
        "    be validated with multi-seed training (3–5 seeds) for publication.",
        "=" * 75,
    ]

    report_path = os.path.join(OUT_DIR, "statistical_tests.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\n  ✅  Report: {report_path}")
    print("\n" + "\n".join(lines))


# ─────────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n📊 Statistical Significance Analysis")
    print("=" * 50)
    with open(METRICS_PATH) as f:
        metrics = json.load(f)
    rewards      = load_rewards(metrics)
    test_results = run_tests(rewards)
    plot_significance_heatmap(rewards, test_results)
    plot_confidence_intervals(rewards)
    write_report(rewards, test_results)
    print("\n✅ Statistical analysis complete!")
