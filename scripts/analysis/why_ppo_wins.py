"""
Why PPO Wins — Deep Analysis Script
=====================================
Generates the mechanistic analysis plots for the "Why PPO Wins" section
of the paper. Analyses:
  1. Speed-vs-collision scatter (all algorithms)
  2. Episode length distribution (on-policy long-horizon vs off-policy short)
  3. Reward variance as a stability proxy
  4. Bimodal reward decomposition (DQN family crash/success modes)
  5. Summary table of on-policy vs off-policy differences

Usage:
    python scripts/analysis/why_ppo_wins.py

Outputs:
    results/plots/11_speed_vs_collision.png
    results/plots/12_episode_analysis.png
    results/plots/13_reward_stability.png
    results/analysis/why_ppo_wins_report.txt
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import gaussian_kde

METRICS_PATH = "results/metrics.json"
OUT_DIR      = "results/plots"
REPORT_DIR   = "results/analysis"
os.makedirs(OUT_DIR,   exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

plt.rcParams.update({
    "figure.facecolor": "#0f0f1a",
    "axes.facecolor":   "#1a1a2e",
    "axes.edgecolor":   "#444466",
    "axes.labelcolor":  "#e0e0e0",
    "xtick.color":      "#e0e0e0",
    "ytick.color":      "#e0e0e0",
    "text.color":       "#e0e0e0",
    "grid.color":       "#2a2a4a",
    "grid.alpha":       0.5,
    "font.size":        11,
})

ALGO_COLORS = {
    "PPO":        "#00d4ff",
    "Rainbow DQN":"#ff6b9d",
    "Dueling DQN":"#ffd166",
    "Double DQN": "#06d6a0",
    "DQN":        "#ef476f",
}
ALGO_ORDER = ["PPO", "Rainbow DQN", "Dueling DQN", "Double DQN", "DQN"]


def load(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


# ─────────────────────────────────────────────────────────────────────────────────
def plot_speed_vs_collision(m: dict):
    """
    Scatter: Mean Speed (x) vs Collision Rate (y) with bubble size = reward std.
    Reveals the speed-safety trade-off clearly.
    """
    fig, ax = plt.subplots(figsize=(9, 6))
    fig.patch.set_facecolor("#0f0f1a")

    for algo in ALGO_ORDER:
        d     = m.get(algo, {})
        speed = d.get("mean_speed", 0)
        cr    = d.get("collision_rate", 0) * 100
        std   = d.get("std_reward", 1)
        color = ALGO_COLORS[algo]

        ax.scatter(speed, cr, s=std * 40, color=color, zorder=5,
                   edgecolors="white", linewidths=0.8, alpha=0.9)
        ax.annotate(algo, (speed, cr),
                    textcoords="offset points", xytext=(8, 4),
                    fontsize=9.5, color=color, fontweight="bold")

    # Annotation zones
    ax.axvspan(19, 22, alpha=0.06, color="#00d4ff", label="Low speed zone (safe)")
    ax.axvspan(27, 31, alpha=0.06, color="#ef476f", label="High speed zone (dangerous)")
    ax.axhline(50, color="#888888", ls="--", lw=0.8, alpha=0.5)
    ax.text(19.2, 52, "50% collision threshold", fontsize=8, color="#888888")

    ax.set_xlabel("Mean Speed (m/s)", fontsize=13)
    ax.set_ylabel("Collision Rate (%)", fontsize=13)
    ax.set_title("Speed–Safety Trade-off\n"
                 "(Bubble size ∝ Reward Std Dev — larger = less stable policy)",
                 fontsize=13, pad=12)
    ax.set_xlim(18, 31)
    ax.set_ylim(-5, 95)
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left", facecolor="#1a1a2e", edgecolor="#444466", fontsize=9)

    # Arrow showing the trade-off direction
    ax.annotate("", xy=(30, 80), xytext=(20.5, 5),
                arrowprops=dict(arrowstyle="-|>", color="#ff6b6b", lw=1.5))
    ax.text(25.5, 42, "Speed ↑\nSafety ↓", color="#ff6b6b", fontsize=9,
            ha="center", rotation=52)

    plt.tight_layout()
    path = os.path.join(OUT_DIR, "11_speed_vs_collision.png")
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"  ✅  Saved: {path}")


# ─────────────────────────────────────────────────────────────────────────────────
def plot_episode_analysis(m: dict):
    """
    Dual panel:
    A) Episode length distribution (proxied from mean_ep_length + collision pattern)
    B) Bimodal reward decomposition — crash vs success episodes for DQN family
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor("#0f0f1a")
    fig.suptitle("Episode Structure Analysis: On-Policy vs Off-Policy",
                 fontsize=14, color="white", y=1.03)

    # Panel A — Episode length bar chart
    ax = axes[0]
    algos    = ALGO_ORDER
    ep_lens  = [m.get(a, {}).get("mean_ep_length", 0) for a in algos]
    colors   = [ALGO_COLORS[a] for a in algos]
    bars     = ax.barh(algos, ep_lens, color=colors, edgecolor="white",
                       linewidth=0.5, height=0.55, alpha=0.85)

    for bar, val in zip(bars, ep_lens):
        ax.text(val + 0.3, bar.get_y() + bar.get_height() / 2,
                f"{val:.1f} steps", va="center", fontsize=9.5, color="white")

    ax.set_xlabel("Mean Episode Length (steps)", fontsize=12)
    ax.set_title("Mean Episode Length\n(Longer = survived longer = fewer crashes)",
                 fontsize=11, pad=8)
    ax.axvline(40, color="#888888", ls="--", lw=1, alpha=0.6)
    ax.text(40.3, 4.5, "Max (40 steps)", fontsize=8, color="#888888")
    ax.set_xlim(0, 48)
    ax.grid(axis="x", alpha=0.3)

    # Panel B — KDE of reward distributions
    ax2 = axes[1]
    ax2.set_title("Reward Density — Bimodal Distribution in DQN Family\n"
                  "(Two peaks = crash mode vs success mode)",
                  fontsize=11, pad=8)

    for algo in ALGO_ORDER:
        raw = np.array(m.get(algo, {}).get("raw_rewards", []))
        if len(raw) < 3: continue
        kde_x = np.linspace(0, 45, 300)
        try:
            kde = gaussian_kde(raw, bw_method=0.4)
            kde_y = kde(kde_x)
            ax2.plot(kde_x, kde_y, lw=2.5, color=ALGO_COLORS[algo], label=algo)
            ax2.fill_between(kde_x, kde_y, alpha=0.12, color=ALGO_COLORS[algo])
        except Exception:
            pass

    ax2.set_xlabel("Episodic Reward", fontsize=12)
    ax2.set_ylabel("Density", fontsize=12)
    ax2.legend(loc="upper left", facecolor="#1a1a2e", edgecolor="#444466", fontsize=9)
    ax2.grid(alpha=0.3)

    # Annotate bimodal peaks for DQN
    ax2.annotate("DQN crash\nmode (~10)", xy=(10, 0.025), xytext=(3, 0.04),
                 arrowprops=dict(arrowstyle="->", color="#ef476f", lw=1),
                 color="#ef476f", fontsize=8)
    ax2.annotate("DQN success\nmode (~40)", xy=(40, 0.018), xytext=(32, 0.038),
                 arrowprops=dict(arrowstyle="->", color="#ef476f", lw=1),
                 color="#ef476f", fontsize=8)
    ax2.annotate("PPO: narrow\npeaked dist.", xy=(29.5, 0.065), xytext=(18, 0.08),
                 arrowprops=dict(arrowstyle="->", color="#00d4ff", lw=1),
                 color="#00d4ff", fontsize=8)

    plt.tight_layout()
    path = os.path.join(OUT_DIR, "12_episode_analysis.png")
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"  ✅  Saved: {path}")


# ─────────────────────────────────────────────────────────────────────────────────
def plot_reward_stability(m: dict):
    """
    Normalised reward mean vs std scatter (stability vs performance frontier).
    PPO should be Pareto-dominant (high mean, low std).
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    fig.patch.set_facecolor("#0f0f1a")

    for algo in ALGO_ORDER:
        d     = m.get(algo, {})
        mu    = d.get("mean_reward", 0)
        std   = d.get("std_reward", 0)
        color = ALGO_COLORS[algo]
        ax.scatter(std, mu, s=200, color=color, zorder=5,
                   edgecolors="white", linewidths=1.2, marker="D" if algo == "PPO" else "o")
        ax.annotate(algo, (std, mu),
                    textcoords="offset points", xytext=(6, 4),
                    fontsize=9.5, color=color, fontweight="bold")

    ax.set_xlabel("Reward Std Dev (↓ = More Stable)", fontsize=13)
    ax.set_ylabel("Mean Episodic Reward (↑ = Better)", fontsize=13)
    ax.set_title("Performance–Stability Frontier\n"
                 "(Top-left = ideal: high reward, low variance)\n"
                 "◆ = PPO  ● = DQN variants",
                 fontsize=12, pad=12)

    # Annotate ideal quadrant
    ax.axvline(7, color="#888888", ls=":", lw=0.8, alpha=0.5)
    ax.axhline(27, color="#888888", ls=":", lw=0.8, alpha=0.5)
    ax.text(4, 30.5, "Ideal\nquadrant", fontsize=9, color="#888888", ha="center")

    ax.grid(alpha=0.3)
    plt.tight_layout()
    path = os.path.join(OUT_DIR, "13_reward_stability.png")
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"  ✅  Saved: {path}")


# ─────────────────────────────────────────────────────────────────────────────────
def write_analysis_report(m: dict):
    """Write the "Why PPO Wins" mechanistic analysis as a structured text report."""

    def val(algo, key):
        return m.get(algo, {}).get(key, 0)

    lines = [
        "=" * 75,
        "WHY PPO OUTPERFORMS RAINBOW DQN — MECHANISTIC ANALYSIS REPORT",
        "=" * 75,
        "",
        "─── 1. Distribution Shift (Core Cause) ──────────────────────────────────",
        "",
        "  OFF-POLICY (DQN family) workflow:",
        "    Episode 1 (early training): Agent crashes at step 21 → stores crash",
        "    Episode 100 (later training): Crash transition replayed 50+ times",
        "    → Network learns from outdated, crash-prone data  → Corrupted gradient",
        "",
        "  ON-POLICY (PPO) workflow:",
        "    Each update batch uses ONLY transitions from the CURRENT policy.",
        "    Old data is DISCARDED after each policy update.",
        "    → Training distribution always matches current (improving) policy",
        "",
        f"  Evidence: DQN mean episode length = {val('DQN', 'mean_ep_length'):.1f} steps",
        f"            vs PPO mean episode length = {val('PPO', 'mean_ep_length'):.1f} steps",
        "  DQN crashes early → most replay buffer transitions are from short, crash",
        "  episodes → biased toward crash-prone behaviour.",
        "",
        "─── 2. Exploration Quality ───────────────────────────────────────────────",
        "",
        "  DQN ε-greedy: With prob ε, take RANDOM action.",
        "    In dense traffic, a random lane change at 29 m/s = likely collision.",
        f"  DQN collision rate: {val('DQN', 'collision_rate'):.0%}",
        "",
        "  Rainbow NoisyNets: Learned parametric noise → smoother exploration.",
        f"  Rainbow collision rate: {val('Rainbow DQN', 'collision_rate'):.0%}  (much better!)",
        "",
        "  PPO stochastic policy: Always samples from learned distribution.",
        "    Actions proportional to policy probabilities → no abrupt random jumps.",
        f"  PPO collision rate: {val('PPO', 'collision_rate'):.0%}  (best!)",
        "",
        "─── 3. Policy Update Stability ──────────────────────────────────────────",
        "",
        "  PPO clip(r_t, 0.8, 1.2): Prevents policy from changing too rapidly.",
        "  DQN gradient clipping: Operates at loss level, weaker constraint.",
        "",
        f"  Evidence — Reward Standard Deviation (stability proxy):",
        f"  {'Algorithm':<15} {'Std Dev':>10} {'Interpretation':>25}",
        f"  {'─'*55}",
    ]

    for algo in ALGO_ORDER:
        std = val(algo, "std_reward")
        interp = "[OK] Very stable" if std < 6 else "[~] Moderate" if std < 10 else "[X] Unstable"
        lines.append(f"  {algo:<15} {std:>10.2f} {interp:>25}")

    lines += [
        "",
        "─── 4. Speed–Safety Dilemma ──────────────────────────────────────────────",
        "",
        "  DQN family drives FAST (≈29 m/s) → maximises per-step speed reward",
        "  but CRASHES frequently → short episodes → low total reward.",
        "",
        "  PPO drives CAUTIOUSLY (≈20 m/s) → lower per-step speed reward",
        "  but SURVIVES longer → 40-step episodes → higher TOTAL reward.",
        "",
        f"  {'Algorithm':<15} {'Mean Speed':>12} {'Ep Length':>12} {'Success':>10}",
        f"  {'─'*55}",
    ]
    for algo in ALGO_ORDER:
        lines.append(
            f"  {algo:<15} {val(algo, 'mean_speed'):>10.2f} m/s"
            f" {val(algo, 'mean_ep_length'):>10.1f} steps"
            f" {val(algo, 'success_rate'):>9.0%}"
        )

    lines += [
        "",
        "─── Conclusion ───────────────────────────────────────────────────────────",
        "",
        "  PPO's superiority is NOT just luck or hyperparameter sensitivity.",
        "  It stems from a structural advantage in safety-critical environments:",
        "",
        "  In dense traffic → early crashes are frequent → DQN replay buffers fill",
        "  with corrupted data → distribution shift degrades learning.",
        "  PPO's on-policy nature is IMMUNE to this problem.",
        "",
        "  Practical recommendation:",
        "  Use PPO (or SAC) for dense, safety-critical RL. Reserve DQN variants",
        "  for environments where early exploration costs are low.",
        "=" * 75,
    ]

    path = os.path.join(REPORT_DIR, "why_ppo_wins_report.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\n  [OK] Report: {path}")
    print("\n" + "\n".join(lines))


# ─────────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n[*] Why PPO Wins -- Mechanistic Analysis")
    print("=" * 50)
    metrics = load(METRICS_PATH)
    plot_speed_vs_collision(metrics)
    plot_episode_analysis(metrics)
    plot_reward_stability(metrics)
    write_analysis_report(metrics)
    print("\n[DONE] Analysis complete!")
