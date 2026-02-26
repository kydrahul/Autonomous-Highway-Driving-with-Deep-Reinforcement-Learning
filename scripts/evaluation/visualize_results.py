"""
Visualize Results
==================
Reads results/metrics.json (produced by evaluate_all_models.py)
and generates 6 comparison plots saved to results/plots/.

Plots:
  1. Mean Reward Comparison         (bar chart + error bars)
  2. Success Rate Comparison        (bar chart)
  3. Mean Speed Comparison          (bar chart)
  4. Reward Distribution            (box plots)
  5. Radar / Spider Chart           (all metrics, all models)
  6. Training Curves                (from TensorBoard event files)

Run:
    python scripts/evaluation/visualize_results.py
"""

import os
import json
import glob
import numpy as np
import matplotlib
matplotlib.use("Agg")           # headless rendering — no display needed
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from math import pi

# ── Paths ───────────────────────────────────────────────────────────────────────
RESULTS_DIR  = "results"
PLOTS_DIR    = os.path.join(RESULTS_DIR, "plots")
METRICS_PATH = os.path.join(RESULTS_DIR, "metrics.json")
os.makedirs(PLOTS_DIR, exist_ok=True)

# ── Colour palette — one colour per model ───────────────────────────────────────
COLORS = {
    "DQN":         "#4C72B0",
    "Double DQN":  "#DD8452",
    "Dueling DQN": "#55A868",
    "Rainbow DQN": "#C44E52",
    "PPO":         "#8172B2",
}

MODEL_ORDER = ["DQN", "Double DQN", "Dueling DQN", "Rainbow DQN", "PPO"]

plt.rcParams.update({
    "figure.dpi":     150,
    "font.size":      11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "legend.fontsize":10,
})


# ── Helpers ──────────────────────────────────────────────────────────────────────
def load_metrics() -> dict:
    with open(METRICS_PATH, "r") as f:
        return json.load(f)


def available_models(data: dict) -> list:
    return [m for m in MODEL_ORDER if m in data]


def bar_chart(ax, models, values, errors=None, ylabel="", title="", color_map=COLORS):
    x      = np.arange(len(models))
    colors = [color_map.get(m, "steelblue") for m in models]
    bars   = ax.bar(x, values, color=colors, width=0.55, edgecolor="white", linewidth=0.8)

    if errors is not None:
        ax.errorbar(x, values, yerr=errors, fmt="none", color="black",
                    capsize=5, linewidth=1.5)

    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=20, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.grid(True, linestyle="--", alpha=0.6)
    ax.set_axisbelow(True)

    # Value labels on bars
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + (max(values) * 0.01),
                f"{v:.3f}", ha="center", va="bottom", fontsize=9)
    return bars


# ══════════════════════════════════════════════════════════════════════════════════
# Plot 1 — Mean Reward Comparison
# ══════════════════════════════════════════════════════════════════════════════════
def plot_mean_reward(data: dict):
    models = available_models(data)
    values = [data[m]["mean_reward"] for m in models]
    errors = [data[m]["std_reward"]  for m in models]

    fig, ax = plt.subplots(figsize=(9, 5))
    bar_chart(ax, models, values, errors,
              ylabel="Mean Episode Reward",
              title="Mean Reward Comparison (± std, 50 episodes)")

    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "1_mean_reward.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  ✅ Saved: {path}")


# ══════════════════════════════════════════════════════════════════════════════════
# Plot 2 — Success Rate Comparison
# ══════════════════════════════════════════════════════════════════════════════════
def plot_success_rate(data: dict):
    models = available_models(data)
    values = [data[m]["success_rate"] * 100 for m in models]

    fig, ax = plt.subplots(figsize=(9, 5))
    bar_chart(ax, models, values,
              ylabel="Success Rate (%)",
              title="Success Rate — % Episodes Without Collision")
    ax.set_ylim(0, 110)
    ax.axhline(y=100, color="green", linestyle="--", alpha=0.4, linewidth=1)

    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "2_success_rate.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  ✅ Saved: {path}")


# ══════════════════════════════════════════════════════════════════════════════════
# Plot 3 — Mean Speed Comparison
# ══════════════════════════════════════════════════════════════════════════════════
def plot_mean_speed(data: dict):
    models = available_models(data)
    values = [data[m]["mean_speed"] for m in models]
    errors = [data[m]["std_speed"]  for m in models]

    fig, ax = plt.subplots(figsize=(9, 5))
    bar_chart(ax, models, values, errors,
              ylabel="Mean Speed (m/s)",
              title="Mean Speed Comparison (target range: 20–30 m/s)")
    ax.axhspan(20, 30, alpha=0.08, color="green", label="Target speed range")
    ax.legend()

    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "3_mean_speed.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  ✅ Saved: {path}")


# ══════════════════════════════════════════════════════════════════════════════════
# Plot 4 — Reward Distribution (Box Plots)
# ══════════════════════════════════════════════════════════════════════════════════
def plot_reward_distribution(data: dict):
    models   = available_models(data)
    raw_data = [data[m]["raw_rewards"] for m in models]
    colors   = [COLORS.get(m, "steelblue") for m in models]

    fig, ax = plt.subplots(figsize=(10, 5))
    bp = ax.boxplot(raw_data, patch_artist=True, notch=False,
                    medianprops=dict(color="white", linewidth=2))

    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)

    ax.set_xticks(range(1, len(models) + 1))
    ax.set_xticklabels(models, rotation=20, ha="right")
    ax.set_ylabel("Episode Reward")
    ax.set_title("Reward Distribution Across 50 Episodes")
    ax.yaxis.grid(True, linestyle="--", alpha=0.6)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Legend
    patches = [mpatches.Patch(color=COLORS.get(m, "steelblue"), label=m) for m in models]
    ax.legend(handles=patches, loc="upper left", framealpha=0.8)

    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "4_reward_distribution.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  ✅ Saved: {path}")


# ══════════════════════════════════════════════════════════════════════════════════
# Plot 5 — Radar / Spider Chart
# ══════════════════════════════════════════════════════════════════════════════════
def plot_radar(data: dict):
    models = available_models(data)

    # Metrics to include (all normalised 0–1, higher = better)
    metric_labels = ["Mean Reward", "Success Rate", "Mean Speed", "Ep Length"]

    def get_values(m):
        r = data[m]
        return [
            r["mean_reward"],
            r["success_rate"],
            r["mean_speed"],
            r["mean_ep_length"],
        ]

    raw = {m: get_values(m) for m in models}

    # Normalise each metric to [0, 1] across models
    n_metrics = len(metric_labels)
    raw_array = np.array([raw[m] for m in models])
    col_min   = raw_array.min(axis=0)
    col_max   = raw_array.max(axis=0)
    col_range = np.where((col_max - col_min) == 0, 1.0, col_max - col_min)
    norm      = (raw_array - col_min) / col_range

    # Angles for each metric
    angles = [n / float(n_metrics) * 2 * pi for n in range(n_metrics)]
    angles += angles[:1]    # close the polygon

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))

    for i, model in enumerate(models):
        values = list(norm[i]) + [norm[i][0]]   # close polygon
        color  = COLORS.get(model, "steelblue")
        ax.plot(angles, values, "o-", linewidth=2, color=color, label=model)
        ax.fill(angles, values, alpha=0.08, color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metric_labels, size=11)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0.25", "0.50", "0.75", "1.00"], size=8)
    ax.set_title("Normalised Performance Comparison\n(All Metrics, All Models)",
                 size=13, pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))

    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "5_radar_chart.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✅ Saved: {path}")


# ══════════════════════════════════════════════════════════════════════════════════
# Plot 6 — Training Curves (from TensorBoard event files)
# ══════════════════════════════════════════════════════════════════════════════════
def _load_merged_runs(log_dir: str, tag: str = "rollout/ep_rew_mean"):
    """
    Loads and merges all TensorBoard run directories for one model.

    Strategy — handles two scenarios:
      1. Overlapping runs (caused by reset_num_timesteps=True on resume):
         Each run starts from step 0.  We take run-1 in full, then only the
         NEW portion of run-2 that extends beyond run-1's final step, etc.
      2. Sequential runs (correct behaviour after bug-fix):
         run-2 starts where run-1 ended → concatenated seamlessly.

    Returns (steps_array, values_array) sorted by step, or (None, None).
    """
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except ImportError:
        return None, None

    run_dirs = sorted(glob.glob(os.path.join(log_dir, "*")))
    if not run_dirs:
        return None, None

    runs = []
    for rd in run_dirs:
        try:
            ea = EventAccumulator(rd)
            ea.Reload()
            scalars = ea.Tags().get("scalars", [])
            t = tag if tag in scalars else (scalars[0] if scalars else None)
            if t is None:
                continue
            events = ea.Scalars(t)
            if not events:
                continue
            steps  = np.array([e.step  for e in events])
            values = np.array([e.value for e in events])
            if len(steps):
                runs.append((steps, values))
        except Exception:
            pass

    if not runs:
        return None, None

    # Sort runs by their first logged step
    runs.sort(key=lambda r: r[0][0])

    merged_steps:  list = []
    merged_values: list = []
    covered_up_to = -1

    for steps, values in runs:
        # Only keep the portion of this run that extends beyond what we have
        mask = steps > covered_up_to
        new_steps  = steps[mask]
        new_values = values[mask]
        if len(new_steps):
            merged_steps.extend(new_steps.tolist())
            merged_values.extend(new_values.tolist())
            covered_up_to = int(new_steps[-1])

    if not merged_steps:
        return None, None

    return np.array(merged_steps), np.array(merged_values)


def plot_training_curves():
    """
    Reads TensorBoard event files from logs/ directories and plots merged
    training curves for all models.  Merges multiple run directories for
    the same model so that resumed training (after a crash or checkpoint)
    appears as one continuous curve.

    Requires tensorboard: pip install tensorboard
    """
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator  # noqa: F401
    except ImportError:
        print("  ⚠️  tensorboard not installed — skipping training curves plot")
        print("       Run: pip install tensorboard")
        return

    log_dirs = {
        "DQN":         "logs/dqn",
        "Double DQN":  "logs/double_dqn",
        "Dueling DQN": "logs/dueling_dqn",
        "Rainbow DQN": "logs/rainbow_dqn",
        "PPO":         "logs/ppo",
    }

    fig, ax = plt.subplots(figsize=(11, 6))
    found_any = False

    for model_name, log_dir in log_dirs.items():
        if not os.path.exists(log_dir):
            continue

        steps, values = _load_merged_runs(log_dir)
        if steps is None:
            print(f"  ⚠️  No usable TensorBoard logs found for {model_name}")
            continue

        color = COLORS.get(model_name, "steelblue")
        ax.plot(steps, values, linewidth=1.8, color=color,
                label=f"{model_name} (max {int(steps[-1]):,})", alpha=0.9)
        found_any = True
        print(f"     {model_name}: {len(steps)} points, "
              f"steps 0 → {int(steps[-1]):,}")

    if not found_any:
        print("  ⚠️  No TensorBoard logs found — run training first")
        plt.close(fig)
        return

    ax.set_xlabel("Training Steps")
    ax.set_ylabel("Mean Episode Reward")
    ax.set_title("Training Curves — Mean Episode Reward vs Training Steps")
    ax.legend()
    ax.yaxis.grid(True, linestyle="--", alpha=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "6_training_curves.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  ✅ Saved: {path}")


# ── Main ─────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 55)
    print("  Generating Visualizations")
    print("=" * 55)

    if not os.path.exists(METRICS_PATH):
        print(f"\n❌ metrics.json not found at {METRICS_PATH}")
        print("   Run evaluate_all_models.py first.\n")
        exit(1)

    data = load_metrics()
    print(f"\n  Models found in results: {list(data.keys())}\n")

    print("  Generating plots...")
    plot_mean_reward(data)
    plot_success_rate(data)
    plot_mean_speed(data)
    plot_reward_distribution(data)
    plot_radar(data)
    plot_training_curves()

    print(f"\n✅ All plots saved to: {PLOTS_DIR}/")
    print(f"   Open the folder to view all 6 comparison charts.")
