"""
Multi-Seed Training Orchestrator
==================================
Trains ALL 5 algorithms across multiple random seeds to enable
statistically robust results for publication.

This addresses the core reviewer concern:
  "Results from a single seed are not reproducible."

Usage:
    python scripts/training/train_all_seeds.py [--seeds 42 123 456] [--steps 300000]

Saves models to:
    models/{algo}/seed_{seed}/   e.g. models/ppo/seed_42/ppo_final.zip

After training, run:
    python scripts/evaluation/evaluate_multi_seed.py --seeds 42 123 456 --episodes 150
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path
from datetime import datetime

ALGORITHMS = {
    "ppo":         "scripts/training/train_ppo.py",
    "a2c":         "scripts/training/train_a2c.py",
    "dqn":         "scripts/training/train_dqn.py",
    "double_dqn":  "scripts/training/train_double_dqn.py",
    "dueling_dqn": "scripts/training/train_dueling_dqn.py",
    "rainbow_dqn": "scripts/training/train_rainbow_dqn.py",
}

LOG_DIR = "results/analysis/multi_seed_training_log.txt"
os.makedirs("results/analysis", exist_ok=True)


def log(msg: str):
    ts = datetime.now().strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line)
    with open(LOG_DIR, "a") as f:
        f.write(line + "\n")


def patch_seed_and_paths(script_path: str, seed: int, total_steps: int) -> str:
    """
    Read the training script and patch:
      - seed=42 → seed=<seed>
      - MODEL_DIR/LOG_DIR to include seed subfolder
      - TOTAL_STEPS if specified
    Returns path to the patched temporary script.
    """
    with open(script_path) as f:
        code = f.read()

    algo_name = Path(script_path).stem.replace("train_", "")

    # Patch seed
    code = code.replace("seed=42", f"seed={seed}")

    # Patch model/log dirs to include seed subfolder
    code = code.replace(
        f'MODEL_DIR = "models/{algo_name}"',
        f'MODEL_DIR = "models/{algo_name}/seed_{seed}"',
    )
    code = code.replace(
        f'MODEL_DIR = "models/rainbow_dqn"',
        f'MODEL_DIR = "models/rainbow_dqn/seed_{seed}"',
    ) if "rainbow" in script_path else code
    code = code.replace(
        f'LOG_DIR   = "logs/{algo_name}"',
        f'LOG_DIR   = "logs/{algo_name}/seed_{seed}"',
    )
    code = code.replace(
        f'LOG_DIR   = "logs/rainbow_dqn"',
        f'LOG_DIR   = "logs/rainbow_dqn/seed_{seed}"',
    ) if "rainbow" in script_path else code

    # Patch training steps
    if total_steps:
        code = code.replace(
            "TOTAL_STEPS = 500_000",
            f"TOTAL_STEPS = {total_steps:_}",
        )

    # Write patched script to temp location
    tmp_dir = "scripts/training/_tmp"
    os.makedirs(tmp_dir, exist_ok=True)
    tmp_path = os.path.join(tmp_dir, f"{algo_name}_seed{seed}.py")
    with open(tmp_path, "w") as f:
        f.write(code)
    return tmp_path


def train_one(algo: str, script: str, seed: int, total_steps: int) -> bool:
    log(f">> Training {algo.upper()} | seed={seed} | steps={total_steps:,}")
    tmp_script = patch_seed_and_paths(script, seed, total_steps)

    result = subprocess.run(
        [sys.executable, tmp_script],
        capture_output=False,
        text=True,
    )

    if result.returncode == 0:
        log(f"[OK] {algo.upper()} seed={seed} DONE")
        return True
    else:
        log(f"[FAIL] {algo.upper()} seed={seed} FAILED (exit code {result.returncode})")
        return False


# ─────────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train all RL agents across multiple random seeds."
    )
    parser.add_argument("--seeds",   type=int, nargs="+", default=[42, 123, 456],
                        help="Random seeds to use (default: 42 123 456)")
    parser.add_argument("--algos",   type=str, nargs="+", default=list(ALGORITHMS.keys()),
                        help="Algorithms to train (default: all)")
    parser.add_argument("--steps",   type=int, default=300_000,
                        help="Training steps per algorithm (default: 300,000)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print plan without running training")
    args = parser.parse_args()

    selected = {k: v for k, v in ALGORITHMS.items() if k in args.algos}

    print("\n" + "=" * 65)
    print("  MULTI-SEED TRAINING ORCHESTRATOR")
    print("=" * 65)
    print(f"  Algorithms : {', '.join(selected.keys())}")
    print(f"  Seeds      : {args.seeds}")
    print(f"  Steps each : {args.steps:,}")
    print(f"  Total runs : {len(selected) * len(args.seeds)}")
    print(f"  ETA (est.) : ~{len(selected) * len(args.seeds) * args.steps / 5000 / 60:.0f} mins (GPU)")
    print("=" * 65)

    if args.dry_run:
        print("\n[DRY RUN] Would train:")
        for algo in selected:
            for seed in args.seeds:
                print(f"  {algo:>12} | seed={seed} | → models/{algo}/seed_{seed}/")
        sys.exit(0)

    stats = {"success": 0, "failed": 0, "skipped": 0}
    start = datetime.now()

    for algo, script in selected.items():
        if not os.path.exists(script):
            log(f"[WARN] Script not found: {script} — skipping {algo}")
            stats["skipped"] += len(args.seeds)
            continue
        for seed in args.seeds:
            ok = train_one(algo, script, seed, args.steps)
            stats["success" if ok else "failed"] += 1

    elapsed = datetime.now() - start
    print("\n" + "=" * 65)
    print(f"  Training Complete in {elapsed}")
    print(f"  [OK] Successful : {stats['success']}")
    print(f"  [FAIL] Failed     : {stats['failed']}")
    print(f"  [WARN] Skipped    : {stats['skipped']}")
    print("=" * 65)
    print(f"\n  Next step:")
    print(f"  python scripts/evaluation/evaluate_multi_seed.py "
          f"--seeds {' '.join(map(str, args.seeds))} --episodes 150")
