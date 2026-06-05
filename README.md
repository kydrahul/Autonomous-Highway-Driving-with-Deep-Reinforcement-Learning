# On-Policy vs Off-Policy Deep RL for Autonomous Highway Driving

> **arXiv preprint** — *under active development (v2 with multi-seed + density sweep results coming)*

[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![License: arXiv](https://img.shields.io/badge/License-arXiv-green.svg)](https://arxiv.org/licenses/nonexclusive-distrib/1.0/license.html)
[![highway-env](https://img.shields.io/badge/Env-highway--env-orange.svg)](https://github.com/eleurent/highway-env)
[![Stable Baselines3](https://img.shields.io/badge/RL-Stable--Baselines3-red.svg)](https://stable-baselines3.readthedocs.io/)

---

## Paper

**Title:** On-Policy vs. Off-Policy Deep Reinforcement Learning for Autonomous Highway Driving: A Comprehensive Empirical Study

**Author:** Rahul Barma, IIIT Naya Raipur

**Abstract:**
We investigate a falsifiable hypothesis: *off-policy deep reinforcement learning algorithms suffer from replay buffer contamination by early crash trajectories, causing a distribution shift that degrades performance in proportion to traffic density, and that this degradation is the primary driver of on-policy PPO's superiority in safety-critical autonomous driving.*

We test this through a controlled comparison of five DRL algorithms on `highway-env`. PPO achieves **96% success rate** (4% collision) vs Rainbow DQN's 88% (12% collision) vs vanilla DQN's 20% (80% collision). Statistical significance confirmed via Wilcoxon signed-rank tests (p < 0.05), Cohen's d = 0.871 (large effect).

📄 **[Read the paper (PDF)](./main.pdf)**

---

## Key Results

| Algorithm | Success Rate | Mean Reward | Collision Rate |
|---|---|---|---|
| **PPO** | **96%** | **29.38 ± 4.54** | **4%** |
| Rainbow DQN | 88% | 29.21 ± 6.95 | 12% |
| Dueling DQN | 46% | 24.31 ± 10.94 | 54% |
| Double DQN | 20% | 23.79 ± 11.83 | 80% |
| DQN | 20% | 20.78 ± 13.04 | 80% |

> *Current results: single seed (42), 50 evaluation episodes, 50 IDM vehicles. Multi-seed + density sweep results in v2.*

---

## Repository Structure

```
├── main.tex                        # Full paper (LaTeX source)
├── main.pdf                        # Compiled paper
├── references.bib                  # Bibliography
├── requirements.txt                # Python dependencies
│
├── scripts/
│   ├── training/
│   │   ├── train_ppo.py            # PPO training
│   │   ├── train_dqn.py            # Vanilla DQN
│   │   ├── train_double_dqn.py     # Double DQN
│   │   ├── train_dueling_dqn.py    # Dueling DQN
│   │   ├── train_rainbow_dqn.py    # Custom Rainbow DQN (PER + NoisyNets + n-step)
│   │   ├── train_sac.py            # Discrete SAC (Christodoulou 2019)
│   │   ├── train_all_seeds.py      # Multi-seed training orchestrator
│   │   ├── train_c51_rainbow.py    # Full Rainbow with C51 distributional RL
│   │   ├── density_sweep.py        # Density experiment (10→50 vehicles)
│   │   └── ablation/
│   │       ├── train_ablation_ladder.py   # Rainbow component ablation
│   │       └── buffer_ablation.py         # Buffer size + periodic wipe experiment
│   │
│   ├── evaluation/
│   │   ├── evaluate_all_models.py  # Evaluate all trained models
│   │   ├── evaluate_multi_seed.py  # Multi-seed evaluation
│   │   ├── render_agents.py        # Render agent behaviour (visual)
│   │   └── visualize_results.py    # Generate result plots
│   │
│   └── analysis/
│       ├── statistical_tests.py    # Wilcoxon, Mann-Whitney, Cohen's d
│       ├── cross_seed_stats.py     # Cross-seed aggregate statistics
│       ├── sample_efficiency.py    # AUC + sample efficiency plots
│       └── why_ppo_wins.py         # Mechanistic analysis plots
│
├── results/
│   ├── metrics.json                # Raw evaluation metrics
│   └── plots/                      # All paper figures (PNG)
│
└── models/                         # Trained model checkpoints
    ├── ppo/
    ├── rainbow_dqn/
    ├── dueling_dqn/
    ├── double_dqn/
    └── dqn/
```

---

## Setup

```bash
# Clone
git clone https://github.com/kydrahul/Autonomous-Highway-Driving-with-Deep-Reinforcement-Learning
cd Autonomous-Highway-Driving-with-Deep-Reinforcement-Learning

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate      # Windows
# source .venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

---

## Reproduce Results

### Train all algorithms (single seed)
```bash
# Individual algorithms
python scripts/training/train_ppo.py
python scripts/training/train_rainbow_dqn.py
python scripts/training/train_dqn.py
python scripts/training/train_double_dqn.py
python scripts/training/train_dueling_dqn.py
```

### Multi-seed training (runs overnight, ~15h)
```bash
python scripts/training/train_all_seeds.py --seeds 42 123 456 --steps 500000
```

### Density sweep experiment
```bash
python scripts/training/density_sweep.py --steps 300000
```

### Buffer ablation (causal proof of distribution shift)
```bash
python scripts/training/ablation/buffer_ablation.py --steps 200000
```

### SAC baseline
```bash
python scripts/training/train_sac.py --seeds 42 123 456
```

### Evaluate trained models
```bash
python scripts/evaluation/evaluate_all_models.py
```

### Generate all paper figures
```bash
python scripts/analysis/statistical_tests.py
python scripts/analysis/why_ppo_wins.py
python scripts/analysis/sample_efficiency.py
python scripts/analysis/cross_seed_stats.py
```

### Compile paper
```bash
pdflatex main.tex
pdflatex main.tex   # run twice to resolve references
```

---

## Algorithm Implementations

| Algorithm | Key Features | File |
|---|---|---|
| DQN | Experience replay, target network, ε-greedy | `train_dqn.py` |
| Double DQN | Decoupled action selection/evaluation | `train_double_dqn.py` |
| Dueling DQN | Value + Advantage streams | `train_dueling_dqn.py` |
| **Rainbow DQN** | Double + Dueling + **PER (SumTree)** + **NoisyLinear** + **n-step(3)** | `train_rainbow_dqn.py` |
| PPO | Clipped surrogate, GAE, entropy regularization | `train_ppo.py` |
| **Discrete SAC** | Off-policy + entropy maximization (Christodoulou 2019) | `train_sac.py` |

---

## Environment Configuration

| Parameter | Value |
|---|---|
| Environment | `highway-v0` (highway-env 1.10) |
| Lanes | 4 |
| Surrounding vehicles | 50 (IDM) |
| Observation | Kinematics — 15 vehicles × 6 features = 90-dim |
| Actions | 5 discrete meta-actions |
| Episode duration | 40 seconds |
| Simulation frequency | 5 Hz |
| Policy frequency | 1 Hz |
| Training steps | 500,000 |
| Random seed | 42 (v1); {42, 123, 456} (v2) |

---

## Roadmap

- [x] **v1** — 5-algorithm comparison, statistical tests, mechanistic analysis
- [ ] **v2** — Multi-seed (3 seeds), density sweep (10→50 vehicles), cross-seed Wilcoxon
- [ ] **v3** — Buffer ablation, SAC baseline, full causal proof of distribution shift hypothesis

---

## Citation

If you use this work, please cite:

```bibtex
@article{barma2025onpolicy,
  title   = {On-Policy vs. Off-Policy Deep Reinforcement Learning for
             Autonomous Highway Driving: A Comprehensive Empirical Study},
  author  = {Barma, Rahul},
  journal = {arXiv preprint},
  year    = {2025},
  url     = {https://github.com/kydrahul/Autonomous-Highway-Driving-with-Deep-Reinforcement-Learning}
}
```
*(Update with arXiv ID once available)*

---

## License

Paper: arXiv.org perpetual, non-exclusive license.
Code: MIT License.
