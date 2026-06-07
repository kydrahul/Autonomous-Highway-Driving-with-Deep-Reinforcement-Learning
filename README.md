# On-Policy vs Off-Policy Deep RL for Autonomous Highway Driving

> Five-algorithm DRL comparison for autonomous highway driving

[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](./LICENSE)
[![highway-env](https://img.shields.io/badge/Env-highway--env-orange.svg)](https://github.com/eleurent/highway-env)
[![Stable Baselines3](https://img.shields.io/badge/RL-Stable--Baselines3-red.svg)](https://stable-baselines3.readthedocs.io/)

---

## Paper

**Title:** On-Policy vs. Off-Policy Deep Reinforcement Learning for Autonomous Highway Driving: A Comprehensive Empirical Study

**Author:** Rahul Barma, IIIT Naya Raipur

**Abstract:**
We investigate a falsifiable hypothesis: *off-policy deep reinforcement learning algorithms suffer from replay buffer contamination by early crash trajectories, causing a distribution shift that degrades learning quality, and that this effect is a primary driver of on-policy PPO's superiority in safety-critical autonomous driving.*

We test this through a controlled comparison of five DRL algorithms on `highway-env`. PPO achieves **96% success rate** (4% collision) vs our 5-component Rainbow DQN's 88% (12% collision) vs vanilla DQN's 20% (80% collision). Statistical significance confirmed via Wilcoxon signed-rank tests (p < 0.05), Cohen's d = 0.871 (large effect).

📄 **[Read the paper (PDF)](./main.pdf)**

---

## Key Results

| Algorithm | Success Rate | Mean Reward | Collision Rate |
|---|---|---|---|
| **PPO** | **96%** | **29.38 ± 4.54** | **4%** |
| Rainbow (5/6) | 88% | 29.21 ± 6.95 | 12% |
| Dueling DQN | 46% | 24.31 ± 10.94 | 54% |
| Double DQN | 20% | 23.79 ± 11.83 | 80% |
| DQN | 20% | 20.78 ± 13.04 | 80% |

> *Results: seed 42, 50 evaluation episodes, 50 IDM vehicles, 500k training steps per algorithm.*

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

- [x] **v1** — 5-algorithm comparison (seed 42), statistical tests, mechanistic analysis
- [ ] **v2** *(in progress)* — Multi-seed (3 seeds), density sweep (10→50 vehicles), SAC baseline
- [ ] **v3** — Buffer ablation, C51 distributional RL, full causal proof of distribution shift

---
*work in progress* 
