# 🚗 Highway RL - Autonomous Driving Agent

Deep Reinforcement Learning project for autonomous tactical decision-making in highway scenarios using multiple RL algorithms.

---

## 🛠️ Tech Stack & Frameworks

| Category | Technology | Version / Notes |
|----------|-----------|-----------------|
| **Language** | Python | 3.10+ |
| **RL Environment** | [highway-env](https://highway-env.farama.org/) | Farama Foundation |
| **RL Framework** | [Stable-Baselines3](https://stable-baselines3.readthedocs.io/) | DQN, PPO & variants |
| **Deep Learning** | [PyTorch](https://pytorch.org/) | Neural network backend |
| **Experiment Tracking** | [TensorBoard](https://www.tensorflow.org/tensorboard) | Training curves & metrics |
| **Visualization** | Matplotlib + Pygame | Plots & live rendering |
| **Data Handling** | NumPy + Pandas | Metrics aggregation |
| **Environment API** | [Gymnasium](https://gymnasium.farama.org/) | OpenAI Gym successor |

### 🤖 Algorithms Implemented

| Algorithm | Description |
|-----------|-------------|
| **DQN** | Vanilla Deep Q-Network (baseline) |
| **Double DQN** | Reduces Q-value overestimation |
| **Dueling DQN** | Separate value & advantage streams |
| **PPO** | Proximal Policy Optimization (actor-critic) |
| **Rainbow DQN** | Combined improvements: Double + Dueling + PER + NoisyNet + Multi-step |

---

## 📊 Project Status

✅ **Training Complete:** 213,000 timesteps  
✅ **Success Rate:** 70% (no crashes)  
✅ **Average Reward:** 33.16 ± 10.13  
✅ **Average Speed:** 29.52 m/s  

---

## 🎯 Project Overview

This project trains an autonomous driving agent to navigate highway traffic using **Deep Q-Network (DQN)** reinforcement learning. The agent learns to:
- Perform tactical maneuvers (overtaking, merging, lane-keeping)
- Balance efficiency (speed) with safety (collision avoidance)
- Adapt to dynamic traffic conditions

### Environment
- **Framework:** Highway-env (via Gymnasium)
- **Scenario:** 4-lane highway with 50 vehicles
- **Observation:** Kinematic data (position & velocity of 15 nearest vehicles)
- **Actions:** Lane Left, Lane Right, Accelerate, Brake, Idle

### Model Architecture
- **Algorithm:** Deep Q-Network (DQN)
- **Network:** 2 hidden layers × 256 neurons
- **Learning Rate:** 5e-4
- **Replay Buffer:** 15,000 transitions
- **Discount Factor (γ):** 0.8

---

## 🚀 Quick Start

### Activate Environment
```bash
.\.venv\Scripts\Activate.ps1
```

### Test the Trained Agent (Visual)
```bash
python scripts\evaluation\test_agent.py
```

### Evaluate Performance (Metrics)
```bash
python scripts\evaluation\evaluate_model.py
```

### Compare Different Checkpoints
```bash
python scripts\evaluation\compare_models.py
```

### View Training Curves
```bash
tensorboard --logdir=./logs
```
Then open: http://localhost:6006

---

## 📁 Project Structure

```
d:\rl\highway\
├── 📄 README.md                    # This file
├── 📄 requirements.txt             # Python dependencies
│
├── 📂 scripts/                     # All Python scripts
│   ├── 📂 training/                # Training scripts
│   │   ├── train_dqn.py            # Main training (completed)
│   │   ├── train_advanced.py       # Advanced training (300K steps)
│   │   └── resume_training.py      # Resume from checkpoint
│   │
│   └── 📂 evaluation/              # Evaluation scripts
│       ├── test_agent.py           # Visualize agent (GUI)
│       ├── evaluate_model.py       # Quantitative metrics
│       ├── compare_models.py       # Compare checkpoints
│       └── understand_mdp.py       # Analyze MDP structure
│
├── 📂 docs/                        # Documentation
│   └── read.md                     # Original project brief
│
├── 📂 results/                     # Evaluation results
│   └── evaluation_results.json     # Latest evaluation
│
├── 📂 models/                      # Saved model checkpoints (213 files)
├── 📂 logs/                        # TensorBoard training logs
└── 📂 .venv/                       # Python virtual environment
```

---

## 📈 Training Results

### Performance Metrics
| Metric | Value |
|--------|-------|
| **Success Rate** | 70.0% |
| **Collision Rate** | 30.0% |
| **Average Reward** | 33.16 ± 10.13 |
| **Best Episode** | 39.42 |
| **Average Speed** | 29.52 m/s |
| **Episode Length** | 34.5 steps avg |

### Training Details
- **Total Timesteps:** 213,000
- **Checkpoints Saved:** 213 models (every 1,000 steps)
- **Training Duration:** ~3-4 hours
- **Latest Model:** `dqn_highway_checkpoint_213000_steps.zip`

---

## 🧪 Next Steps

### 1. Advanced Training
Train with optimized hyperparameters:
```bash
python scripts\training\train_advanced.py
```
- Larger network (512 neurons)
- More observation data (15 vehicles)
- 300K timesteps target

### 2. Custom Configurations
Modify environment in training scripts:
```python
config = {
    "lanes_count": 4,        # 2-5 lanes
    "vehicles_count": 50,    # Traffic density
    "duration": 40,          # Episode length
}
```

### 3. Analyze Learning
- Open TensorBoard to view reward curves
- Compare early vs. late training performance
- Identify convergence points

---

## 📚 Key Concepts

### Markov Decision Process (MDP)
- **State:** Kinematic observations (position, velocity of nearby vehicles)
- **Actions:** Discrete driving maneuvers
- **Rewards:** Speed + Lane discipline - Collisions

### Deep Q-Network (DQN)
- **Experience Replay:** Breaks correlation in training data
- **Target Network:** Stabilizes learning
- **ε-greedy Exploration:** Balances exploration vs. exploitation

---

## 🛠️ Troubleshooting

### Resume Training
If training stops unexpectedly:
```bash
python scripts\training\resume_training.py
```

### View Specific Checkpoint
Edit `scripts\evaluation\test_agent.py` to load a specific model:
```python
model = DQN.load("models/dqn_highway_checkpoint_100000_steps", env=env)
```

### Adjust Training Speed
Reduce timesteps in training scripts for faster testing:
```python
TIME_STEPS = 50000  # Instead of 200000
```

---

## 📊 Expected Outcomes

✅ **Autonomous Navigation:** >90% success rate (target)  
✅ **Tactical Maneuvers:** Successful overtaking, merging, lane-keeping  
✅ **Safety:** Reduced collision rate  
✅ **Efficiency:** High-speed navigation (20-30 m/s)  
✅ **Adaptability:** Performance across varying traffic densities  

**Current Achievement:** 70% success rate at 213K steps

---

## 🔗 Resources

- **Highway-env Documentation:** https://highway-env.farama.org/
- **Stable-Baselines3 Docs:** https://stable-baselines3.readthedocs.io/
- **DQN Paper:** [Playing Atari with Deep Reinforcement Learning](https://arxiv.org/abs/1312.5602)

---

## 📝 License

Educational project for reinforcement learning research.

---

**Last Updated:** February 10, 2026  
**Status:** ✅ Training Complete | 🧪 Ready for Evaluation
