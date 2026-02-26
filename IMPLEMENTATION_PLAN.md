# Highway RL — Implementation Plan
## Multi-Model Comparison: DQN vs Double DQN vs Dueling DQN vs Rainbow vs PPO

---

## Project Goal
Train and compare 5 RL agents on `highway-v0` environment.
Analyze which algorithm achieves best mean reward, success rate, and speed.
Compare theoretical expected ranking vs actual experimental results.

---

## Environment — Standardized Config (ALL Models)

```python
ENV_CONFIG = {
    "observation": {
        "type": "Kinematics",
        "vehicles_count": 15,       # observe 15 nearest vehicles
        "features": ["x", "y", "vx", "vy", "cos_h", "sin_h"],  # 6 features
        "normalize": True,
        "absolute": False
    },
    "action": {"type": "DiscreteMetaAction"},  # 5 actions
    "lanes_count": 4,
    "vehicles_count": 50,           # 50 traffic vehicles
    "duration": 40,
    "initial_spacing": 2,
    "collision_reward": -1,
    "right_lane_reward": 0.1,
    "high_speed_reward": 0.4,
    "reward_speed_range": [20, 30],
    "normalize_reward": True,
    "simulation_frequency": 15,
    "policy_frequency": 1,
    "other_vehicles_type": "highway_env.vehicle.behavior.IDMVehicle"
}
```

### Left Lane Preference Wrapper (ALL Models)
```python
class LeftLaneRewardWrapper(gym.Wrapper):
    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)
        current_lane = self.env.vehicle.lane_index[2]
        total_lanes  = self.env.config["lanes_count"]
        left_reward  = (total_lanes - 1 - current_lane) / (total_lanes - 1)
        # lane 0 (leftmost) → +0.1 | lane 3 (rightmost) → +0.0
        reward += 0.1 * left_reward
        return obs, reward, done, truncated, info
```

---

## Global Settings

| Setting               | Value                      |
|-----------------------|----------------------------|
| Observation vehicles  | 15                         |
| Traffic vehicles      | 50                         |
| Features              | x, y, vx, vy, cos_h, sin_h |
| Training steps        | 500,000 each               |
| Checkpoint interval   | every 100k steps           |
| Lane preference       | LEFT (custom wrapper)      |
| Device                | cuda (RTX 3050 Ti)         |
| Evaluation episodes   | 50                         |
| Random seed           | 42                         |

---

## Model 1 — Basic DQN (Baseline)

```
Algorithm       DQN (Mnih et al. 2015)
Purpose         Establish baseline — everything compared to this
Implementation  stable-baselines3 DQN
Script          scripts/training/train_dqn.py

Network         MlpPolicy [256, 256] ReLU
Device          cuda

Learning Rate   5e-4
Buffer Size     100,000
Batch Size      64
Gamma           0.99
Tau             1.0  (hard target update)
Target Update   1000 steps
Train Freq      4 steps
Exploration     eps=1.0 → 0.05 over 100k steps

Training Steps  500,000
Checkpoints     100k, 200k, 300k, 400k, 500k
Logs            logs/dqn/
Models          models/dqn/
```

---

## Model 2 — Double DQN

```
Algorithm       Double DQN (van Hasselt et al. 2016)
Purpose         Fix overestimation bias of basic DQN
Implementation  SB3 DQN (double_q=True is default in SB3)
Script          scripts/training/train_double_dqn.py

Network         MlpPolicy [256, 256] ReLU
Device          cuda

Same hyperparams as DQN above

Key Difference:
  DQN:     Q_target = r + γ * max_a  Q_target(s', a)
  Double:  Q_target = r + γ * Q_target(s', argmax_a Q_online(s', a))
           → Online net SELECTS action
           → Target net EVALUATES it
           → Reduces overestimation bias

Training Steps  500,000
Logs            logs/double_dqn/
Models          models/double_dqn/
```

---

## Model 3 — Dueling DQN

```
Algorithm       Dueling DQN (Wang et al. 2016)
Purpose         Separate state value V(s) from action advantage A(s,a)
Implementation  SB3 DQN + CustomDuelingPolicy
Script          scripts/training/train_dueling_dqn.py

Network Architecture (CUSTOM):
  Input (90,)  [15 vehicles × 6 features]
      ↓
  Shared Net   [256, 256, ReLU]
      ↓                   ↓
  Value Stream        Advantage Stream
  Linear(256→128)     Linear(256→128)
  ReLU                ReLU
  Linear(128→1)       Linear(128→n_actions)
  V(s)                A(s,a)
      ↓                   ↓
  Q(s,a) = V(s) + A(s,a) - mean(A(s,a))
                             ↑
                    MUST subtract mean! (stability)

Device          cuda

Learning Rate   5e-4
Buffer Size     100,000
Batch Size      64
Exploration     eps=1.0 → 0.05 over 100k steps

Training Steps  500,000
Logs            logs/dueling_dqn/
Models          models/dueling_dqn/
```

---

## Model 4 — Rainbow DQN (5/6 components)

```
Algorithm       Rainbow DQN (Hessel et al. 2017) — Partial
Purpose         Best DQN variant, performance ceiling
Implementation  SB3 DQN + custom policy + custom buffer
Script          scripts/training/train_rainbow_dqn.py

Components:
  ✅ Double DQN          → native SB3
  ✅ Dueling DQN         → custom policy (same as Dueling)
  ✅ Prioritized Replay  → custom ReplayBuffer (SumTree)
  ✅ Multi-step Returns  → n=3 inside custom buffer
  ✅ Noisy Networks      → NoisyLinear replaces Linear layers
  ❌ Distributional C51  → skipped (too invasive for SB3)

NoisyLinear Layer:
  replaces Linear(in, out) with:
  weight = weight_mu + weight_sigma * noise
  bias   = bias_mu   + bias_sigma   * noise
  → Exploration via learned noise (no epsilon-greedy)

PER SumTree:
  Priority p_i = |TD error|^α + ε
  Sample probability P(i) = p_i / Σp
  Importance weight w_i = (1 / N*P(i))^β
  β anneals from 0.4 → 1.0 over training

Multi-step (n=3):
  G_t = r_t + γ*r_{t+1} + γ²*r_{t+2} + γ³*Q(s_{t+3})
  → Faster reward propagation

Device          cuda

Learning Rate   5e-4
Buffer Size     100,000
Batch Size      64
Gamma           0.99
n_step          3
PER alpha       0.6
PER beta        0.4 → 1.0

Training Steps  500,000
Logs            logs/rainbow_dqn/
Models          models/rainbow_dqn/
```

---

## Model 5 — PPO

```
Algorithm       PPO (Schulman et al. 2017)
Purpose         Policy gradient comparison vs value-based family
Implementation  stable-baselines3 PPO
Script          scripts/training/train_ppo.py

Network         MlpPolicy [256, 256] tanh
Device          cuda

Learning Rate   3e-4      (lower than DQN family)
n_steps         512       (rollout buffer size)
n_epochs        10        (optimization epochs per rollout)
Batch Size      64
Gamma           0.99
GAE Lambda      0.95      (generalized advantage estimation)
Clip Range      0.2       (PPO clipping parameter)
Entropy Coef    0.01      (encourages exploration)
VF Coef         0.5       (value function loss weight)
Max Grad Norm   0.5

No replay buffer (on-policy)
No epsilon-greedy (stochastic policy)

Key Difference vs DQN family:
  DQN  → off-policy, Q-values, replay buffer
  PPO  → on-policy, direct policy, no replay
  Objective: L = E[min(r_t*A_t, clip(r_t, 1±0.2)*A_t)]

Training Steps  500,000
Logs            logs/ppo/
Models          models/ppo/
```

---

## Evaluation Plan

**Script:** `scripts/evaluation/evaluate_all_models.py`

```
For each model (50 episodes):
  - Total episode reward
  - Collision occurred (True/False)
  - Average speed
  - Episode length
  - Lanes visited

Metrics computed:
  - Mean ± Std reward
  - Success rate (% no collision)
  - Mean speed
  - Collision rate
  - Mean episode length

Output: results/metrics.json
```

---

## Visualization Plan

**Script:** `scripts/evaluation/visualize_results.py`

```
Plot 1  Mean Reward Comparison     → bar chart + error bars
Plot 2  Success Rate Comparison    → bar chart (% no collision)
Plot 3  Mean Speed Comparison      → bar chart
Plot 4  Reward Distribution        → box plots per model
Plot 5  Training Curves            → line chart (TensorBoard events)
Plot 6  Radar/Spider Chart         → all metrics, all models

Output: results/plots/
```

---

## Project Structure

```
d:\rl\highway\
├── scripts/
│   ├── training/
│   │   ├── train_dqn.py
│   │   ├── train_double_dqn.py
│   │   ├── train_dueling_dqn.py
│   │   ├── train_rainbow_dqn.py
│   │   └── train_ppo.py
│   └── evaluation/
│       ├── evaluate_all_models.py
│       └── visualize_results.py
├── models/
│   ├── dqn/
│   ├── double_dqn/
│   ├── dueling_dqn/
│   ├── rainbow_dqn/
│   └── ppo/
├── logs/
│   ├── dqn/
│   ├── double_dqn/
│   ├── dueling_dqn/
│   ├── rainbow_dqn/
│   └── ppo/
├── results/
│   ├── plots/
│   ├── metrics.json
│   └── comparison_report.txt
├── IMPLEMENTATION_PLAN.md   ← this file
├── requirements.txt
└── README.md
```

---

## Training Order & Time Estimates (RTX 3050 Ti)

| Order | Model       | Steps | Est. Time |
|-------|-------------|-------|-----------|
| 1st   | DQN         | 500k  | ~20 min   |
| 2nd   | Double DQN  | 500k  | ~20 min   |
| 3rd   | Dueling DQN | 500k  | ~25 min   |
| 4th   | Rainbow     | 500k  | ~30 min   |
| 5th   | PPO         | 500k  | ~20 min   |
| **Total** |         | 2.5M  | ~2 hrs    |

---

## Theoretical Expected Ranking

| Rank | Model       | Reason                                    |
|------|-------------|-------------------------------------------|
| 1st  | Rainbow     | All 5 components combined                 |
| 2nd  | Dueling DQN | Value/Advantage separation                |
| 3rd  | Double DQN  | Fixes overestimation                      |
| 4th  | PPO         | Different paradigm, stable but on-policy  |
| 5th  | DQN         | Baseline, most limitations                |

## What May Actually Happen

| Rank | Model       | Reason                                    |
|------|-------------|-------------------------------------------|
| 1st  | PPO         | Handles 50-vehicle noise well             |
| 2nd  | Rainbow     | Best DQN despite noise                    |
| 3rd  | Double DQN  |                                           |
| 4th  | Dueling DQN |                                           |
| 5th  | DQN         | Weakest baseline                          |

> Theory vs Reality gap = key analysis in report

---

## Execution Checklist

- [ ] Step 1  — Write `train_dqn.py`
- [ ] Step 2  — Write `train_double_dqn.py`
- [ ] Step 3  — Write `train_dueling_dqn.py`
- [ ] Step 4  — Write `train_rainbow_dqn.py`
- [ ] Step 5  — Write `train_ppo.py`
- [ ] Step 6  — Write `evaluate_all_models.py`
- [ ] Step 7  — Write `visualize_results.py`
- [ ] Step 8  — Train all 5 models
- [ ] Step 9  — Run evaluation (50 eps each)
- [ ] Step 10 — Generate all plots
- [ ] Step 11 — Analyze Theory vs Reality
