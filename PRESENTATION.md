# Autonomous Highway Driving with Deep Reinforcement Learning

**Project:** Multi-algorithm RL comparison on highway-v0  
**Algorithms:** DQN · Double DQN · Dueling DQN · Rainbow DQN · PPO  
**Environment:** Highway-env (Gymnasium)  
**RL Framework:** Stable-Baselines3
**Logging:** TensorBoard


Problem: 1. Optimise the Speed
2. Stay Left
3. Avoid Collision


### Configuration
| Parameter | Value | 
|-----------|-------|
| Lanes | 4 | 
| Total vehicles | 50 |
| Ego observes | 15 nearest | 
| Features per vehicle | 6 (x, y, vx, vy, cos_h, sin_h) |
| Episode duration | 40 seconds | 
| Traffic behavior | IDM (Intelligent Driver Model) |

### Action Space (5 discrete actions)
| ID | Action | Effect |
|----|--------|--------|
| 0 | LANE LEFT | Move to left lane |
| 1 | IDLE | Maintain speed/lane |
| 2 | LANE RIGHT | Move to right lane |
| 3 | FASTER | Accelerate |
| 4 | SLOWER | Brake |


## Markov Decision Process

$$\text{MDP} = (S, A, P, R, \gamma)$$

| Symbol | Meaning | In Our Problem |
|--------|---------|----------------|
| $S$ | State space | 90-dim kinematic vector |
| $A$ | Action space | {LEFT, IDLE, RIGHT, FASTER, SLOWER} |
| $P$ | Transition | Highway-env physics simulator |
| $R$ | Reward | Shaped reward (below) |
| $\gamma$ | Discount | 0.99 (care about future) |

### Reward Function (per timestep)

$$R = \underbrace{0.4 \cdot R_{\text{speed}}}_{\text{go fast}} + \underbrace{0.1 \cdot R_{\text{right lane}}}_{\text{stay right}} + \underbrace{(-1) \cdot \mathbb{1}_{\text{collision}}}_{\text{don't crash}} + \underbrace{0.1 \cdot R_{\text{left lane}}}_{\text{custom bonus}}$$

**Left Lane Wrapper:** Our custom modification — rewards lane 0 most.
```
Lane 0 (leftmost)  → +0.10 bonus  (overtaking lane)
Lane 1             → +0.067 bonus
Lane 2             → +0.033 bonus
Lane 3 (rightmost) → +0.00 bonus
```
## Algorithm Family Tree

```
 Deep Q-Network — DQN 
            ├── Double DQN [Fix overestimation]
            ├── Dueling DQN [Better V/A separation]
            └── Rainbow DQN  [All combined]

Policy Gradient 
    └── PPO — Proximal Policy Optimization [Different paradigm]
```

### Hyperparameters
| Param | Value |
|-------|-------|
| LR | 5e-4 |
| Buffer | 100,000 |
| Batch | 64 |
| γ | 0.99 |
| Target update | every 1000 steps |
| Train freq | every 4 steps |

Weakness: Overestimation Bias
Q-values → systematically overestimates → unstable learning



| | DQN | Double DQN |
|--|-----|------------|
| Selects action | online net | **online net** |
| Evaluates Q | online net | target net |



- DQN: same network picks AND scores → always picks the noisily-highest → overestimates
- Double: different networks decouple selection from evaluation → bias cancels out

## Dueling DQN 

Not every action matters in every state.  
At high speed in an empty lane → brake/accelerate difference is SMALL.  
But the state itself has HIGH VALUE.

### Split Q into Value + Advantage

```
         ↙                                        ↘
Value Stream                              Advantage Stream
         ↘                                        ↙
          Q(s,a) = V(s) + A(s,a) − mean(A)
```

### Why subtract mean A?
Identifiability: Without it, you can add any constant to V and subtract from A — infinite solutions. Subtracting the mean forces a unique decomposition.

### Why it helps
- The network explicitly learns WHICH STATES are valuable  
- Better generalization: learns V(s) that applies to ALL actions  
- More stable gradient flow


## Rainbow DQN 


| Component | What it does | Status |
|-----------|-------------|--------|
| Double DQN | Fix overestimation bias | ✅ |
| Dueling DQN | V(s)/A(s,a) separation | ✅ |
| Noisy Networks | Replace ε-greedy with learned noise | ✅ |

### Noisy Networks
$$y = (\mu^w + \sigma^w \odot \varepsilon^w) x + (\mu^b + \sigma^b \odot \varepsilon^b)$$
Exploration is **learned** — no ε schedule needed. Noise adapts during training.

Weights ($w = \mu_w + \sigma_w \odot \epsilon_w$): Instead of fixed weights, each weight is treated as a random variable sampled from a learned distribution.

Biases ($b = \mu_b + \sigma_b \odot \epsilon_b$): Similarly, the bias term is randomized using a learned mean and scaled noise.

$\mu$ (Mean): These are the standard learnable parameters that represent the "core" value of the weight or bias.

$\sigma$ (Standard Deviation): These are learnable parameters that control the scale of the noise. If $\sigma$ is high, the agent explores more; as the agent learns, $\sigma$ typically decreases.

$\epsilon$ (Random Noise): These are stochastic variables usually sampled from a standard normal distribution $\mathcal{N}(0, 1)$ during every forward pass.

$\odot$ (Hadamard Product): This indicates element-wise multiplication between the noise scale ($\sigma$) and the random noise ($\epsilon$).

Purpose: It allows the agent to learn its own exploration strategy. The noise is "consistent" for a single step but "stochastic" across episodes, leading to more efficient state-space coverage than simple random action selection.

## PPO — Proximal Policy Optimization 

### Completely Different Paradigm

| | DQN Family | PPO |
|--|-----------|-----|
| Type | Off-policy, value-based | On-policy, policy-based |
| What it learns | Q(s,a) → derive policy | π(a\|s) directly |
| Replay buffer | ✅ Yes | ❌ No |
| ε-greedy | ✅ Yes | ❌ No (stochastic policy) |
| Data efficiency | High (reuses old data) | Low (discards after use) |
| Stability | Can diverge | Very stable |
---


**The clip prevents large policy updates** — if the ratio goes too far from 1, we clip it.  
This is the "proximal" part — stay close to the old policy.

### Hyperparameters
| Param | Value |
|-------|-------|
| LR | 3e-4 |
| n_steps (rollout) | 512 |
| n_epochs | 10 |
| GAE λ | 0.95 |
| Clip ε | 0.2 |
| Entropy coef | 0.01 |

---

---


### Environment → all 5 models see IDENTICAL config
```python
ENV_CONFIG = {
    "vehicles_count": 50,      # dense traffic
    "lanes_count": 4,
    "duration": 40,            # seconds per episode
    "observation": Kinematics(15 vehicles, 6 features),
    "action": DiscreteMetaAction(),   # 5 actions
    "collision_reward": -1,
    "high_speed_reward": 0.4,
    "right_lane_reward": 0.1,
    "normalize_reward": True
}
```
## Theoretical Prediction (Before Running)

### Expected Ranking
| Predicted Rank | Model | Reason |
|---------------|-------|--------|
| 🥇 1st | Rainbow DQN | All 5 components = maximum power |
| 🥈 2nd | Dueling DQN | Best architecture separation |
| 🥉 3rd | Double DQN | Fixes DQN's overestimation |
| 4th | PPO | Different paradigm — stable but slower convergence |
| 5th | DQN | Baseline — all the weaknesses |

### My Hypothesis
- Rainbow would dominate because it combines every known improvement
- Noisy nets would explore the state space better than ε-greedy
- PPO would be somewhere in the middle — decent but not top
- DQN would clearly be worst

### What seemed obvious:
1. More components = better performance
2. The Q-learning family would be most sample-efficient

---

---
## Reality vs Theory

### Actual Results (50 episodes each)

| Model | Mean Reward | Success Rate | Collision Rate | Mean Speed | Ep Length |
|-------|-------------|--------------|----------------|------------|-----------|
| **PPO** | **29.38 ± 4.54** | **96%** | **4%** | 20.16 m/s | 38.82 |
| **Rainbow** | **29.21 ± 6.95** | **88%** | **12%** | 20.67 m/s | 37.44 |
| Dueling DQN | 24.31 ± 10.94 | 46% | 54% | 25.80 m/s | 27.8 |
| Double DQN | 23.79 ± 11.83 | 20% | 80% | 28.75 m/s | 24.38 |
| DQN | 20.78 ± 13.04 | **20%** | **80%** | 28.85 m/s | 21.24 |


---
### 1. On-policy is better for dense traffic
- PPO always trains on FRESH data from the current policy
- DQN family reuses old transitions — but those were from an inferior, crash-prone policy
- In 50-car traffic: the situation changes rapidly — stale data misleads

### 2. No exploration dilemma
- ε-greedy forces random actions (crashes) during exploration
- PPO's stochastic policy explores naturally — smaller random steps, not wild actions
- Result: PPO discovers "go slow and stay left" early on; DQN randomly crashes

### 3. PPO's stability = consistent convergence
- Clip objective prevents catastrophic policy updates
- DQN can "forget" good behaviors when Q-values suddenly shift

---

### DQN & Double DQN
```
[90 inputs]
     ↓
[Linear 256 + ReLU]
     ↓
[Linear 256 + ReLU]
     ↓
[Linear → 5 Q-values]
  Action with max Q → execute
```

### Dueling DQN
```
[90 inputs]
     ↓
[Linear 256 + ReLU] → [Linear 256 + ReLU]
          ↙                      ↘
[Value head: 256→128→1]   [Advantage head: 256→128→5]
       V(s)                       A(s,a)
          ↘                      ↙
    Q(s,a) = V(s) + A(s,a) − mean(A)
```

### Rainbow DQN (NoisyLinear layers)
```
[90 inputs]
     ↓
[NoisyLinear 256 + ReLU] → [NoisyLinear 256 + ReLU]
          ↙                                    ↘
[NoisyLinear 256→128→1]          [NoisyLinear 256→128→5]
       V(s)                               A(s,a)
          ↘                                    ↙
    Q(s,a) = V(s) + A(s,a) − mean(A)
    + Sampled from PER with n=3 returns
```

### PPO (Actor-Critic)
```
[90 inputs]
     ↓
[Shared: Linear 256 + tanh → Linear 256 + tanh]
          ↙                    ↘
[Actor head: →5]         [Critic head: →1]
π(a|s) prob dist          V(s) value
     ↓                         ↓
Sample action              Compute advantage A
    ↓
Clip ratio & update
```

---

---
