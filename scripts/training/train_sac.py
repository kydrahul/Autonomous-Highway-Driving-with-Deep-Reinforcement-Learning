"""
train_sac.py
============
Discrete Soft Actor-Critic (SAC) for highway-env.

Based on: Christodoulou (2019) "Soft Actor-Critic for Discrete Action Settings"
arXiv:1910.07207

Why SAC matters for this paper:
  SAC is OFF-POLICY (uses replay buffer) + ENTROPY-MAXIMIZING (stochastic policy).
  PPO is ON-POLICY (no replay buffer) + ENTROPY-REGULARIZED.

  If SAC ≈ PPO: entropy is the key, not on-policy structure → reframe paper.
  If SAC < PPO: on-policy structure is the key → H4 confirmed.
  If SAC > Rainbow: entropy helps even off-policy → partial H4 support.

Architecture:
  Actor: MLP → softmax → categorical distribution π(a|s)
  Critic: Two separate Q networks (Q1, Q2) for double-Q trick
  Entropy target: H_target = -log(1/|A|) = log(|A|)
  Temperature α: learned automatically via dual gradient descent

Usage:
    python scripts/training/train_sac.py
    python scripts/training/train_sac.py --seeds 42 123 456 --steps 300000
    python scripts/training/train_sac.py --seed 42 --eval-only models/sac/seed42/
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import deque
import random
import gymnasium as gym
import highway_env  # noqa: F401

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

os.makedirs("models/sac",        exist_ok=True)
os.makedirs("results/sac",       exist_ok=True)
os.makedirs("results/plots",     exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[SAC] Using device: {DEVICE}")

# ── Hyperparameters ────────────────────────────────────────────────────────────
STEPS         = 300_000
SEEDS         = [42, 123, 456]
LR            = 3e-4
GAMMA         = 0.99
BATCH_SIZE    = 64
BUFFER_SIZE   = 100_000
UPDATE_EVERY  = 1       # update every step after warm-up
WARMUP_STEPS  = 5_000
TAU           = 0.005   # soft target update
EVAL_EPS      = 50
DENSITY       = 50


# ── Networks ──────────────────────────────────────────────────────────────────
class DiscreteActor(nn.Module):
    """
    Outputs a categorical distribution π(a|s) via softmax.
    Entropy computed in closed form: H = -Σ π(a|s) log π(a|s)
    """
    def __init__(self, obs_dim: int, n_actions: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden),  nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, x):
        logits = self.net(x)
        probs  = F.softmax(logits, dim=-1)
        log_p  = F.log_softmax(logits, dim=-1)
        return probs, log_p

    def get_action(self, x):
        probs, log_p = self(x)
        dist   = torch.distributions.Categorical(probs)
        action = dist.sample()
        return action, probs, log_p


class DiscreteCritic(nn.Module):
    """Q(s, a) for all a simultaneously — standard for discrete SAC."""
    def __init__(self, obs_dim: int, n_actions: int, hidden: int = 256):
        super().__init__()
        self.q1 = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden),  nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )
        self.q2 = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden),  nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, x):
        return self.q1(x), self.q2(x)


# ── Replay Buffer ─────────────────────────────────────────────────────────────
class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buf = deque(maxlen=capacity)

    def push(self, s, a, r, s2, done):
        self.buf.append((s, a, r, s2, done))

    def sample(self, n: int):
        batch = random.sample(self.buf, n)
        s, a, r, s2, d = zip(*batch)
        return (
            torch.FloatTensor(np.array(s)).to(DEVICE),
            torch.LongTensor(a).to(DEVICE),
            torch.FloatTensor(r).to(DEVICE).unsqueeze(1),
            torch.FloatTensor(np.array(s2)).to(DEVICE),
            torch.FloatTensor(d).to(DEVICE).unsqueeze(1),
        )

    def __len__(self):
        return len(self.buf)


# ── Discrete SAC Agent ────────────────────────────────────────────────────────
class DiscreteSAC:
    def __init__(self, obs_dim: int, n_actions: int):
        self.n_actions = n_actions

        self.actor   = DiscreteActor(obs_dim, n_actions).to(DEVICE)
        self.critic  = DiscreteCritic(obs_dim, n_actions).to(DEVICE)
        self.critic_target = DiscreteCritic(obs_dim, n_actions).to(DEVICE)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_opt  = optim.Adam(self.actor.parameters(),  lr=LR)
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=LR)

        # Learnable temperature α (log scale for stability)
        self.log_alpha     = torch.zeros(1, requires_grad=True, device=DEVICE)
        self.alpha_opt     = optim.Adam([self.log_alpha], lr=LR)
        self.target_entropy = -np.log(1.0 / n_actions) * 0.98  # H_target

        self.buffer = ReplayBuffer(BUFFER_SIZE)

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def select_action(self, obs: np.ndarray, deterministic: bool = False):
        with torch.no_grad():
            x = torch.FloatTensor(obs).unsqueeze(0).to(DEVICE)
            probs, _ = self.actor(x)
            if deterministic:
                return int(probs.argmax(dim=-1).item())
            dist = torch.distributions.Categorical(probs)
            return int(dist.sample().item())

    def update(self):
        if len(self.buffer) < BATCH_SIZE:
            return {}

        s, a, r, s2, done = self.buffer.sample(BATCH_SIZE)

        with torch.no_grad():
            next_probs, next_log_p = self.actor(s2)
            q1_next, q2_next = self.critic_target(s2)
            q_next = torch.min(q1_next, q2_next)
            # V(s') = Σ π(a'|s') [Q(s',a') - α log π(a'|s')]
            v_next = (next_probs * (q_next - self.alpha * next_log_p)).sum(dim=1, keepdim=True)
            target_q = r + GAMMA * (1.0 - done) * v_next

        q1, q2 = self.critic(s)
        q1_a = q1.gather(1, a.unsqueeze(1))
        q2_a = q2.gather(1, a.unsqueeze(1))
        critic_loss = F.mse_loss(q1_a, target_q) + F.mse_loss(q2_a, target_q)

        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        # Actor update
        probs, log_p = self.actor(s)
        q1_cur, q2_cur = self.critic(s)
        q_cur = torch.min(q1_cur, q2_cur)
        # J(π) = Σ π(a|s) [α log π(a|s) - Q(s,a)]
        actor_loss = (probs * (self.alpha.detach() * log_p - q_cur)).sum(dim=1).mean()

        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

        # Temperature update
        # J(α) = E[-α log π(a|s) - α H_target]
        entropy = -(probs * log_p).sum(dim=1).mean().detach()
        alpha_loss = self.log_alpha * (entropy - self.target_entropy)

        self.alpha_opt.zero_grad()
        alpha_loss.backward()
        self.alpha_opt.step()

        # Soft target update
        for p, pt in zip(self.critic.parameters(), self.critic_target.parameters()):
            pt.data.copy_(TAU * p.data + (1.0 - TAU) * pt.data)

        return {
            "critic_loss": critic_loss.item(),
            "actor_loss":  actor_loss.item(),
            "alpha":       self.alpha.item(),
            "entropy":     entropy.item(),
        }

    def save(self, path: str):
        os.makedirs(path, exist_ok=True)
        torch.save({
            "actor":  self.actor.state_dict(),
            "critic": self.critic.state_dict(),
        }, os.path.join(path, "sac.pt"))

    def load(self, path: str):
        ckpt = torch.load(os.path.join(path, "sac.pt"), map_location=DEVICE)
        self.actor.load_state_dict(ckpt["actor"])
        self.critic.load_state_dict(ckpt["critic"])


# ── Environment helpers ───────────────────────────────────────────────────────
# ── Left Lane Reward Wrapper (must match other training scripts) ──────────────
class LeftLaneRewardWrapper(gym.Wrapper):
    """Adds a bonus reward for staying in the left-most lane."""
    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)
        current_lane = self.unwrapped.vehicle.lane_index[2]
        total_lanes  = self.unwrapped.config["lanes_count"]
        left_reward  = (total_lanes - 1 - current_lane) / (total_lanes - 1)
        reward      += 0.1 * left_reward
        return obs, reward, done, truncated, info


def make_env(seed: int, density: int = DENSITY):
    env = gym.make("highway-v0", render_mode=None)
    env.unwrapped.config.update({
        "vehicles_count": density,
        "lanes_count": 4,
        "observation": {
            "type": "Kinematics",
            "vehicles_count": 15,
            "features": ["x", "y", "vx", "vy", "cos_h", "sin_h"],
            "normalize": True,
            "absolute": False,
        },
        "action": {"type": "DiscreteMetaAction"},
        "initial_spacing": 2,
        "collision_reward": -1,
        "right_lane_reward": 0.1,
        "high_speed_reward": 0.4,
        "reward_speed_range": [20, 30],
        "normalize_reward": True,
        "duration": 40,
        "simulation_frequency": 5,
        "policy_frequency": 1,
        "other_vehicles_type": "highway_env.vehicle.behavior.IDMVehicle",
    })
    env = LeftLaneRewardWrapper(env)
    env.reset(seed=seed)
    return env


def evaluate_sac(agent: DiscreteSAC, seed: int, n_eps: int = EVAL_EPS):
    env = make_env(seed + 9999)
    rewards, crashed = [], []
    obs, _ = env.reset()
    obs = obs.flatten()
    ep_r, ep_count = 0.0, 0
    while ep_count < n_eps:
        action = agent.select_action(obs, deterministic=False)
        obs, r, terminated, truncated, info = env.step(action)
        obs = obs.flatten()
        ep_r += r
        if terminated or truncated:
            rewards.append(ep_r)
            crashed.append(1 if info.get("crashed", False) else 0)
            obs, _ = env.reset()
            obs = obs.flatten()
            ep_r = 0.0
            ep_count += 1
    env.close()
    return {
        "mean_reward":    float(np.mean(rewards)),
        "std_reward":     float(np.std(rewards)),
        "success_rate":   float(1.0 - np.mean(crashed)),
        "collision_rate": float(np.mean(crashed)),
        "n_episodes":     n_eps,
    }


# ── Training loop ─────────────────────────────────────────────────────────────
def train_sac(seed: int, steps: int = STEPS, density: int = DENSITY):
    tag = f"seed{seed}"
    result_path = f"results/sac/sac_{tag}_results.json"
    model_dir   = f"models/sac/{tag}"

    if os.path.exists(result_path):
        print(f"  [skip] SAC {tag} — already trained")
        with open(result_path) as f:
            return json.load(f)

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    env = make_env(seed, density)
    obs_sample, _ = env.reset()
    obs_dim   = obs_sample.flatten().shape[0]
    n_actions = env.action_space.n
    print(f"\n[SAC] seed={seed}  obs_dim={obs_dim}  n_actions={n_actions}  steps={steps:,}")

    agent = DiscreteSAC(obs_dim, n_actions)
    obs, _ = env.reset()
    obs = obs.flatten()
    ep_r = 0.0
    ep_count = 0
    log_interval = 5_000

    for step in range(1, steps + 1):
        if step < WARMUP_STEPS:
            action = env.action_space.sample()
        else:
            action = agent.select_action(obs)

        next_obs, r, terminated, truncated, info = env.step(action)
        next_obs = next_obs.flatten()
        done = terminated or truncated
        agent.buffer.push(obs, action, r, next_obs, float(done))
        obs = next_obs
        ep_r += r

        if done:
            ep_count += 1
            obs, _ = env.reset()
            obs = obs.flatten()
            ep_r = 0.0

        if step >= WARMUP_STEPS and step % UPDATE_EVERY == 0:
            agent.update()

        if step % log_interval == 0:
            print(f"  step {step:>7,}/{steps:,}  episodes={ep_count}  "
                  f"alpha={agent.alpha.item():.3f}  buffer={len(agent.buffer):,}")

    agent.save(model_dir)
    metrics = evaluate_sac(agent, seed)
    metrics.update({"algo": "sac", "seed": seed, "steps": steps, "density": density})
    with open(result_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"  [SAC seed={seed}] success={metrics['success_rate']:.1%}  "
          f"reward={metrics['mean_reward']:.2f}  collision={metrics['collision_rate']:.1%}")
    return metrics


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Discrete SAC for highway-env")
    parser.add_argument("--seeds",         nargs="+", default=SEEDS, type=int)
    parser.add_argument("--steps",         default=STEPS,            type=int)
    parser.add_argument("--density",       default=DENSITY,          type=int)
    parser.add_argument("--eval-only",     default=None,             type=str,
                        help="Path to saved model dir for eval only")
    args = parser.parse_args()

    if args.eval_only:
        env = make_env(42)
        obs, _ = env.reset()
        obs_dim   = obs.flatten().shape[0]
        n_actions = env.action_space.n
        env.close()
        agent = DiscreteSAC(obs_dim, n_actions)
        agent.load(args.eval_only)
        metrics = evaluate_sac(agent, 42)
        print(f"\nEval results: {json.dumps(metrics, indent=2)}")
        return

    all_results = []
    for seed in args.seeds:
        r = train_sac(seed, args.steps, args.density)
        all_results.append(r)

    with open("results/sac/all_sac_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    # Summary
    successes = [r["success_rate"] for r in all_results]
    rewards   = [r["mean_reward"]  for r in all_results]
    print(f"\n{'='*50}")
    print(f"SAC Summary ({len(args.seeds)} seeds):")
    print(f"  Success rate: {np.mean(successes):.1%} +/- {np.std(successes):.1%}")
    print(f"  Mean reward:  {np.mean(rewards):.2f} +/- {np.std(rewards):.2f}")
    print(f"{'='*50}")
    print("[done] SAC training complete!")


if __name__ == "__main__":
    main()
