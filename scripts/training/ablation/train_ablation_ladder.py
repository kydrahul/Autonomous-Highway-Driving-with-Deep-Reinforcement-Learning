"""
Ablation Study: Incremental Rainbow Components
================================================
Trains a sequence of agents, each adding one Rainbow component
on top of the previous. This isolates the contribution of each
component in the dense highway traffic setting.

Ablation Ladder:
  Level 0 — DQN               (baseline, epsilon-greedy)
  Level 1 — DQN + Double      (+Double DQN: remove overestimation bias)
  Level 2 — DQN + Double + Dueling   (+Dueling: V(s) + A(s,a))
  Level 3 — + PER             (+Prioritized Experience Replay)
  Level 4 — + NoisyNets       (+NoisyLinear instead of epsilon-greedy)
  Level 5 — + n-step (n=3)    (+Multi-step returns) = Full Rainbow (5/6)

Usage:
    python scripts/training/ablation/train_ablation_ladder.py [--steps 300000]

Saves to: models/ablation/level_{N}_{name}/
"""

import os
import sys
import math
import numpy as np
import gymnasium as gym
import highway_env  # noqa: F401
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
from typing import Dict, List, Optional, Tuple, Type, Union

from stable_baselines3 import DQN
from stable_baselines3.dqn.policies import DQNPolicy, QNetwork
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.type_aliases import ReplayBufferSamples

# Import Rainbow components from the main training script
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, "scripts/training")

# ── Shared Environment Config ────────────────────────────────────────────────────
ENV_CONFIG = {
    "observation": {
        "type": "Kinematics",
        "vehicles_count": 15,
        "features": ["x", "y", "vx", "vy", "cos_h", "sin_h"],
        "normalize": True,
        "absolute": False,
    },
    "action": {"type": "DiscreteMetaAction"},
    "lanes_count": 4,
    "vehicles_count": 50,
    "duration": 40,
    "initial_spacing": 2,
    "collision_reward": -1,
    "right_lane_reward": 0.1,
    "high_speed_reward": 0.4,
    "reward_speed_range": [20, 30],
    "normalize_reward": True,
    "simulation_frequency": 5,
    "policy_frequency": 1,
    "other_vehicles_type": "highway_env.vehicle.behavior.IDMVehicle",
}


class LeftLaneRewardWrapper(gym.Wrapper):
    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)
        lane = self.unwrapped.vehicle.lane_index[2]
        n    = self.unwrapped.config["lanes_count"]
        reward += 0.1 * (n - 1 - lane) / (n - 1)
        return obs, reward, done, truncated, info


def make_env():
    env = gym.make("highway-v0", config=ENV_CONFIG)
    env = LeftLaneRewardWrapper(env)
    env = Monitor(env)
    return env


# ════════════════════════════════════════════════════════════════════════════════
# Re-implement NoisyLinear & components inline (self-contained ablation file)
# ════════════════════════════════════════════════════════════════════════════════

class NoisyLinear(nn.Module):
    def __init__(self, in_f: int, out_f: int, sigma_init: float = 0.5):
        super().__init__()
        self.in_f, self.out_f = in_f, out_f
        self.weight_mu    = nn.Parameter(th.empty(out_f, in_f))
        self.weight_sigma = nn.Parameter(th.empty(out_f, in_f))
        self.bias_mu      = nn.Parameter(th.empty(out_f))
        self.bias_sigma   = nn.Parameter(th.empty(out_f))
        self.register_buffer("weight_epsilon", th.empty(out_f, in_f))
        self.register_buffer("bias_epsilon",   th.empty(out_f))
        mu_range = 1.0 / math.sqrt(in_f)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(sigma_init / math.sqrt(in_f))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(sigma_init / math.sqrt(out_f))
        self.reset_noise()

    @staticmethod
    def _fn(x):
        return x.sign().mul_(x.abs().sqrt_())

    def reset_noise(self):
        ei = self._fn(th.randn(self.in_f))
        eo = self._fn(th.randn(self.out_f))
        self.weight_epsilon.copy_(eo.outer(ei))
        self.bias_epsilon.copy_(eo)

    def forward(self, x):
        if self.training:
            w = self.weight_mu + self.weight_sigma * self.weight_epsilon
            b = self.bias_mu   + self.bias_sigma   * self.bias_epsilon
        else:
            w, b = self.weight_mu, self.bias_mu
        return F.linear(x, w, b)


class DuelingQNet(QNetwork):
    """Dueling architecture (no noisy)."""
    def __init__(self, obs_space, act_space, features_extractor, features_dim,
                 net_arch=None, activation_fn=nn.ReLU, normalize_images=True):
        super().__init__(obs_space, act_space, features_extractor, features_dim,
                         net_arch, activation_fn, normalize_images)
        n_act   = int(act_space.n)
        n_arch  = net_arch if net_arch else [256, 256]
        in_dim  = features_dim
        shared  = []
        for h in n_arch:
            shared += [nn.Linear(in_dim, h), activation_fn()]
            in_dim  = h
        self.q_net          = nn.Sequential(*shared)
        self.value_stream   = nn.Sequential(nn.Linear(in_dim, 128), activation_fn(), nn.Linear(128, 1))
        self.advantage_stream = nn.Sequential(nn.Linear(in_dim, 128), activation_fn(), nn.Linear(128, n_act))

    def forward(self, obs):
        f = self.extract_features(obs, self.features_extractor)
        s = self.q_net(f)
        v = self.value_stream(s)
        a = self.advantage_stream(s)
        return v + a - a.mean(dim=1, keepdim=True)


class DuelingPolicy(DQNPolicy):
    def make_q_net(self): return DuelingQNet(**self._update_features_extractor(self.net_args, None)).to(self.device)


class NoisyDuelingQNet(QNetwork):
    """Dueling + NoisyNets."""
    def __init__(self, obs_space, act_space, features_extractor, features_dim,
                 net_arch=None, activation_fn=nn.ReLU, normalize_images=True):
        super().__init__(obs_space, act_space, features_extractor, features_dim,
                         net_arch, activation_fn, normalize_images)
        n_act  = int(act_space.n)
        n_arch = net_arch if net_arch else [256, 256]
        in_dim = features_dim
        shared = []
        for h in n_arch:
            shared += [NoisyLinear(in_dim, h), activation_fn()]
            in_dim  = h
        self.q_net            = nn.Sequential(*shared)
        self.value_stream     = nn.Sequential(NoisyLinear(in_dim, 128), activation_fn(), NoisyLinear(128, 1))
        self.advantage_stream = nn.Sequential(NoisyLinear(in_dim, 128), activation_fn(), NoisyLinear(128, n_act))

    def reset_noise(self):
        for m in self.modules():
            if isinstance(m, NoisyLinear): m.reset_noise()

    def forward(self, obs):
        f = self.extract_features(obs, self.features_extractor)
        s = self.q_net(f)
        v = self.value_stream(s); a = self.advantage_stream(s)
        return v + a - a.mean(dim=1, keepdim=True)


class NoisyDuelingPolicy(DQNPolicy):
    def make_q_net(self): return NoisyDuelingQNet(**self._update_features_extractor(self.net_args, None)).to(self.device)


# ── Minimal SumTree + PER for ablation (re-used from main script) ─────────────
class SumTree:
    def __init__(self, cap):
        self.cap = cap; self.tree = np.zeros(2*cap-1, dtype=np.float64)
        self.ptr = 0; self.n = 0
    def _prop(self, idx, d):
        p = (idx-1)//2; self.tree[p] += d
        if p: self._prop(p, d)
    def _get(self, idx, s):
        l = 2*idx+1; r = l+1
        if l >= len(self.tree): return idx
        return self._get(l, s) if s <= self.tree[l] else self._get(r, s-self.tree[l])
    @property
    def total(self): return float(self.tree[0])
    @property
    def min_p(self): return float(np.min(self.tree[self.cap-1:self.cap-1+self.n])) if self.n else 1.0
    def add(self, p):
        li = self.ptr+self.cap-1; di = self.ptr
        self.update(li, p); self.ptr=(self.ptr+1)%self.cap; self.n=min(self.n+1,self.cap); return di
    def update(self, li, p):
        d = p-self.tree[li]; self.tree[li]=p; self._prop(li, d)
    def get(self, s):
        li = self._get(0, s); di = li-self.cap+1; return li, self.tree[li], di


class AblationPER(ReplayBuffer):
    """PER buffer for ablation (same as main but without n-step)."""
    def __init__(self, buf_size, obs_space, act_space, device="auto", n_envs=1,
                 optimize_memory_usage=False, handle_timeout_termination=True,
                 alpha=0.6, beta_start=0.4, beta_frames=300_000):
        super().__init__(buf_size, obs_space, act_space, device, n_envs,
                         optimize_memory_usage, handle_timeout_termination)
        self.alpha=alpha; self.beta=beta_start; self.beta_start=beta_start
        self.beta_frames=beta_frames; self.max_p=1.0; self.eps=1e-6
        self.tree=SumTree(buf_size); self._frame=0; self._last_li=None

    def add(self, obs, next_obs, action, reward, done, infos):
        pos = self.pos; super().add(obs, next_obs, action, reward, done, infos)
        p = self.max_p**self.alpha; li = pos+self.tree.cap-1
        self.tree.update(li, p); self.tree.n = self.buffer_size if self.full else self.pos

    def sample(self, batch_size, env=None):
        self._frame += 1
        self.beta = min(1.0, self.beta_start + self._frame*(1-self.beta_start)/self.beta_frames)
        idxs=np.empty(batch_size,dtype=np.int64); lis=np.empty(batch_size,dtype=np.int64)
        pris=np.empty(batch_size,dtype=np.float64)
        tot=self.tree.total; seg=tot/batch_size
        for i in range(batch_size):
            li,p,di = self.tree.get(np.random.uniform(seg*i, seg*(i+1)))
            di=max(0,min(di,self.buffer_size-1)); idxs[i]=di; lis[i]=li; pris[i]=p
        n=self.tree.n; pm=self.tree.min_p/tot; mw=(n*pm)**(-self.beta)
        ws=np.array([(n*p/tot)**(-self.beta)/mw for p in pris],dtype=np.float32)
        self._last_li=lis; samp=self._get_samples(idxs, env=env)
        return samp, th.tensor(ws,dtype=th.float32,device=self.device).unsqueeze(1), idxs

    def update_priorities(self, idxs, td_errors):
        ps=(np.abs(td_errors)+self.eps)**self.alpha
        for di,li,p in zip(idxs,self._last_li,ps):
            self.tree.update(int(li),float(p)); self.max_p=max(self.max_p,float(p))


class DoubleDQNWithPER(DQN):
    """Double DQN + PER (no noisy, no dueling, no n-step)."""
    def train(self, gradient_steps, batch_size=100):
        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)
        losses = []
        for _ in range(gradient_steps):
            rd, ws, idxs = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)
            with th.no_grad():
                na = self.q_net(rd.next_observations).argmax(dim=1, keepdim=True)
                nq = th.gather(self.q_net_target(rd.next_observations), 1, na)
                tq = rd.rewards + (1-rd.dones)*self.gamma*nq
            cq = th.gather(self.q_net(rd.observations), 1, rd.actions.long())
            tde = (cq-tq).detach().abs().squeeze(1).cpu().numpy()
            loss = (ws*F.smooth_l1_loss(cq, tq, reduction="none")).mean()
            self.policy.optimizer.zero_grad(); loss.backward()
            th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.policy.optimizer.step()
            self.replay_buffer.update_priorities(idxs, tde)
            losses.append(loss.item())
        self._n_updates += gradient_steps
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/loss", float(np.mean(losses)))


class NoisyDoubleDQNWithPER(DoubleDQNWithPER):
    """Double DQN + PER + NoisyNets + Dueling."""
    def train(self, gradient_steps, batch_size=100):
        super().train(gradient_steps, batch_size)
        self.q_net.reset_noise(); self.q_net_target.reset_noise()


# ─────────────────────────────────────────────────────────────────────────────────
def train_level(name: str, model, total_steps: int, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    ckpt_cb = CheckpointCallback(save_freq=100_000, save_path=out_dir, name_prefix=name, verbose=0)
    print(f"\n  ▶  Training ablation level: {name}")
    model.learn(total_timesteps=total_steps, callback=ckpt_cb,
                tb_log_name=name, reset_num_timesteps=True, progress_bar=True)
    model.save(os.path.join(out_dir, f"{name}_final"))
    print(f"  ✅  Saved: {out_dir}/{name}_final.zip")


# ─────────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps",   type=int, default=300_000)
    parser.add_argument("--seed",    type=int, default=42)
    parser.add_argument("--levels",  type=int, nargs="+", default=[0,1,2,3,4,5],
                        help="Which ablation levels to train (0–5)")
    args = parser.parse_args()

    S = args.steps; seed = args.seed

    LEVELS = [
        # (level, name, model_class, policy, extra_kwargs)
        (0, "L0_DQN",           DQN, "MlpPolicy",     {}),
        (1, "L1_Double_DQN",    DQN, "MlpPolicy",     {}),
        (2, "L2_Dueling",       DQN, DuelingPolicy,   {}),
        (3, "L3_PER",           DoubleDQNWithPER, DuelingPolicy,
            {"replay_buffer_class": AblationPER, "replay_buffer_kwargs": dict(alpha=0.6, beta_start=0.4, beta_frames=S)}),
        (4, "L4_Noisy",         NoisyDoubleDQNWithPER, NoisyDuelingPolicy,
            {"replay_buffer_class": AblationPER, "replay_buffer_kwargs": dict(alpha=0.6, beta_start=0.4, beta_frames=S),
             "exploration_fraction": 0.0, "exploration_initial_eps": 0.0, "exploration_final_eps": 0.0}),
        # Level 5 is the full Rainbow from train_rainbow_dqn.py
    ]

    for level, name, Cls, policy, kwargs in LEVELS:
        if level not in args.levels:
            continue
        env = make_env()
        out = f"models/ablation/{name}"
        base_kwargs = dict(
            policy=policy, env=env,
            learning_rate=5e-4, buffer_size=100_000, batch_size=64,
            gamma=0.99, target_update_interval=1000, train_freq=4,
            gradient_steps=1, learning_starts=1000,
            exploration_fraction=0.1, exploration_initial_eps=1.0,
            exploration_final_eps=0.05,
            policy_kwargs=dict(net_arch=[256, 256]),
            tensorboard_log=f"logs/ablation/{name}",
            device="auto", verbose=0, seed=seed,
        )
        base_kwargs.update(kwargs)
        model = Cls(**base_kwargs)
        train_level(name, model, S, out)
        env.close()

    print("\n" + "=" * 60)
    print("  Ablation Training Complete!")
    print("  Next: python scripts/evaluation/evaluate_ablation.py")
    print("=" * 60)
