"""
BackwardValueNet — backward-discounted value function V(s; S_proto).
RewardNormalizer — online Welford mean/variance for reward normalisation.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class RewardNormalizer:
    """Online Welford mean/variance tracker for reward normalisation."""

    def __init__(self):
        self.mean  = 0.0
        self.M2    = 0.0
        self.count = 0

    def update(self, rewards) -> None:
        for r in rewards:
            self.count += 1
            delta      = float(r) - self.mean
            self.mean += delta / self.count
            delta2     = float(r) - self.mean
            self.M2   += delta * delta2

    @property
    def std(self) -> float:
        return float(np.sqrt(self.M2 / max(self.count - 1, 1)))

    def normalize(self, reward: float) -> float:
        return (float(reward) - self.mean) / (self.std + 1e-8)

    def denormalize(self, value: float) -> float:
        return float(value) * (self.std + 1e-8) + self.mean


class BackwardValueNet(nn.Module):
    """
    Backward-discounted value function V(s; S_proto).
    BatchNorm1d after each linear layer for gradient stability.
    Batch size must be > 1 during training.
    """

    def __init__(self, state_dim: int, set_embedding_dim: int = 64, hidden_dim: int = 256):
        super().__init__()
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
        )
        self.set_encoder = nn.Sequential(
            nn.Linear(state_dim, set_embedding_dim),
            nn.BatchNorm1d(set_embedding_dim),
            nn.ReLU(),
            nn.Linear(set_embedding_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
        )
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, s: torch.Tensor, S_proto: torch.Tensor) -> torch.Tensor:
        s_feat   = self.state_encoder(s)
        set_feat = self.set_encoder(S_proto)
        return self.value_head(torch.cat([s_feat, set_feat], dim=-1))


def train_backward_value_net(
    adapter,
    value_net: BackwardValueNet,
    optimizer,
    gamma: float         = 0.99,
    n_epochs: int        = 10,
    batch_size: int      = 256,
    device: str          = "cpu",
    clip_grad: float     = 1.0,
    reward_normalizer    = None,
    target_clip: float   = None,
    hit_threshold: float = 1.0,
) -> list:
    """
    Train BackwardValueNet with reverse-trajectory Monte Carlo targets.

    Target for s_t: Φ_t = Σ_{k=t}^{hit-1} γ^{hit-1-k} · r̃_k
    Only terminal or hit episodes are used.  Returns per-epoch mean losses.
    """
    all_s, all_s_hit, all_targets = [], [], []

    for ep in adapter.iter_episodes():
        states  = np.asarray(ep["states"],  dtype=np.float32)
        rewards = np.asarray(ep["rewards"], dtype=np.float32)
        dones   = np.asarray(ep["dones"],   dtype=bool)
        T = len(rewards)
        if T == 0:
            continue

        s_terminal = states[-1]
        hit_idx    = None
        for i in range(T):
            if float(np.linalg.norm(states[i + 1] - s_terminal)) < hit_threshold:
                hit_idx = i
                break
        if hit_idx is None:
            if not dones[-1]:
                continue
            hit_idx = T - 1

        T_use = hit_idx + 1
        raw_r = rewards[:T_use]
        norm_r = (np.array([reward_normalizer.normalize(r) for r in raw_r], dtype=np.float32)
                  if reward_normalizer is not None else raw_r.copy())

        gam_pows   = gamma ** np.arange(T_use - 1, -1, -1, dtype=np.float32)
        weighted   = gam_pows * norm_r
        mc_targets = np.flip(np.cumsum(np.flip(weighted))).astype(np.float32)

        if target_clip is not None:
            mc_targets = np.clip(mc_targets, -target_clip, target_clip)

        s_hit = states[hit_idx + 1]
        for t in range(T_use):
            all_s.append(states[t])
            all_s_hit.append(s_hit)
            all_targets.append(float(mc_targets[t]))

    if not all_s:
        return []

    s_tensor   = torch.tensor(np.array(all_s,       dtype=np.float32), device=device)
    st_tensor  = torch.tensor(np.array(all_s_hit,   dtype=np.float32), device=device)
    tgt_tensor = torch.tensor(np.array(all_targets, dtype=np.float32),
                               device=device).unsqueeze(1)

    N = len(s_tensor)
    epoch_losses = []
    value_net.train()

    for _ in range(n_epochs):
        perm = torch.randperm(N, device=device)
        batch_losses = []
        for start in range(0, N, batch_size):
            idx = perm[start : start + batch_size]
            if len(idx) <= 1:
                continue
            pred = value_net(s_tensor[idx], st_tensor[idx])
            loss = F.huber_loss(pred, tgt_tensor[idx], delta=1.0)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(value_net.parameters(), clip_grad)
            optimizer.step()
            batch_losses.append(loss.item())
        if batch_losses:
            epoch_losses.append(float(np.mean(batch_losses)))

    value_net.eval()
    return epoch_losses
