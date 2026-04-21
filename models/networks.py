"""
Standard actor-critic networks used by sub-policies and the meta-policy.
"""

import torch
import torch.nn as nn
from torch.distributions import Categorical, Normal


class PolicyNetwork(nn.Module):
    """
    Actor π(a | s).

    Discrete action spaces  → Categorical distribution over logits.
    Continuous action spaces → Diagonal Normal distribution (mean + log_std).
    """

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256,
                 discrete: bool = True):
        super().__init__()
        self.discrete = discrete
        self.trunk = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.action_head = nn.Linear(hidden_dim, action_dim)
        if not discrete:
            self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, s: torch.Tensor):
        """Returns a torch.distributions object (Categorical or Normal)."""
        feat   = self.trunk(s)
        logits = self.action_head(feat)
        if self.discrete:
            return Categorical(logits=logits)
        else:
            std = self.log_std.exp().expand_as(logits)
            return Normal(logits, std)

    def get_action(self, s: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        """Sample (or argmax) an action. s may be [state_dim] or [B, state_dim]."""
        single = s.dim() == 1
        if single:
            s = s.unsqueeze(0)
        dist = self.forward(s)
        if deterministic:
            a = dist.probs.argmax(dim=-1) if self.discrete else dist.mean
        else:
            a = dist.sample()
        return a.squeeze(0) if single else a

    def log_prob(self, s: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        """Log probability of action a under π(·|s). [B] for discrete, [B] for continuous."""
        dist = self.forward(s)
        if self.discrete:
            return dist.log_prob(a.long())
        else:
            return dist.log_prob(a).sum(dim=-1)


class ValueNetwork(nn.Module):
    """Critic V(s) → scalar."""

    def __init__(self, state_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        """s: [B, state_dim] → [B, 1]."""
        return self.net(s)
