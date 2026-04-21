"""
History-conditioned meta-policy and value network.

MetaPolicy      — π_θ(a | s, τ)  GRU encodes task history τ, outputs action distribution.
MetaValueNetwork — V(s)           memoryless baseline used in the policy gradient update.
"""

import numpy as np
import torch
import torch.nn as nn


class MetaPolicy(nn.Module):
    """
    π_θ(a | s, τ) — meta-policy conditioned on current state and task history.

    τ is a list of (s, a, r) tuples accumulated during the current episode.
    A GRU encodes the history into a context vector that is concatenated with
    the current state before the action head.

    Args:
        state_dim:  observation dimensionality
        action_dim: number of discrete actions, or continuous action dimension
        hidden_dim: width of the MLP action head
        gru_hidden: GRU hidden-state size
        discrete:   True → Categorical output; False → Normal output
    """

    def __init__(
        self,
        state_dim:  int,
        action_dim: int,
        hidden_dim: int = 256,
        gru_hidden: int = 128,
        discrete:   bool = True,
    ):
        super().__init__()
        self.state_dim  = state_dim
        self.action_dim = action_dim
        self.gru_hidden = gru_hidden
        self.discrete   = discrete

        tau_step_dim = state_dim + action_dim + 1
        self.gru = (nn.GRU(tau_step_dim, gru_hidden, batch_first=True)
                    if gru_hidden > 0 else None)

        head_in = state_dim + gru_hidden
        self.policy_head = nn.Sequential(
            nn.Linear(head_in, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )
        if not discrete:
            self.log_std = nn.Parameter(torch.zeros(action_dim))

    # ------------------------------------------------------------------

    def _encode_action(self, a) -> torch.Tensor:
        """Return a float tensor of shape [action_dim] for any action type."""
        if self.discrete:
            if isinstance(a, torch.Tensor):
                idx = int(a.item())
            else:
                idx = int(np.asarray(a).flat[0])
            oh  = torch.zeros(self.action_dim)
            oh[idx % self.action_dim] = 1.0
            return oh
        arr = np.asarray(a, dtype=np.float32).flatten()
        if len(arr) < self.action_dim:
            arr = np.pad(arr, (0, self.action_dim - len(arr)))
        return torch.tensor(arr[: self.action_dim], dtype=torch.float32)

    def _encode_tau(self, tau: list, device) -> torch.Tensor:
        """Encode history list → context tensor [1, gru_hidden].  O(T) but
        rebuilds the full sequence from scratch; kept for backward compat."""
        if self.gru_hidden == 0 or not tau:
            return torch.zeros(1, self.gru_hidden, device=device)
        steps = []
        for s, a, r in tau:
            s_t = torch.tensor(
                np.asarray(s, dtype=np.float32).flatten()[: self.state_dim],
                dtype=torch.float32,
            )
            a_t = self._encode_action(a)
            r_t = torch.tensor([float(r)], dtype=torch.float32)
            steps.append(torch.cat([s_t, a_t, r_t]))
        seq = torch.stack(steps).unsqueeze(0).to(device)   # [1, T, step_dim]
        _, h_n = self.gru(seq)                              # h_n: [1, 1, gru_hidden]
        return h_n.squeeze(0)                               # [1, gru_hidden]

    # ── Incremental GRU interface ──────────────────────────────────────

    def init_hidden(self, device=None) -> torch.Tensor:
        """Zero hidden state [1, gru_hidden] for the start of an episode."""
        if device is None:
            device = next(self.parameters()).device
        return torch.zeros(1, self.gru_hidden, device=device)

    def forward_with_hidden(self, s, h: torch.Tensor):
        """
        Compute action distribution from current state s and GRU hidden state h.

        Args:
            s: current observation — np.ndarray or tensor [state_dim]
            h: GRU context          — Tensor [1, gru_hidden]

        Returns the same distribution type as forward().
        """
        device = next(self.parameters()).device
        if isinstance(s, torch.Tensor):
            s_arr = s.detach().cpu().numpy().flatten()
        else:
            s_arr = np.asarray(s, dtype=np.float32).flatten()

        s_t     = torch.tensor(s_arr[: self.state_dim], dtype=torch.float32, device=device)
        head_in = torch.cat([s_t.unsqueeze(0), h], dim=-1)   # [1, state_dim+gru_hidden]
        logits  = self.policy_head(head_in)                   # [1, action_dim]

        if self.discrete:
            return torch.distributions.Categorical(logits=logits)
        mean = logits
        std  = self.log_std.exp().clamp(1e-6, 1.0).expand_as(mean)
        return torch.distributions.Normal(mean, std)

    def update_hidden(self, s, a, r: float, h: torch.Tensor) -> torch.Tensor:
        """
        Advance the GRU hidden state by one (s, a, r) transition.

        Args:
            s: state before the action  — np.ndarray or tensor [state_dim]
            a: action taken             — int / np.ndarray / tensor
            r: reward received          — float
            h: current hidden state     — Tensor [1, gru_hidden]

        Returns h_new [1, gru_hidden].  Gradients flow through h_new → h.
        """
        device = next(self.parameters()).device
        if isinstance(s, torch.Tensor):
            s_arr = s.detach().cpu().numpy().flatten()
        else:
            s_arr = np.asarray(s, dtype=np.float32).flatten()

        if self.gru_hidden == 0:
            return h
        s_t   = torch.tensor(s_arr[: self.state_dim], dtype=torch.float32, device=device)
        a_t   = self._encode_action(a).to(device)
        r_t   = torch.tensor([float(r)], dtype=torch.float32, device=device)
        step  = torch.cat([s_t, a_t, r_t]).unsqueeze(0).unsqueeze(0)  # [1, 1, step_dim]
        _, h_new = self.gru(step, h.unsqueeze(0))   # h_new: [1, 1, gru_hidden]
        return h_new.squeeze(0)                      # [1, gru_hidden]

    def evaluate_actions(
        self,
        states_t:        torch.Tensor,
        actions_t:       torch.Tensor,
        hidden_states_t: torch.Tensor,
    ) -> tuple:
        """
        Re-evaluate log-probs and entropies for stored (s, a, h) triples under
        current policy parameters.  Used in importance-sampled gradient updates.

        Args:
            states_t:        [T, state_dim]
            actions_t:       [T] int (discrete) or [T, action_dim] float (continuous)
            hidden_states_t: [T, gru_hidden] — frozen behaviour context (detached)

        Returns:
            log_probs  Tensor [T]
            entropies  Tensor [T]
        """
        device  = next(self.parameters()).device
        s       = states_t.to(device)[:, : self.state_dim]       # [T, state_dim]
        h       = hidden_states_t.to(device)                     # [T, gru_hidden]
        head_in = torch.cat([s, h], dim=-1)                      # [T, state_dim+gru_hidden]
        logits  = self.policy_head(head_in)                      # [T, action_dim]

        if self.discrete:
            dist      = torch.distributions.Categorical(logits=logits)
            log_probs = dist.log_prob(actions_t.to(device).view(-1).long())
            entropies = dist.entropy()
        else:
            mean      = logits
            std       = self.log_std.exp().clamp(1e-6, 1.0).expand_as(mean)
            dist      = torch.distributions.Normal(mean, std)
            log_probs = dist.log_prob(actions_t.to(device)).sum(-1)
            entropies = dist.entropy().sum(-1)

        return log_probs, entropies

    # ------------------------------------------------------------------

    def forward(self, s, tau: list = None):
        """
        Args:
            s:   current state — np.ndarray [state_dim] or tensor
            tau: task history — list of (s, a, r) tuples (may be empty or None)

        Returns:
            torch.distributions.Categorical  (discrete)
            torch.distributions.Normal       (continuous)
        """
        if tau is None:
            tau = []
        device = next(self.parameters()).device

        if isinstance(s, torch.Tensor):
            s_arr = s.detach().cpu().numpy().flatten()
        else:
            s_arr = np.asarray(s, dtype=np.float32).flatten()

        s_t      = torch.tensor(s_arr[: self.state_dim], dtype=torch.float32, device=device)
        ctx      = self._encode_tau(tau, device)                      # [1, gru_hidden]
        head_in  = torch.cat([s_t.unsqueeze(0), ctx], dim=-1)        # [1, head_in]
        logits   = self.policy_head(head_in)                          # [1, action_dim]

        if self.discrete:
            return torch.distributions.Categorical(logits=logits)
        else:
            mean = logits
            std  = self.log_std.exp().clamp(1e-6, 1.0).expand_as(mean)
            return torch.distributions.Normal(mean, std)


class MetaValueNetwork(nn.Module):
    """V(s, ctx) — history-conditioned baseline for the meta-policy gradient update.

    ctx is the GRU hidden state from MetaPolicy, detached so value gradients
    do not flow back into the shared GRU weights.
    """

    def __init__(self, state_dim: int, gru_hidden: int = 0, hidden_dim: int = 256):
        super().__init__()
        in_dim = state_dim + gru_hidden
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, s: torch.Tensor, h: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            s: state tensor  [T, state_dim]
            h: GRU context   [T, gru_hidden]  — detached; pass None to use
               state only (backward compat when gru_hidden=0)
        """
        if not isinstance(s, torch.Tensor):
            s = torch.tensor(np.asarray(s, dtype=np.float32))
        if s.dim() == 1:
            s = s.unsqueeze(0)
        if h is not None:
            if h.dim() == 1:
                h = h.unsqueeze(0)
            x = torch.cat([s, h], dim=-1)
        else:
            x = s
        return self.net(x)
