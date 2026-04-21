"""
Training utilities for the meta-RL pipeline.

Provides:
  train_hitting_time_value()    – TD learning of V_H(s; {c}) for each critical state.
  train_sub_policy()            – A2C update for a single SubPolicy option.
  HistoryMetaPolicy             – GRU-conditioned meta-policy (sophisticated variant).
  HistoryMetaValueNetwork       – Matching value function.
  train_history_meta_policy()   – Full training loop for the history-conditioned policy.
"""

import sys
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Resolve project root so sibling packages are importable
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from models.HTVNet    import HittingTimeValueNet
from models.networks  import PolicyNetwork, ValueNetwork
from algos.HTTD       import train_hitting_time_value_net
from utils.utils      import compute_discounted_returns, eval_mode


# ── Hitting-time value training ────────────────────────────────────────────

def train_hitting_time_value(
    skeleton_data: dict,
    replay_buffer,
    gamma: float = 0.99,
    epochs: int  = 20,
    batch_size: int = 256,
    lr: float = 1e-3,
    tau: float = 0.005,
    device: str = "cpu",
) -> dict:
    """
    Train V_H(s; {c}) for each critical state c via TD learning.

    Args:
        skeleton_data: dict returned by build_skeleton(); must have 'critical_states'.
        replay_buffer: ReplayBuffer with .sample(batch_size) → dict of tensors.

    Returns:
        dict mapping c_id → trained HittingTimeValueNet.
    """
    state_dim       = replay_buffer.state_dim
    critical_states = skeleton_data["critical_states"]
    hitting_nets    = {}

    for c_id, c_np in critical_states.items():
        c_t = torch.tensor(c_np, dtype=torch.float32, device=device)

        value_net  = HittingTimeValueNet(state_dim).to(device)
        target_net = HittingTimeValueNet(state_dim).to(device)
        target_net.load_state_dict(value_net.state_dict())
        optimizer  = torch.optim.Adam(value_net.parameters(), lr=lr)

        # S_prototype_fn: given task_id tensor [B], return S_proto [B, state_dim]
        def S_proto_fn(task_id, _c=c_t):
            B = task_id.shape[0] if task_id.dim() > 0 else 1
            return _c.unsqueeze(0).expand(B, -1)

        for _ in range(epochs):
            train_hitting_time_value_net(
                value_net, target_net, replay_buffer,
                S_proto_fn, optimizer,
                gamma=gamma, batch_size=batch_size, tau=tau,
            )

        hitting_nets[c_id] = value_net

    return hitting_nets


# ── Sub-policy training ────────────────────────────────────────────────────

def train_sub_policy(
    sub_policy,
    replay_buffer,
    c_state: torch.Tensor,
    hitting_time_net=None,
    gamma: float = 0.99,
    epochs: int = 50,
    batch_size: int = 256,
    beta_intrinsic: float = 0.1,
    reach_bonus: float = 1.0,
    reach_threshold: float = 0.5,
    discrete: bool = True,
    on_policy_buffer=None,
    device: str = "cpu",
) -> dict:
    """
    Train a single SubPolicy option using A2C with augmented rewards.

    Augmented reward = r_env + reach_bonus * 1[||s'−c|| < δ]
    Optional intrinsic shaping: + β * (V_H(s) − V_H(s'))

    on_policy_buffer: when provided and large enough, each batch is drawn
    50% from replay_buffer (SB3 / off-policy) and 50% from on_policy_buffer
    (sub-policy rollouts). This reduces the off-policy distribution gap.

    Modifies sub_policy.policy_net and sub_policy.value_net in-place.
    Returns dict with keys pi_losses and v_losses (lists of per-step floats).
    """
    opt_pi   = torch.optim.Adam(sub_policy.policy_net.parameters(), lr=3e-4)
    opt_v    = torch.optim.Adam(sub_policy.value_net.parameters(),  lr=1e-3)
    pi_losses: list = []
    v_losses:  list = []

    c_t = c_state.to(device) if isinstance(c_state, torch.Tensor) else \
        torch.tensor(c_state, dtype=torch.float32, device=device)

    half = batch_size // 2

    for _ in range(epochs):
        if len(replay_buffer) < batch_size:
            break

        if on_policy_buffer is not None and len(on_policy_buffer) >= half:
            b_off = replay_buffer.sample(half)
            b_on  = on_policy_buffer.sample(half)
            batch = {k: torch.cat([b_off[k], b_on[k]], dim=0) for k in b_off}
        else:
            batch  = replay_buffer.sample(batch_size)
        s      = batch["state"]
        a      = batch["action"]
        r_env  = batch["reward"]
        s_next = batch["next_state"]
        done   = batch["done"]

        # Augmented reward: bonus for reaching the subgoal
        dist_c   = torch.norm(s_next - c_t, dim=-1, keepdim=True)
        r_aug    = r_env + reach_bonus * (dist_c < reach_threshold).float()

        # Optional: intrinsic shaping via hitting-time value difference
        if hitting_time_net is not None:
            c_proto = c_t.unsqueeze(0).expand(len(s), -1)
            with torch.no_grad(), eval_mode(hitting_time_net):
                r_int = hitting_time_net(s, c_proto) - hitting_time_net(s_next, c_proto)
            r_int = torch.clamp(r_int, -1.0, 1.0)
            r_total = r_aug + beta_intrinsic * r_int
        else:
            r_total = r_aug

        # TD target for sub-policy value
        with torch.no_grad(), eval_mode(sub_policy.target_value_net):
            V_next = sub_policy.target_value_net(s_next)
            target = r_total + gamma * V_next * (1.0 - done)

        # Value update
        V_cur  = sub_policy.value_net(s)
        v_loss = F.mse_loss(V_cur, target)
        opt_v.zero_grad()
        v_loss.backward()
        nn.utils.clip_grad_norm_(sub_policy.value_net.parameters(), 0.5)
        opt_v.step()

        # Recompute baseline with updated value net weights so advantage
        # reflects the current critic rather than the pre-update V_cur.
        with torch.no_grad(), eval_mode(sub_policy.value_net):
            V_cur_fresh = sub_policy.value_net(s)

        # Policy gradient
        advantage = (target - V_cur_fresh)
        # Normalise advantage within the batch to reduce variance
        advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)
        if discrete:
            a_idx = a[:, 0].long() if a.dim() > 1 else a.squeeze().long()
            log_p = sub_policy.policy_net.log_prob(s, a_idx)
        else:
            log_p = sub_policy.policy_net.log_prob(s, a)

        pi_loss = -(log_p * advantage.squeeze(-1)).mean()
        opt_pi.zero_grad()
        pi_loss.backward()
        nn.utils.clip_grad_norm_(sub_policy.policy_net.parameters(), 0.5)
        opt_pi.step()
        sub_policy._soft_update_targets()
        pi_losses.append(pi_loss.item())
        v_losses.append(v_loss.item())

    return {"pi_losses": pi_losses, "v_losses": v_losses}


# ── History-conditioned meta-policy (GRU variant) ─────────────────────────

class HistoryMetaPolicy(nn.Module):
    """
    π_θ(c | s, h) — selects subgoal conditioned on current state AND task history.

    Task history h is maintained by a GRU that processes (state, reward) pairs.
    This is a more powerful alternative to the memoryless MetaPolicy in
    algos/MetaPolicy.py, useful when tasks differ in subtle ways.
    """

    def __init__(self, state_dim: int, num_subgoals: int,
                 state_embed_dim: int = 64, history_dim: int = 128):
        super().__init__()
        self.history_dim  = history_dim
        self.state_embed  = nn.Linear(state_dim, state_embed_dim)
        self.gru          = nn.GRU(state_dim + 1, history_dim, batch_first=True)
        self.policy_head  = nn.Sequential(
            nn.Linear(state_embed_dim + history_dim, 256),
            nn.ReLU(),
            nn.Linear(256, num_subgoals),
        )

    def forward(self, s: torch.Tensor, h: torch.Tensor):
        """
        s: [B, state_dim] or [state_dim]
        h: [1, B, history_dim] GRU hidden state.
        Returns Categorical distribution and unchanged h.
        """
        single = s.dim() == 1
        if single:
            s = s.unsqueeze(0)
        s_feat  = F.relu(self.state_embed(s))              # [B, state_embed_dim]
        combined = torch.cat([s_feat, h.squeeze(0)], dim=-1)
        logits  = self.policy_head(combined)
        return torch.distributions.Categorical(logits=logits), h

    def step_history(self, h: torch.Tensor, s: torch.Tensor, r: float):
        """Update GRU hidden state with one (state, reward) observation."""
        s = s.detach()
        if s.dim() == 1:
            s = s.unsqueeze(0)
        r_t  = torch.tensor([[r]], dtype=torch.float32, device=s.device)
        inp  = torch.cat([s, r_t], dim=-1).unsqueeze(1)   # [1, 1, state_dim+1]
        _, h = self.gru(inp, h)
        return h

    def init_hidden(self, batch_size: int = 1, device: str = "cpu") -> torch.Tensor:
        return torch.zeros(1, batch_size, self.history_dim, device=device)


class HistoryMetaValueNetwork(nn.Module):
    """V(s, h) for the history-conditioned meta-policy."""

    def __init__(self, state_dim: int, history_dim: int = 128,
                 state_embed_dim: int = 64):
        super().__init__()
        self.state_embed = nn.Linear(state_dim, state_embed_dim)
        self.net = nn.Sequential(
            nn.Linear(state_embed_dim + history_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, s: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        if s.dim() == 1:
            s = s.unsqueeze(0)
        s_feat = F.relu(self.state_embed(s))
        return self.net(torch.cat([s_feat, h.squeeze(0)], dim=-1))


def train_history_meta_policy(
    sub_policies: dict,
    skeleton_data: dict,
    task_distribution,
    state_dim: int,
    gamma: float = 0.99,
    meta_epochs: int = 200,
    max_option_steps: int = 50,
    history_dim: int = 128,
    device: str = "cpu",
    verbose: bool = True,
):
    """
    Train HistoryMetaPolicy via REINFORCE with GRU history.

    Sub-policies are executed inside torch.no_grad() — their parameters
    are never updated here.

    Returns:
        (HistoryMetaPolicy, HistoryMetaValueNetwork)
    """
    subgoals     = skeleton_data["critical_states"]
    c_list       = list(subgoals.keys())
    num_subgoals = len(c_list)

    if num_subgoals == 0:
        print("[train_history_meta_policy] No subgoals; skipping.")
        return None, None

    meta_policy    = HistoryMetaPolicy(state_dim, num_subgoals,
                                       history_dim=history_dim).to(device)
    meta_value_net = HistoryMetaValueNetwork(state_dim,
                                             history_dim=history_dim).to(device)
    opt_meta  = torch.optim.Adam(meta_policy.parameters(),    lr=1e-4)
    opt_metav = torch.optim.Adam(meta_value_net.parameters(), lr=1e-3)

    for epoch in range(meta_epochs):
        task = task_distribution.sample()
        env  = task.create_env()
        obs, _ = env.reset()
        obs  = torch.tensor(np.asarray(obs, dtype=np.float32).flatten(), device=device)
        done = False
        h    = meta_policy.init_hidden(device=device)   # [1, 1, history_dim]

        episode, ep_h = [], []  # (s, c_idx, R_option, T_option)

        while not done:
            with torch.no_grad():
                dist_obj, h_next = meta_policy(obs, h)
            c_idx = dist_obj.sample().item()
            sp    = sub_policies[c_list[c_idx]]

            s_start  = obs.clone()
            h_start  = h.clone()
            R_option = 0.0
            T_c      = 0

            with torch.no_grad():
                while not sp.is_terminated(obs, done, T_c):
                    a = sp.get_action(obs)
                    a_np = a.cpu().numpy()
                    if a_np.ndim == 0:
                        a_np = a_np.item()
                    obs_next, r_env, terminated, truncated, _ = env.step(a_np)
                    done = terminated or truncated
                    obs_next = torch.tensor(
                        np.asarray(obs_next, dtype=np.float32).flatten(), device=device
                    )
                    R_option += (gamma ** T_c) * r_env
                    h = meta_policy.step_history(h, obs, r_env)
                    obs  = obs_next
                    T_c += 1
                    if done:
                        break

            episode.append((s_start, h_start, c_idx, R_option, T_c))
            ep_h.append(h_next)

        env.close()
        if not episode:
            continue

        returns = compute_discounted_returns([e[3] for e in episode], gamma)

        pi_loss_total = torch.zeros(1, device=device)
        v_loss_total  = torch.zeros(1, device=device)

        for t, (s_t, h_t, c_idx, R_t, T_t) in enumerate(episode):
            V_cur = meta_value_net(s_t, h_t)
            with torch.no_grad():
                if t < len(episode) - 1:
                    s_n, h_n = episode[t + 1][0], episode[t + 1][1]
                    V_next = meta_value_net(s_n, h_n)
                else:
                    V_next = torch.zeros(1, 1, device=device)
                target    = R_t + (gamma ** T_t) * V_next
                advantage = target - V_cur

            dist_obj, _ = meta_policy(s_t, h_t)
            log_p       = dist_obj.log_prob(torch.tensor(c_idx, device=device))
            pi_loss_total = pi_loss_total + (-log_p * advantage.squeeze().detach())
            v_loss_total  = v_loss_total  + F.mse_loss(V_cur, target.detach())

        opt_meta.zero_grad()
        pi_loss_total.backward()
        torch.nn.utils.clip_grad_norm_(meta_policy.parameters(), 0.5)
        opt_meta.step()

        opt_metav.zero_grad()
        v_loss_total.backward()
        opt_metav.step()

        if verbose and (epoch + 1) % 50 == 0:
            avg_R = np.mean([e[3] for e in episode])
            print(f"  [HistoryMeta] epoch {epoch+1}/{meta_epochs}: "
                  f"avg option return={avg_R:.4f}")

    return meta_policy, meta_value_net
