"""
Hitting-Time TD Learning (docs/HTTD.md).

Trains V_θ(s; S) — the expected discounted hitting-time cost to reach set S — via
temporal-difference learning with a Polyak-averaged target network.
"""

import torch
import torch.nn.functional as F

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils.utils import eval_mode


def train_hitting_time_value_net(
    value_net,
    target_net,
    replay_buffer,
    S_prototype_fn,   # Callable[[task_id_tensor [B]], S_proto [B, state_dim]]
    optimizer,
    gamma: float = 0.99,
    threshold: float = 0.1,
    batch_size: int = 256,
    tau: float = 0.005,
    reward_normalizer=None,
    target_clip: float = None,
):
    """
    Single gradient step of hitting-time TD.

    Args:
        value_net:          V_θ(s, S_proto) network being trained.
        target_net:         V_θ⁻ Polyak-averaged copy; not updated by optimizer.
        replay_buffer:      ReplayBuffer with .sample(batch_size) → dict of tensors.
        S_prototype_fn:     Maps task_id [B] → S_proto [B, state_dim].
        optimizer:          Adam (or similar) wrapping value_net.parameters().
        gamma:              Discount factor.
        threshold:          Distance below which s_next is considered inside S.
        batch_size:         Transitions per gradient step.
        tau:                Polyak averaging rate for target network.
        reward_normalizer:  Optional normalizer with .normalize(reward_tensor) method.
        target_clip:        Symmetric clamp applied to the TD target.

    Returns:
        float: scalar loss value for logging.
    """
    if len(replay_buffer) < batch_size:
        return float("nan")

    batch   = replay_buffer.sample(batch_size)
    s       = batch["state"]        # [B, state_dim]
    r       = batch["reward"]       # [B, 1]
    s_next  = batch["next_state"]   # [B, state_dim]
    task_id = batch["task_id"]      # [B]

    # Skip single-sample batches — BatchNorm requires B > 1
    if s.shape[0] <= 1:
        return float("nan")

    S_proto = S_prototype_fn(task_id)  # [B, state_dim]

    # Optionally normalize rewards before TD target computation
    if reward_normalizer is not None:
        r = (r - reward_normalizer.mean) / (reward_normalizer.std + 1e-8)

    with torch.no_grad(), eval_mode(target_net):
        dist_next = torch.norm(s_next - S_proto, dim=-1, keepdim=True)
        in_S_next = (dist_next < threshold).float()

        V_next = target_net(s_next, S_proto)
        target = r + gamma * V_next * (1.0 - in_S_next)
        if target_clip is not None:
            target = torch.clamp(target, -target_clip, target_clip)

    V_current = value_net(s, S_proto)
    loss = F.huber_loss(V_current, target, delta=1.0)

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(value_net.parameters(), 1.0)
    optimizer.step()

    with torch.no_grad():
        for p, tp in zip(value_net.parameters(), target_net.parameters()):
            tp.data.mul_(1.0 - tau).add_(p.data, alpha=tau)

    return loss.item()
