"""
Shared utility functions used across the meta-RL pipeline.
"""

import contextlib

import numpy as np
import torch


# ── Inference helper ───────────────────────────────────────────────────────

@contextlib.contextmanager
def eval_mode(*nets):
    """
    Context manager that switches one or more nn.Module objects to eval mode,
    then restores each module's original training state on exit.

    Use this at every inference call site for nets that contain BatchNorm or
    Dropout layers — calling them in training mode produces batch-dependent
    outputs that corrupt value estimates.

    Usage:
        with torch.no_grad(), eval_mode(net):
            y = net(x)

        with torch.no_grad(), eval_mode(net_a, net_b):
            ya = net_a(x)
            yb = net_b(x)
    """
    was_training = [net.training for net in nets]
    try:
        for net in nets:
            net.eval()
        yield
    finally:
        for net, was in zip(nets, was_training):
            net.train(was)


# ── Hitting-time helpers ───────────────────────────────────────────────────

def is_in_set(s: torch.Tensor, S_prototype: torch.Tensor, threshold: float = 0.1) -> torch.Tensor:
    """
    Returns True (per batch element) when s is within `threshold` of S_prototype.

    s, S_prototype: broadcastable tensors (e.g. [B, d] and [d] or [B, d]).
    """
    return torch.norm(s - S_prototype, dim=-1) < threshold


def find_first_hit(trajectory: dict, S_prototype, threshold: float = 0.1):
    """
    Find the step index at which next_state first enters S.

    Args:
        trajectory:  dict with key 'states' of shape [T+1, state_dim].
        S_prototype: array-like or tensor of shape [state_dim].

    Returns:
        int: index t such that states[t+1] ∈ S, or None if never reached.
    """
    states    = trajectory["states"]          # [T+1, state_dim]
    next_states = torch.tensor(
        np.asarray(states[1:], dtype=np.float32)
    )
    proto = torch.tensor(np.asarray(S_prototype, dtype=np.float32))
    dists = torch.norm(next_states - proto, dim=-1)   # [T]
    hits  = (dists < threshold).nonzero(as_tuple=False)
    return int(hits[0].item()) if len(hits) > 0 else None


# ── Return computations ────────────────────────────────────────────────────

def compute_discounted_returns(rewards, gamma: float) -> list:
    """
    Standard forward-discounted returns G_t = r_t + γ G_{t+1}.

    Args:
        rewards: list or 1-D array of scalar rewards.

    Returns:
        list of floats, same length as rewards.
    """
    returns = []
    G = 0.0
    for r in reversed(rewards):
        G = float(r) + gamma * G
        returns.insert(0, G)
    return returns


# ── Skeleton refinement ────────────────────────────────────────────────────

def refine_skeleton(skeleton_data: dict, replay_buffer, **kwargs) -> dict:
    """
    Delegate to utils/skeleton.py::refine_skeleton for a clean call site.
    Accepts the same keyword arguments as build_skeleton().
    """
    from utils.skeleton import refine_skeleton as _refine
    return _refine(skeleton_data, replay_buffer, **kwargs)


# ── Data collection helper ─────────────────────────────────────────────────

def collect_initial_data(task_distribution, num_episodes: int = 200,
                         max_steps: int = 500, device: str = "cpu"):
    """
    Collect random-policy episodes from a task distribution.

    Args:
        task_distribution: object with .sample() → Task (has .create_env(), .id).
        num_episodes:      number of episodes to collect.
        max_steps:         per-episode step cap.
        device:            device string for the returned ReplayBuffer.

    Returns:
        ReplayBuffer populated with transitions.
    """
    from utils.replay_buffer import ReplayBuffer

    rb = ReplayBuffer(device=device)
    for _ in range(num_episodes):
        task = task_distribution.sample()
        env  = task.create_env()
        obs, _ = env.reset()
        obs = np.asarray(obs, dtype=np.float32).flatten()
        for _ in range(max_steps):
            a = env.action_space.sample()
            obs_next, r, terminated, truncated, _ = env.step(a)
            obs_next = np.asarray(obs_next, dtype=np.float32).flatten()
            done = terminated or truncated
            rb.push(obs, a, r, obs_next, done, task.id)
            obs = obs_next
            if done:
                break
        env.close()
    return rb
