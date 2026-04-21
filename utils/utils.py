"""
Shared utility functions used across the meta-RL pipeline.
"""

import numpy as np
import torch


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
