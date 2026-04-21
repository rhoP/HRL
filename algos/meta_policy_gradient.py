"""
Meta-policy gradient training with skeleton-based reward shaping and
importance-sampled (IS) off-policy correction.

Entry point:
    meta_policy_gradient_with_skeleton_shaping(meta_policy, task_distribution,
                                               skeleton_data, ...)

skeleton_data must contain:
    "meta_subgoals"       — {sg_id: {"state": np.ndarray}}
    "skeleton_potential"  — SkeletonPotential or EmpiricalHittingTimePotential

Collection is fully off-policy: each episode is collected under torch.no_grad()
and stored in a rolling TrajectoryBuffer together with the behaviour log-probs
and frozen GRU context.  At each gradient step the current policy is
re-evaluated via MetaPolicy.evaluate_actions, the IS ratio is computed and
clipped (PPO-style), and the clipped surrogate loss is minimised.

Returns (meta_policy, meta_value_net, epoch_losses).
"""

import os
import sys
from collections import deque

import numpy as np
import torch
import torch.nn.functional as F

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from algos.potential import compute_sparse_shaped_reward
from models.meta_policy_net import MetaValueNetwork
from utils.utils import compute_discounted_returns


# ── IS trajectory buffer ───────────────────────────────────────────────────

class TrajectoryBuffer:
    """
    Rolling buffer of recent episodes for importance-sampled policy gradient.
    Stores the last `max_episodes` collected episodes; older ones are evicted.
    Each episode dict must contain:
        states_t, actions_t, hidden_states_t, log_probs_old_t, returns_t
    """

    def __init__(self, max_episodes: int = 16):
        self._episodes: deque = deque(maxlen=max_episodes)

    def add(self, ep: dict) -> None:
        self._episodes.append(ep)

    def __len__(self) -> int:
        return len(self._episodes)

    def get_batch(self) -> dict | None:
        """Concatenate all stored episodes into a single flat batch."""
        if not self._episodes:
            return None
        keys = ("states_t", "actions_t", "hidden_states_t",
                "log_probs_old_t", "returns_t")
        return {k: torch.cat([e[k] for e in self._episodes], dim=0) for k in keys}


# ── Evaluation ─────────────────────────────────────────────────────────────

def evaluate_meta_policy(
    meta_policy,
    task_distribution,
    n_episodes: int          = 20,
    max_steps: int           = 500,
    device: str              = "cpu",
    meta_subgoals: dict      = None,
    skeleton_potential       = None,
    shaping_scale: float     = 1.0,
    subgoal_threshold: float = float("inf"),
) -> float:
    """Run n_episodes with the meta-policy; return mean success rate.

    When meta_subgoals and skeleton_potential are supplied the shaped reward
    is used as the τ signal, matching the training-time GRU input exactly.
    Success is still measured on raw env reward only.
    """
    use_shaping = (meta_subgoals is not None) and (skeleton_potential is not None)
    meta_policy.eval()
    successes = []

    for _ in range(n_episodes):
        task   = task_distribution.sample()
        env    = task.create_env()
        result = env.reset()
        s      = result[0] if isinstance(result, tuple) else result
        done   = False
        t      = 0
        success = False

        with torch.no_grad():
            h = meta_policy.init_hidden(device)

            while not done and t < max_steps:
                s_arr  = np.asarray(s, dtype=np.float32)
                a_dist = meta_policy.forward_with_hidden(s_arr, h)
                a      = a_dist.sample()
                a_np   = a.cpu().numpy().flatten()
                a_env  = int(a_np[0]) if meta_policy.discrete else a_np

                s_next, r_env, terminated, truncated, info = env.step(a_env)
                done    = terminated or truncated
                success = success or bool(info.get("success", 0.0) > 0.5)

                if use_shaping:
                    r_tau, _ = compute_sparse_shaped_reward(
                        s_arr, np.asarray(s_next, dtype=np.float32),
                        r_env, meta_subgoals, skeleton_potential,
                        shaping_scale, subgoal_threshold,
                    )
                else:
                    r_tau = r_env

                h = meta_policy.update_hidden(s_arr, a_np, r_tau, h)
                s = s_next
                t += 1

        env.close()
        successes.append(float(success))

    meta_policy.train()
    return float(np.mean(successes))


# ── Episode collection (behaviour policy, no gradients) ────────────────────

@torch.no_grad()
def _collect_episode(
    meta_policy,
    task,
    meta_subgoals: dict,
    skeleton_potential,
    gamma: float,
    shaping_scale: float,
    subgoal_threshold: float,
    max_episode_steps: int,
    device: str,
) -> dict | None:
    """
    Roll out one episode for `task` under the current meta_policy with no
    gradient tracking (behaviour policy).

    Returns a dict with:
        states_t        Tensor [T, state_dim]
        actions_t       Tensor [T] (discrete) or [T, action_dim] (continuous)
        hidden_states_t Tensor [T, gru_hidden]   — frozen behaviour context
        log_probs_old_t Tensor [T]               — behaviour log-probs (detached)
        returns_t       Tensor [T]               — discounted shaped returns
        shaped_rewards  list[float]
        env_rewards     list[float]              — raw env rewards for replay buffer
        transitions     list[(s, a_np, r_env, s_next, done)]
        task_id         int
    or None if the episode produced no steps.
    """
    env    = task.create_env()
    result = env.reset()
    s      = result[0] if isinstance(result, tuple) else result
    h      = meta_policy.init_hidden(device)
    done   = False

    states:         list = []
    actions:        list = []
    hidden_states:  list = []
    log_probs_old:  list = []
    shaped_rewards: list = []
    env_rewards:    list = []
    transitions:    list = []

    while not done and len(states) < max_episode_steps:
        s_arr  = np.asarray(s, dtype=np.float32)
        a_dist = meta_policy.forward_with_hidden(s_arr, h)
        a      = a_dist.sample()
        lp     = a_dist.log_prob(a)
        if lp.dim() > 0:
            lp = lp.sum()

        a_np  = a.cpu().numpy().flatten()
        a_env = int(a_np[0]) if meta_policy.discrete else a_np

        s_next, r_env, terminated, truncated, _ = env.step(a_env)
        done       = terminated or truncated
        s_next_arr = np.asarray(s_next, dtype=np.float32)

        r_shaped, _ = compute_sparse_shaped_reward(
            s_arr, s_next_arr,
            r_env, meta_subgoals, skeleton_potential,
            shaping_scale, subgoal_threshold,
        )

        states.append(torch.tensor(s_arr, dtype=torch.float32, device=device))
        # squeeze(0): [1] → [] (discrete scalar) or [1, ad] → [ad] (continuous)
        actions.append(a.detach().squeeze(0))
        hidden_states.append(h.detach().squeeze(0))   # [gru_hidden]
        log_probs_old.append(lp)
        shaped_rewards.append(r_shaped)
        env_rewards.append(float(r_env))
        transitions.append((s_arr.copy(), a_np, float(r_env),
                            s_next_arr.copy(), done, bool(terminated)))

        h = meta_policy.update_hidden(s_arr, a_np, r_shaped, h)
        s = s_next

    env.close()

    if not states:
        return None

    returns_list = compute_discounted_returns(shaped_rewards, gamma)
    return {
        "states_t":        torch.stack(states),
        "actions_t":       torch.stack(actions),
        "hidden_states_t": torch.stack(hidden_states),
        "log_probs_old_t": torch.stack(log_probs_old),
        "returns_t":       torch.tensor(returns_list, dtype=torch.float32, device=device),
        "shaped_rewards":  shaped_rewards,
        "env_rewards":     env_rewards,
        "transitions":     transitions,
        "task_id":         task.id,
    }


# ── Entry point ────────────────────────────────────────────────────────────

def meta_policy_gradient_with_skeleton_shaping(
    meta_policy,
    task_distribution,
    skeleton_data: dict,
    meta_epochs: int         = 1000,
    episodes_per_update: int = 4,
    gamma: float             = 0.99,
    shaping_scale: float     = 1.0,
    subgoal_threshold: float = float("inf"),
    entropy_coef: float      = 0.01,
    lr: float                = 3e-4,
    max_episode_steps: int   = 500,
    eval_every: int          = 50,
    eval_episodes: int       = 10,
    is_buffer_size: int      = 16,
    is_clip_epsilon: float   = 0.2,
    replay_buffer            = None,
    flush_buffer: bool       = False,
    flush_optimizer: bool    = False,
    device: str              = "cpu",
    verbose: bool            = True,
    training_state: dict     = None,
):
    """
    Train meta-policy using IS-corrected policy gradient with skeleton-based
    reward shaping.

    Episodes are collected under the current behaviour policy (no_grad) and
    stored in a rolling TrajectoryBuffer of size `is_buffer_size`.  At each
    gradient step the current policy is re-evaluated via evaluate_actions,
    the IS ratio ρ = π_θ(a|s,h) / π_old(a|s,h) is computed and clipped to
    [1−ε, 1+ε] (PPO-style), and the clipped surrogate loss is minimised.

    If `replay_buffer` is provided each collected transition (using raw env
    reward) is pushed to it, so Phase 1 skeleton data grows during training.

    `training_state` carries {meta_value_net, optimizer, is_buffer} across
    outer-loop iterations so Adam moments, baseline weights, and buffered
    episodes are not thrown away between calls.  Pass None on the first call;
    pass back the returned dict on subsequent calls.

    `flush_buffer` should be True whenever the shaped-reward potential has
    changed (i.e. every outer iteration), so that stale returns from the
    previous potential do not corrupt the IS correction.  It discards the
    episode buffer and clears Adam's first/second moment accumulators.

    Returns:
        meta_policy      — trained policy (same object, in-place)
        meta_value_net   — trained value baseline
        epoch_losses     — list of per-epoch loss component dicts
        training_state   — dict to pass back on the next call for continuity
    """
    meta_subgoals      = skeleton_data["meta_subgoals"]
    skeleton_potential = skeleton_data["skeleton_potential"]

    n_tasks             = len(task_distribution.tasks)
    episodes_per_update = max(1, min(episodes_per_update, n_tasks))

    state_dim = meta_policy.state_dim
    ts        = training_state or {}

    meta_value_net = (ts.get("meta_value_net") or
                      MetaValueNetwork(state_dim, meta_policy.gru_hidden).to(device))

    if ts.get("optimizer") is not None:
        optimizer = ts["optimizer"]
    else:
        optimizer = torch.optim.Adam(
            list(meta_policy.parameters()) + list(meta_value_net.parameters()),
            lr=lr,
        )

    is_buffer = (ts.get("is_buffer") or
                 TrajectoryBuffer(max_episodes=is_buffer_size))

    if flush_buffer:
        is_buffer = TrajectoryBuffer(max_episodes=is_buffer_size)

    if flush_optimizer:
        optimizer.state.clear()

    epoch_losses = []
    meta_policy.train()
    meta_value_net.train()

    task_list   = list(task_distribution.tasks)
    task_cursor = 0

    for epoch in range(meta_epochs):
        # ── Collect fresh episodes (no gradient tracking) ─────────────────
        fresh_rewards: list = []

        for _ in range(episodes_per_update):
            task = task_list[task_cursor % n_tasks]
            task_cursor += 1

            ep = _collect_episode(
                meta_policy, task,
                meta_subgoals, skeleton_potential,
                gamma, shaping_scale, subgoal_threshold,
                max_episode_steps, device,
            )
            if ep is None:
                continue

            is_buffer.add(ep)
            fresh_rewards.extend(ep["shaped_rewards"])

            # Push raw-reward transitions to caller's replay buffer
            if replay_buffer is not None:
                for s_arr, a_np, r_env, s_next_arr, done, terminated in ep["transitions"]:
                    replay_buffer.push(
                        s_arr, a_np, r_env, s_next_arr, done, ep["task_id"],
                        terminated=terminated,
                    )

        # ── Gradient update using full IS buffer ──────────────────────────
        batch = is_buffer.get_batch()
        if batch is None:
            continue

        # Re-evaluate current policy on stored (s, a, h_frozen) — gradients flow
        curr_lp, curr_ent = meta_policy.evaluate_actions(
            batch["states_t"], batch["actions_t"], batch["hidden_states_t"]
        )

        # Value baseline (gradients flow through values → value_loss only)
        values = meta_value_net(
            batch["states_t"], batch["hidden_states_t"]
        ).squeeze(-1)
        adv = batch["returns_t"] - values.detach()
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        # IS ratio with PPO clipping
        log_ratio     = curr_lp - batch["log_probs_old_t"].detach()
        ratio         = log_ratio.exp()
        ratio_clipped = ratio.clamp(1.0 - is_clip_epsilon, 1.0 + is_clip_epsilon)

        policy_loss  = -torch.min(ratio * adv, ratio_clipped * adv).mean()
        value_loss   = F.mse_loss(values, batch["returns_t"])
        entropy_loss = curr_ent.mean()
        total_loss   = policy_loss + 0.5 * value_loss - entropy_coef * entropy_loss

        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(meta_policy.parameters(),    1.0)
        torch.nn.utils.clip_grad_norm_(meta_value_net.parameters(), 1.0)
        optimizer.step()

        epoch_losses.append({
            "total":   float(total_loss.item()),
            "policy":  float(policy_loss.item()),
            "value":   float(value_loss.item()),
            "entropy": float(entropy_loss.item()),
        })

        if verbose and (epoch + 1) % eval_every == 0:
            sr = evaluate_meta_policy(
                meta_policy, task_distribution,
                n_episodes=eval_episodes, device=device,
                meta_subgoals=meta_subgoals,
                skeleton_potential=skeleton_potential,
                shaping_scale=shaping_scale,
                subgoal_threshold=subgoal_threshold,
            )
            avg_r      = float(np.mean(fresh_rewards)) if fresh_rewards else 0.0
            mean_ratio = float(ratio.mean().item())
            n_buf      = len(is_buffer)
            print(f"  [PG] epoch {epoch+1}/{meta_epochs}  "
                  f"buf={n_buf}ep  "
                  f"loss={total_loss.item():.4f}  "
                  f"entropy={entropy_loss.item():.3f}  "
                  f"IS_ratio={mean_ratio:.3f}  "
                  f"avg_shaped_r={avg_r:.4f}  "
                  f"success_rate={sr:.1%}")

    return meta_policy, meta_value_net, epoch_losses, {
        "meta_value_net": meta_value_net,
        "optimizer":      optimizer,
        "is_buffer":      is_buffer,
    }
