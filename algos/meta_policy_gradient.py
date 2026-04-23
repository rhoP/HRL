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


torch.set_default_dtype(torch.float32)

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from algos.potential import compute_sparse_shaped_reward
from models.meta_policy_net import MetaValueNetwork
from utils.utils import compute_discounted_returns


# ── Running return normalizer ──────────────────────────────────────────────

class RunningMeanStd:
    """Welford online estimator for mean and variance of a scalar stream.

    Persists across iterations so the normalizer tracks the full lifetime
    scale of returns, decoupling value-loss magnitude from shaping schedule.
    """

    def __init__(self, eps: float = 1e-4):
        self.mean  = 0.0
        self.var   = 1.0
        self.count = eps  # seed avoids div-by-zero before first update

    def update(self, x: torch.Tensor) -> None:
        x_flat      = x.detach().cpu().float().view(-1)
        batch_count = x_flat.numel()
        batch_mean  = float(x_flat.mean())
        batch_var   = float(x_flat.var()) if batch_count > 1 else 0.0

        total       = self.count + batch_count
        delta       = batch_mean - self.mean
        self.mean   = self.mean + delta * batch_count / total
        self.var    = (self.var * self.count
                       + batch_var * batch_count
                       + delta ** 2 * self.count * batch_count / total) / total
        self.count  = total

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / (self.var ** 0.5 + 1e-8)


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


def evaluate_meta_policy_per_task(
    meta_policy,
    task_distribution,
    n_episodes: int = 20,
    max_steps: int  = 500,
    device: str     = "cpu",
) -> dict:
    """Return {task_id: success_rate} for every task in the distribution.

    Runs exactly n_episodes episodes per task (not sampled randomly) so each
    task gets a fair, independent evaluation.
    """
    meta_policy.eval()
    per_task = {}

    for task in task_distribution.tasks:
        successes = []
        for _ in range(n_episodes):
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
                    h       = meta_policy.update_hidden(s_arr, a_np, r_env, h)
                    s       = s_next
                    t      += 1

            env.close()
            successes.append(float(success))

        per_task[task.id] = float(np.mean(successes))

    meta_policy.train()
    return per_task


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
    meta_epochs: int          = 1000,
    episodes_per_update: int  = 4,
    gamma: float              = 0.99,
    shaping_scale: float      = 1.0,
    subgoal_threshold: float  = float("inf"),
    entropy_coef: float       = 0.01,
    lr: float                 = 3e-4,
    max_episode_steps: int    = 500,
    eval_every: int           = 50,
    eval_episodes: int        = 10,
    is_buffer_size: int       = 16,
    is_clip_epsilon: float    = 0.2,
    replay_buffer             = None,
    flush_buffer: bool        = False,
    flush_optimizer: bool     = False,
    device: str               = "cpu",
    verbose: bool             = True,
    training_state: dict      = None,
    graduation_callback       = None,
    # ── Smooth shaping ─────────────────────────────────────────────────
    buffer_warmup_epochs: int  = -1,
    adaptive_shaping: bool     = True,
    max_ratio_deviation: float = 0.3,
    shaping_adapt_rate: float  = 0.5,
):
    """
    Train meta-policy using IS-corrected policy gradient with skeleton-based
    reward shaping.

    Episodes are collected under the current behaviour policy (no_grad) and
    stored in a rolling TrajectoryBuffer of size `is_buffer_size`.  At each
    gradient step the current policy is re-evaluated via evaluate_actions,
    the IS ratio ρ = π_θ(a|s,h) / π_old(a|s,h) is computed and clipped to
    [1−ε, 1+ε] (PPO-style), and the clipped surrogate loss is minimised.

    Smooth-shaping parameters
    ─────────────────────────
    buffer_warmup_epochs (-1 = auto)
        Epochs at the start during which episodes are collected but NO gradient
        update is applied.  Auto sets this to ceil(is_buffer_size/episodes_per_update)
        so the first gradient step always operates on a full IS buffer, removing
        the noisy-first-update jump.

    adaptive_shaping (default True)
        Tracks an EMA of |mean(IS-ratio) − 1| after each gradient step.  When
        the EMA exceeds max_ratio_deviation the effective shaping scale is
        reduced toward shaping_adapt_rate × shaping_scale (floor).  If the
        scale change is >10% the IS buffer is flushed so stale returns (computed
        under the old scale) are discarded before the next update.

    training_state carries {meta_value_net, policy_optimizer, value_optimizer,
    is_buffer, ema_ratio_dev, effective_shaping} across iterations.  On
    flush_buffer=True (new iteration) the EMA and effective_shaping are reset
    to 0 / shaping_scale so each iteration starts fresh.
    """
    meta_subgoals      = skeleton_data.get("meta_subgoals")
    skeleton_potential = skeleton_data.get("skeleton_potential")

    n_tasks             = len(task_distribution.tasks)
    episodes_per_update = max(1, min(episodes_per_update, n_tasks))

    # Auto warmup: fill the IS buffer completely before the first gradient step.
    if buffer_warmup_epochs < 0:
        buffer_warmup_epochs = max(
            0, -(-is_buffer_size // episodes_per_update)  # ceil division
        )

    state_dim = meta_policy.state_dim
    ts        = training_state or {}

    meta_value_net = (ts.get("meta_value_net") or
                      MetaValueNetwork(state_dim, meta_policy.gru_hidden).to(device))

    policy_optimizer = (ts.get("policy_optimizer") or
                        torch.optim.Adam(meta_policy.parameters(), lr=lr))

    value_optimizer  = (ts.get("value_optimizer") or
                        torch.optim.Adam(meta_value_net.parameters(), lr=lr))

    is_buffer = (ts.get("is_buffer") or
                 TrajectoryBuffer(max_episodes=is_buffer_size))

    # Return normalizer persists across iterations — intentionally not reset on
    # flush so the running stats track the full lifetime scale of returns.
    returns_normalizer = ts.get("returns_normalizer") or RunningMeanStd()

    # Adaptive-shaping state: reset on each flush (new potential = new iteration).
    if flush_buffer:
        is_buffer = TrajectoryBuffer(max_episodes=is_buffer_size)
        value_optimizer.state.clear()
        ema_ratio_dev    = 0.0
        effective_shaping = shaping_scale
    else:
        ema_ratio_dev    = float(ts.get("ema_ratio_dev",    0.0))
        effective_shaping = float(ts.get("effective_shaping", shaping_scale))

    if flush_optimizer:
        policy_optimizer.state.clear()
        value_optimizer.state.clear()

    _ema_alpha = 0.15   # EMA smoothing for IS ratio deviation

    epoch_losses: list = []
    meta_policy.train()
    meta_value_net.train()

    task_list   = list(task_distribution.tasks)
    task_cursor = 0

    for epoch in range(meta_epochs):

        # ── Adaptive shaping: adjust effective_shaping before this collection ─
        if adaptive_shaping and epoch > 0:
            # shaping_factor ∈ [shaping_adapt_rate, 1.0]: decreases when IS
            # ratios indicate the policy has drifted far from behaviour.
            excess = max(0.0, ema_ratio_dev - max_ratio_deviation)
            shaping_factor = max(shaping_adapt_rate, 1.0 - excess)
            new_effective  = shaping_scale * shaping_factor

            # If scale would shift by >10% the buffered returns are stale —
            # flush before collecting at the new scale.
            if effective_shaping > 1e-8:
                change_frac = abs(new_effective - effective_shaping) / effective_shaping
            else:
                change_frac = 1.0
            if change_frac > 0.10:
                is_buffer = TrajectoryBuffer(max_episodes=is_buffer_size)
                value_optimizer.state.clear()
                if verbose:
                    print(f"  [PG] epoch {epoch}: adaptive flush  "
                          f"scale {effective_shaping:.3f}→{new_effective:.3f}  "
                          f"ema_ratio_dev={ema_ratio_dev:.3f}")
                effective_shaping = new_effective

        # ── Collect fresh episodes (behaviour policy, no gradients) ──────────
        fresh_rewards: list = []

        for _ in range(episodes_per_update):
            task = task_list[task_cursor % n_tasks]
            task_cursor += 1

            ep = _collect_episode(
                meta_policy, task,
                meta_subgoals, skeleton_potential,
                gamma, effective_shaping, subgoal_threshold,
                max_episode_steps, device,
            )
            if ep is None:
                continue

            is_buffer.add(ep)
            fresh_rewards.extend(ep["shaped_rewards"])

            if replay_buffer is not None:
                for s_arr, a_np, r_env, s_next_arr, done, terminated in ep["transitions"]:
                    replay_buffer.push(
                        s_arr, a_np, r_env, s_next_arr, done, ep["task_id"],
                        terminated=terminated,
                    )

        # ── Warmup: skip gradient until IS buffer is full ─────────────────────
        if epoch < buffer_warmup_epochs:
            continue

        # ── Gradient update using full IS buffer ──────────────────────────────
        batch = is_buffer.get_batch()
        if batch is None:
            continue

        curr_lp, curr_ent = meta_policy.evaluate_actions(
            batch["states_t"], batch["actions_t"], batch["hidden_states_t"]
        )

        # Normalize returns: update running stats then map to ~N(0,1) scale.
        # Value network learns to predict normalized targets; advantages are
        # computed in the same space, keeping loss magnitude scale-invariant.
        returns_normalizer.update(batch["returns_t"])
        returns_norm = returns_normalizer.normalize(batch["returns_t"])

        values = meta_value_net(
            batch["states_t"], batch["hidden_states_t"]
        ).squeeze(-1)
        adv = returns_norm - values.detach()
        adv_std = adv.std()
        if adv_std > 1e-6:
            adv = (adv - adv.mean()) / (adv_std + 1e-8)

        log_ratio     = curr_lp - batch["log_probs_old_t"].detach()
        ratio         = log_ratio.exp()
        ratio_clipped = ratio.clamp(1.0 - is_clip_epsilon, 1.0 + is_clip_epsilon)

        policy_loss  = -torch.min(ratio * adv, ratio_clipped * adv).mean()
        value_loss   = F.mse_loss(values, returns_norm)
        entropy_loss = curr_ent.mean()
        total_loss   = policy_loss + 0.5 * value_loss - entropy_coef * entropy_loss

        value_optimizer.zero_grad()
        (0.5 * value_loss).backward()
        torch.nn.utils.clip_grad_norm_(meta_value_net.parameters(), 1.0)
        value_optimizer.step()

        policy_optimizer.zero_grad()
        (policy_loss - entropy_coef * entropy_loss).backward()
        torch.nn.utils.clip_grad_norm_(meta_policy.parameters(), 1.0)
        policy_optimizer.step()

        # Update EMA of IS ratio deviation for next epoch's adaptive shaping.
        with torch.no_grad():
            ratio_dev_t = (ratio.mean() - 1.0).abs().item()
        ema_ratio_dev = (1.0 - _ema_alpha) * ema_ratio_dev + _ema_alpha * ratio_dev_t

        epoch_losses.append({
            "total":            float(total_loss.item()),
            "policy":           float(policy_loss.item()),
            "value":            float(value_loss.item()),
            "entropy":          float(entropy_loss.item()),
            "effective_shaping": float(effective_shaping),
            "ema_ratio_dev":    float(ema_ratio_dev),
        })

        if (epoch + 1) % eval_every == 0:
            sr = evaluate_meta_policy(
                meta_policy, task_distribution,
                n_episodes=eval_episodes, device=device,
                meta_subgoals=meta_subgoals,
                skeleton_potential=skeleton_potential,
                shaping_scale=shaping_scale,
                subgoal_threshold=subgoal_threshold,
            )
            if verbose:
                avg_r      = float(np.mean(fresh_rewards)) if fresh_rewards else 0.0
                mean_ratio = float(ratio.mean().item())
                n_buf      = len(is_buffer)
                print(f"  [PG] epoch {epoch+1}/{meta_epochs}  "
                      f"buf={n_buf}ep  "
                      f"scale={effective_shaping:.3f}  "
                      f"IS_ratio={mean_ratio:.3f}  ema_dev={ema_ratio_dev:.3f}  "
                      f"loss={total_loss.item():.4f}  "
                      f"avg_r={avg_r:.4f}  sr={sr:.1%}")

            # Per-task graduation check
            per_task_sr = evaluate_meta_policy_per_task(
                meta_policy, task_distribution,
                n_episodes=eval_episodes, device=device,
            )
            newly_graduated = [
                t for t in task_distribution.tasks
                if per_task_sr.get(t.id, 0.0) >= 1.0
            ]
            if newly_graduated:
                if graduation_callback is not None:
                    graduation_callback(newly_graduated)
                # Refresh round-robin list after task_distribution may have changed
                task_list = list(task_distribution.tasks)
                n_tasks   = len(task_list)
                task_cursor = 0
                if n_tasks == 0:
                    break

    return meta_policy, meta_value_net, epoch_losses, {
        "meta_value_net":      meta_value_net,
        "policy_optimizer":    policy_optimizer,
        "value_optimizer":     value_optimizer,
        "is_buffer":           is_buffer,
        "ema_ratio_dev":       ema_ratio_dev,
        "effective_shaping":   effective_shaping,
        "returns_normalizer":  returns_normalizer,
    }
