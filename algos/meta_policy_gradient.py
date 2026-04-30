"""
Meta-policy gradient training with advantage-weighted regression (AWR) and
a cross-task replay buffer.

Entry point:
    meta_policy_gradient_with_skeleton_shaping(meta_policy, task_distribution,
                                               skeleton_data, ...)

skeleton_data must contain:
    "meta_subgoals"       — {sg_id: {"state": np.ndarray}}
    "skeleton_potential"  — SkeletonPotential or EmpiricalHittingTimePotential

Each epoch:
  1. Collect `episodes_per_update` episodes under the current policy (round-robin
     over tasks) and add them to MetaPolicyBuffer.
  2. Sample `episodes_per_batch` episodes (50% current task, 50% other tasks).
  3. Re-roll the GRU for each episode, compute normalised advantages, and apply
     advantage-weighted policy gradient (no IS clipping).

Advantage weighting naturally suppresses stale / low-quality episodes, so the
buffer can persist across iterations without poisoning training.

Returns (meta_policy, meta_value_net, epoch_losses, training_state).
"""

import os
import sys
import random
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
    Used for per-task return normalization in AWR.
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


class EMANormalizer:
    """Exponential moving average normalizer for per-step reward streams.

    Unlike Welford, old observations decay so the normalizer adapts to
    shifting reward distributions as the policy improves.  Used for the
    per-task reward channel fed to the GRU; NOT used for return normalization
    (returns use RunningMeanStd so the value baseline stays stable).

    alpha=0.99 → effective window of ~100 steps.
    """

    def __init__(self, alpha: float = 0.99, eps: float = 1e-4):
        self.alpha        = alpha
        self.eps          = eps
        self.mean         = 0.0
        self.var          = 1.0
        self._initialized = False

    def update(self, x: torch.Tensor) -> None:
        x_flat     = x.detach().cpu().float().view(-1)
        batch_mean = float(x_flat.mean())
        batch_var  = float(x_flat.var()) if x_flat.numel() > 1 else 0.0
        if not self._initialized:
            self.mean         = batch_mean
            self.var          = max(batch_var, self.eps)
            self._initialized = True
            return
        delta     = batch_mean - self.mean
        self.mean = self.alpha * self.mean + (1.0 - self.alpha) * batch_mean
        self.var  = self.alpha * self.var  + (1.0 - self.alpha) * (batch_var + delta ** 2)
        self.var  = max(self.var, self.eps)

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / (self.var ** 0.5 + 1e-8)


# ── AWR replay buffer ──────────────────────────────────────────────────────

class MetaPolicyBuffer:
    """
    Rolling replay buffer for advantage-weighted meta-policy training.

    Internally stores (task_id, iteration_id, episode_dict) triples up to
    `capacity` (FIFO eviction).  The iteration_id allows purging episodes
    collected under a previous potential (which has a different return scale)
    at the start of each outer iteration.

    sample_batch returns (task_id, episode) pairs — iteration_id is internal.
    """

    def __init__(self, capacity: int = 10_000):
        self.buffer: deque = deque(maxlen=capacity)

    def add(self, task_id: int, episode: dict, iteration_id: int = 0) -> None:
        self.buffer.append((task_id, iteration_id, episode))

    def purge_before(self, iteration_id: int) -> int:
        """Discard episodes whose iteration_id < iteration_id. Returns count removed."""
        before = len(self.buffer)
        self.buffer = deque(
            ((t, iid, ep) for t, iid, ep in self.buffer if iid >= iteration_id),
            maxlen=self.buffer.maxlen,
        )
        return before - len(self.buffer)

    def __len__(self) -> int:
        return len(self.buffer)

    def sample_batch(
        self,
        batch_size: int,
        current_task: int = None,
    ) -> list:
        """Return up to batch_size (task_id, episode) pairs."""
        if not self.buffer:
            return []
        if current_task is not None:
            task_eps  = [(t, ep) for t, iid, ep in self.buffer if t == current_task]
            other_eps = [(t, ep) for t, iid, ep in self.buffer if t != current_task]
            n_current = min(len(task_eps),  batch_size // 2)
            n_other   = min(len(other_eps), batch_size - n_current)
            # Reallocate unused slots to the other group
            n_current = min(len(task_eps),  batch_size - n_other)
            batch = []
            if n_current > 0:
                batch += random.sample(task_eps, n_current)
            if n_other > 0:
                batch += random.sample(other_eps, n_other)
            return batch
        n = min(len(self.buffer), batch_size)
        return random.sample([(t, ep) for t, iid, ep in self.buffer], n)


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
    reward_normalizers: dict | None = None,
) -> float:
    """Run n_episodes with the meta-policy; return mean success rate.

    When meta_subgoals and skeleton_potential are supplied the shaped reward
    is used as the τ signal, matching the training-time GRU input exactly.
    reward_normalizers, if provided, applies the same per-task EMA
    normalization used during training so the GRU hidden state evolves
    under the same reward distribution it was trained on.  Stats are NOT
    updated during eval — only normalize() is called, not update().
    Success is measured on raw env reward only.
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

                # Apply same normalization as training (read-only — no update)
                if reward_normalizers is not None and task.id in reward_normalizers:
                    _r_t  = torch.tensor([r_tau], dtype=torch.float32)
                    r_tau = float(reward_normalizers[task.id].normalize(_r_t)[0])

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

    Runs exactly n_episodes per task so each task gets a fair, independent
    evaluation.
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
    reward_normalizers: dict | None = None,
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
        env_rewards     list[float]              — raw env rewards
        transitions     list[(s, a_np, r_env, s_next, done, terminated)]
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

        if reward_normalizers is not None:
            tid = task.id
            if tid not in reward_normalizers:
                reward_normalizers[tid] = EMANormalizer()
            _r_t = torch.tensor([r_shaped], dtype=torch.float32)
            reward_normalizers[tid].update(_r_t)
            r_shaped = float(reward_normalizers[tid].normalize(_r_t)[0])

        states.append(torch.tensor(s_arr, dtype=torch.float32, device=device))
        actions.append(a.detach().squeeze(0))
        hidden_states.append(h.detach().squeeze(0))
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
        "states_t":         torch.stack(states),
        "actions_t":        torch.stack(actions),
        "hidden_states_t":  torch.stack(hidden_states),
        "shaped_rewards_t": torch.tensor(shaped_rewards, dtype=torch.float32, device=device),
        "log_probs_old_t":  torch.stack(log_probs_old),
        "returns_t":        torch.tensor(returns_list, dtype=torch.float32, device=device),
        "shaped_rewards":   shaped_rewards,
        "env_rewards":      env_rewards,
        "transitions":      transitions,
        "task_id":          task.id,
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
    shaping_scales: dict      = None,
    subgoal_threshold: float  = float("inf"),
    entropy_coef: float       = 0.01,
    lr: float                 = 3e-4,
    max_episode_steps: int    = 500,
    eval_every: int           = 50,
    eval_episodes: int        = 10,
    buffer_capacity: int      = 10_000,
    episodes_per_batch: int   = 40,
    awr_beta: float           = 1.0,
    replay_buffer             = None,
    flush_buffer: bool        = False,
    flush_optimizer: bool     = False,
    device: str               = "cpu",
    verbose: bool             = True,
    training_state: dict      = None,
    graduation_callback       = None,
):
    """
    Train meta-policy using advantage-weighted regression (AWR) with a
    cross-task replay buffer.

    buffer_capacity   — max episodes stored across all tasks (FIFO eviction).
    episodes_per_batch — episodes sampled per gradient step; half come from the
                         task most recently collected, half from other tasks.
                         Set to 40–80 for 20–40 episodes per task per update.

    The MetaPolicyBuffer persists across outer iterations (flush_buffer only
    resets the value-optimizer state).  Advantage weighting naturally
    down-weights stale low-return episodes, so old data does not harm training.
    """
    meta_subgoals      = skeleton_data.get("meta_subgoals")
    skeleton_potential = skeleton_data.get("skeleton_potential")

    n_tasks             = len(task_distribution.tasks)
    episodes_per_update = max(1, min(episodes_per_update, n_tasks))

    state_dim = meta_policy.state_dim
    ts        = training_state or {}

    meta_value_net = (ts.get("meta_value_net") or
                      MetaValueNetwork(state_dim, meta_policy.gru_hidden).to(device))

    policy_optimizer = (ts.get("policy_optimizer") or
                        torch.optim.Adam(meta_policy.parameters(), lr=lr))

    value_optimizer  = (ts.get("value_optimizer") or
                        torch.optim.Adam(meta_value_net.parameters(), lr=lr))

    # Buffer persists across iterations — old episodes are reused under AWR.
    meta_policy_buffer = (ts.get("meta_policy_buffer") or
                          MetaPolicyBuffer(capacity=buffer_capacity))

    # Iteration counter tags every episode so stale-potential data can be purged.
    iteration_id = ts.get("iteration_id", 0)

    # Per-task return normalizers — one RunningMeanStd per task_id.
    # Keeping them separate prevents a task with large raw rewards (e.g.
    # HumanoidStandup at ~50 000/ep) from dominating the shared statistics and
    # zeroing the advantage signal for tasks with small rewards (e.g. Humanoid
    # at ~100/ep).  Each task's returns are independently centred and scaled to
    # unit std before advantages are computed, so every task contributes a
    # meaningful gradient regardless of its absolute reward magnitude.
    # Reset on flush_buffer for the same reason as before: a new potential
    # changes return scale, so stale statistics would anchor the normalizer.
    returns_normalizers: dict = {} if flush_buffer else (
        ts.get("returns_normalizers") or {}
    )
    reward_normalizers: dict = {} if flush_buffer else (
        ts.get("reward_normalizers") or {}
    )

    # Per-task EMA of mean episode env return — used to weight AWR gradients
    # by inverse task difficulty so struggling tasks receive more signal.
    # Persists across outer iterations (EMA is self-decaying so stale values
    # fade naturally; no need to reset on flush_buffer).
    task_ema_env_returns: dict = ts.get("task_ema_env_returns") or {}
    _ema_alpha_task: float = 0.9   # slow-moving so weights don't thrash

    # New iteration: increment the tag, purge episodes from previous iterations
    # (their shaped rewards reflect a stale potential with a different scale),
    # and reset value-optimizer momentum.
    if flush_buffer:
        iteration_id += 1
        n_purged = meta_policy_buffer.purge_before(iteration_id)
        value_optimizer.state.clear()
        if verbose and n_purged:
            print(f"  [PG] Purged {n_purged} stale-potential episodes from buffer.")

    if flush_optimizer:
        policy_optimizer.state.clear()
        value_optimizer.state.clear()

    epoch_losses: list = []
    meta_policy.train()
    meta_value_net.train()

    task_list   = list(task_distribution.tasks)
    task_cursor = 0

    for epoch in range(meta_epochs):

        # ── Collect fresh episodes (behaviour policy, no gradients) ──────────
        fresh_rewards:     list = []
        fresh_env_rewards: list = []   # raw env reward, unaffected by shaping scale

        for _ in range(episodes_per_update):
            task = task_list[task_cursor % n_tasks]
            task_cursor += 1

            task_shaping_scale = (
                shaping_scales.get(task.id, shaping_scale)
                if shaping_scales else shaping_scale
            )
            ep = _collect_episode(
                meta_policy, task,
                meta_subgoals, skeleton_potential,
                gamma, task_shaping_scale, subgoal_threshold,
                max_episode_steps, device,
                reward_normalizers=reward_normalizers,
            )
            if ep is None:
                continue

            meta_policy_buffer.add(task.id, ep, iteration_id)
            fresh_rewards.extend(ep["shaped_rewards"])
            fresh_env_rewards.extend(ep["env_rewards"])

            # Update per-task EMA return for difficulty-weighted gradients
            ep_env_r = float(np.mean(ep["env_rewards"])) if ep["env_rewards"] else 0.0
            tid = task.id
            if tid not in task_ema_env_returns:
                task_ema_env_returns[tid] = ep_env_r
            else:
                task_ema_env_returns[tid] = (
                    _ema_alpha_task * task_ema_env_returns[tid]
                    + (1.0 - _ema_alpha_task) * ep_env_r
                )

            if replay_buffer is not None:
                for s_arr, a_np, r_env, s_next_arr, done, terminated in ep["transitions"]:
                    replay_buffer.push(
                        s_arr, a_np, r_env, s_next_arr, done, ep["task_id"],
                        terminated=terminated,
                    )

        # ── Wait for enough data for a meaningful batch ───────────────────────
        if len(meta_policy_buffer) < max(1, episodes_per_batch // 2):
            continue

        # ── Compute per-task gradient weights from inverse difficulty ─────────
        # Tasks with lower EMA env returns are harder; weight them more so
        # the meta-policy receives equal gradient signal regardless of whether
        # one task dominates by absolute reward magnitude.
        # Weights are normalized to mean=1.0 so the total gradient scale is
        # unchanged — we're redistributing signal, not scaling it.
        if len(task_ema_env_returns) == n_tasks:
            _raw_w = {tid: 1.0 / max(abs(r), 1.0)
                      for tid, r in task_ema_env_returns.items()}
            _w_mean = float(np.mean(list(_raw_w.values())))
            task_gradient_weights = {tid: w / max(_w_mean, 1e-8)
                                     for tid, w in _raw_w.items()}
        else:
            task_gradient_weights = {}

        # ── One gradient step per task (consistent gradient target per step) ──
        # Each task samples its own batch: 50% episodes from that task, 50%
        # from other tasks. This keeps the gradient direction fixed within each
        # step rather than rotating with the round-robin collector.
        epoch_task_losses: list = []
        n_zero_adv = 0  # tasks whose batch had near-zero advantage variance

        for task in task_list:
            batch = meta_policy_buffer.sample_batch(episodes_per_batch, task.id)
            if not batch:
                continue
            n_eps = len(batch)

            # Pass 1 (no grad): collect raw advantages for batch normalisation.
            all_returns_norm: list = []
            all_raw_adv:      list = []

            with torch.no_grad():
                for _tid, ep in batch:
                    if _tid not in returns_normalizers:
                        returns_normalizers[_tid] = RunningMeanStd()
                    returns_normalizers[_tid].update(ep["returns_t"])
                    ret_norm = returns_normalizers[_tid].normalize(ep["returns_t"])
                    vals = meta_value_net(
                        ep["states_t"],
                        ep["hidden_states_t"].to(device),
                    ).squeeze(-1)
                    all_returns_norm.append(ret_norm)
                    all_raw_adv.append(ret_norm - vals)

            all_adv_cat = torch.cat(all_raw_adv)
            adv_mean    = all_adv_cat.mean()
            adv_std     = all_adv_cat.std()
            if adv_std <= 1e-6:
                n_zero_adv += 1

            # Pass 2 (with grad): re-roll GRU and apply batch-normalised AWR update.
            policy_optimizer.zero_grad()
            value_optimizer.zero_grad()

            ep_policy_losses: list = []
            ep_value_losses:  list = []
            ep_ent_losses:    list = []

            for i, (_tid, ep) in enumerate(batch):
                curr_lp, curr_ent, ctx = meta_policy.evaluate_actions_gru(
                    ep["states_t"], ep["actions_t"], ep["shaped_rewards_t"]
                )

                returns_norm = all_returns_norm[i]

                raw_adv = all_raw_adv[i]
                if adv_std > 1e-6:
                    advantages = (raw_adv - adv_mean) / (adv_std + 1e-8)
                else:
                    advantages = raw_adv

                awr_weight = (advantages / awr_beta).clamp(-5.0, 2.0).exp().detach()

                values = meta_value_net(
                    ep["states_t"], ctx.detach()
                ).squeeze(-1)

                policy_loss   = -(awr_weight * curr_lp).mean()
                value_loss    = F.mse_loss(values, returns_norm)
                entropy_bonus = curr_ent.mean()

                task_w = task_gradient_weights.get(task.id, 1.0)
                total_ep_loss = (
                    policy_loss + 0.5 * value_loss - entropy_coef * entropy_bonus
                ) * task_w
                total_ep_loss.div(n_eps).backward()

                ep_policy_losses.append(policy_loss.item())
                ep_value_losses.append(value_loss.item())
                ep_ent_losses.append(entropy_bonus.item())

            torch.nn.utils.clip_grad_norm_(meta_policy.parameters(),    0.5)
            torch.nn.utils.clip_grad_norm_(meta_value_net.parameters(), 0.5)
            policy_optimizer.step()
            value_optimizer.step()

            epoch_task_losses.append({
                "policy":  float(np.mean(ep_policy_losses)),
                "value":   float(np.mean(ep_value_losses)),
                "entropy": float(np.mean(ep_ent_losses)),
            })

        if not epoch_task_losses:
            continue

        policy_loss  = float(np.mean([x["policy"]  for x in epoch_task_losses]))
        value_loss   = float(np.mean([x["value"]   for x in epoch_task_losses]))
        entropy_loss = float(np.mean([x["entropy"] for x in epoch_task_losses]))
        total_loss   = policy_loss + 0.5 * value_loss - entropy_coef * entropy_loss
        # Fraction of per-task gradient steps where adv_std ≤ 1e-6 (gradient silenced).
        zero_adv_frac = n_zero_adv / max(1, n_tasks)

        avg_env_r = float(np.mean(fresh_env_rewards)) if fresh_env_rewards else 0.0
        epoch_losses.append({
            "total":         total_loss,
            "policy":        policy_loss,
            "value":         value_loss,
            "entropy":       entropy_loss,
            "zero_adv_frac": zero_adv_frac,
            "avg_env_r":     avg_env_r,
        })

        if (epoch + 1) % eval_every == 0:
            sr = evaluate_meta_policy(
                meta_policy, task_distribution,
                n_episodes=eval_episodes, device=device,
                meta_subgoals=meta_subgoals,
                skeleton_potential=skeleton_potential,
                shaping_scale=shaping_scale,
                subgoal_threshold=subgoal_threshold,
                reward_normalizers=reward_normalizers,
            )
            if verbose:
                avg_shaped_r = float(np.mean(fresh_rewards)) if fresh_rewards else 0.0
                print(f"  [PG] epoch {epoch+1}/{meta_epochs}  "
                      f"buf={len(meta_policy_buffer)}ep  tasks={n_tasks}  "
                      f"loss={total_loss:.4f}  "
                      f"avg_env_r={avg_env_r:.2f}  avg_shaped_r={avg_shaped_r:.4f}  "
                      f"sr={sr:.1%}  zero_adv={zero_adv_frac:.0%}")

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
                task_list   = list(task_distribution.tasks)
                n_tasks     = len(task_list)
                task_cursor = 0
                if n_tasks == 0:
                    break

    return meta_policy, meta_value_net, epoch_losses, {
        "meta_value_net":        meta_value_net,
        "policy_optimizer":      policy_optimizer,
        "value_optimizer":       value_optimizer,
        "meta_policy_buffer":    meta_policy_buffer,
        "returns_normalizers":   returns_normalizers,
        "reward_normalizers":    reward_normalizers,
        "task_ema_env_returns":  task_ema_env_returns,
        "iteration_id":          iteration_id,
    }
