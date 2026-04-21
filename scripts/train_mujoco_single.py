"""
Single-task topology-guided reward shaping for MuJoCo environments.

Alternates reward shaping with task-policy training:

  Phase 0  — Bootstrap SAC on the raw environment to fill the replay buffer.
  Per iteration:
    Phase 1  — Build Morse complex from the replay buffer (FPS landmarks →
               witness complex → persistent critical simplices → DBSCAN subgoals).
    Phase 2  — Build combined potential:
               Φ = α · Φ_skeleton  +  (1−α) · Φ_empirical_hitting_time
    Phase 3  — Continue SAC training with Φ-shaped rewards.
    Collect  — Roll out the updated policy to grow the replay buffer.

There is no meta-policy here: the SAC task policy IS the final agent.
Topology only guides reward shaping to provide a denser learning signal.

Usage:
    python scripts/train_mujoco_single.py --env Hopper-v4
    python scripts/train_mujoco_single.py --env HalfCheetah-v4 \\
        --iterations 6 --landmarks 48 --phase3-steps 50000
    python scripts/train_mujoco_single.py --env Ant-v4 \\
        --save-dir results/ant_run1 --device cuda
"""

import argparse
import json
import os
import sys
import tempfile

import gymnasium as gym
import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from stable_baselines3 import SAC
from stable_baselines3.common.env_util import make_vec_env

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# MPS (Apple Silicon) does not support float64.  Force float32 globally so that
# any tensor created from a Python float or default-dtype numpy array (float64)
# is automatically float32.  Must be set before any model or tensor is created.
torch.set_default_dtype(torch.float32)

from algos.potential import (
    CombinedPotential,
    EmpiricalHittingTimePotential,
    ShapedRewardWrapper,
    SkeletonPotential,
)
from utils.checkpoint import load_replay_buffer, save_replay_buffer
from utils.replay_buffer import ReplayBuffer
from utils.skeleton import build_skeleton
from utils.viz import Phase3TrainingCallback, _P, _c, _save_fig, plot_skeleton_topology


# ── Environment helpers ────────────────────────────────────────────────────────

def _env_dims(env_id: str) -> tuple[int, int]:
    env = gym.make(env_id)
    s = int(np.prod(env.observation_space.shape))
    a = int(np.prod(env.action_space.shape))
    env.close()
    return s, a


def _make_raw(env_id: str):
    return gym.make(env_id)


def _make_shaped(env_id: str, potential):
    """Return a ShapedRewardWrapper around env_id."""
    return ShapedRewardWrapper(gym.make(env_id), potential)


# ── Phase 0: bootstrap ─────────────────────────────────────────────────────────

def phase0_bootstrap(
    env_id: str,
    replay_buffer: ReplayBuffer,
    total_steps: int = 20_000,
    n_envs: int = 10,
    device: str = "cpu",
    verbose: bool = True,
) -> SAC:
    """Train SAC on the raw env and push all collected transitions to replay_buffer."""
    if verbose:
        print(f"  [Phase 0] SAC bootstrap  env={env_id}  steps={total_steps}  n_envs={n_envs}")

    vec_env = make_vec_env(env_id, n_envs=n_envs)
    model = SAC("MlpPolicy", vec_env, verbose=0, device=device)
    model.learn(total_timesteps=total_steps)

    # Collect a rollout under the trained policy to fill our replay buffer.
    obs = vec_env.reset()
    steps_per_env = max(1, total_steps // n_envs)
    for _ in range(steps_per_env):
        action, _ = model.predict(obs, deterministic=False)
        obs_next, reward, done, info = vec_env.step(action)
        for i in range(n_envs):
            trunc_i = info[i].get("TimeLimit.truncated", False)
            term_i  = bool(done[i]) and not trunc_i
            replay_buffer.push(
                obs[i], action[i], float(reward[i]),
                obs_next[i], bool(done[i]), task_id=0,
                terminated=term_i,
            )
        obs = obs_next

    vec_env.close()
    if verbose:
        print(f"  [Phase 0] Buffer: {len(replay_buffer)} transitions")
    return model


# ── Phase 1: skeleton ──────────────────────────────────────────────────────────

def phase1_build_skeleton(
    replay_buffer: ReplayBuffer,
    state_dim: int,
    action_dim: int,
    num_landmarks: int = 32,
    device: str = "cpu",
    verbose: bool = True,
    **kwargs,
) -> dict:
    """Build Morse skeleton from replay buffer."""
    if verbose:
        print(f"  [Phase 1] Building Morse skeleton  landmarks={num_landmarks}")
    skeleton = build_skeleton(
        replay_buffer,
        state_dim=state_dim,
        action_dim=action_dim,
        num_landmarks=num_landmarks,
        # Single task: accept all critical states found in this task.
        min_task_support=0.0,
        device=device,
        verbose=verbose,
        **kwargs,
    )
    n_sub = len(skeleton.get("meta_subgoals") or skeleton.get("critical_states", {}))
    if verbose:
        print(f"  [Phase 1] Found {n_sub} meta-subgoal(s).")
    skeleton["_potential_stale"] = True
    return skeleton


# ── Phase 2: potential ─────────────────────────────────────────────────────────

def phase2_build_potential(
    skeleton: dict,
    replay_buffer: ReplayBuffer,
    alpha: float = 0.5,
    k: int = 10,
    gamma: float = 0.99,
    hit_threshold: float = 0.5,
    verbose: bool = True,
):
    """Build CombinedPotential from topology + empirical hitting times."""
    raw = skeleton.get("meta_subgoals") or skeleton.get("critical_states", {})
    if raw and not isinstance(next(iter(raw.values())), dict):
        raw = {k_: {"state": v} for k_, v in raw.items()}
    meta_subgoals = {
        k_: {"state": np.asarray(v["state"], dtype=np.float32)} for k_, v in raw.items()
    }

    if not meta_subgoals:
        return None

    lm = skeleton["landmarks"]
    lm_np = lm.cpu().numpy() if hasattr(lm, "cpu") else np.asarray(lm, dtype=np.float32)

    if skeleton.get("_potential_stale", True) or "_skel_potential_cached" not in skeleton:
        skel_pot = SkeletonPotential(lm_np, skeleton["simplices"], meta_subgoals)
        skeleton["_skel_potential_cached"] = skel_pot
        skeleton["_potential_stale"] = False
        if verbose:
            print(
                f"  [Phase 2] SkeletonPotential  "
                f"{skel_pot.G.number_of_nodes()} nodes  "
                f"{skel_pot.G.number_of_edges()} edges"
            )
    else:
        skel_pot = skeleton["_skel_potential_cached"]
        if verbose:
            print("  [Phase 2] Reusing cached SkeletonPotential.")

    emp_pot = EmpiricalHittingTimePotential(
        replay_buffer, meta_subgoals, k=k, gamma=gamma, hit_threshold=hit_threshold,
    )
    n_covered = sum(len(v) > 0 for v in emp_pot._trajs.values())
    if verbose:
        print(
            f"  [Phase 2] EmpiricalPotential  "
            f"{n_covered}/{len(meta_subgoals)} subgoals covered"
        )

    combined = CombinedPotential(skel_pot, emp_pot, lm_np, alpha=alpha)
    if verbose:
        print(
            f"  [Phase 2] CombinedPotential  α={alpha:.2f}  "
            f"skel_scale={combined._skel_scale:.4f}  "
            f"emp_scale={combined._emp_scale:.4f}"
        )
    return combined


# ── Phase 3: shaped SAC training ───────────────────────────────────────────────

def phase3_train(
    env_id: str,
    potential,
    prior_model: SAC,
    replay_buffer: ReplayBuffer,
    total_steps: int = 20_000,
    shaping_scale: float = 1.0,
    device: str = "cpu",
    verbose: bool = True,
) -> tuple[SAC, dict]:
    """
    Continue SAC from prior_model using shaped rewards.

    Returns (updated_model, stats_dict).
    """
    if verbose:
        print(f"  [Phase 3] SAC with shaping  steps={total_steps}")

    def _make(pot=potential, scale=shaping_scale):
        base = gym.make(env_id)
        return ShapedRewardWrapper(base, pot, shaping_scale=scale) if pot is not None else base

    vec_env = make_vec_env(_make, n_envs=1)
    cb = Phase3TrainingCallback()

    # Continue from prior weights; SB3 load() handles rollout buffer rebuild.
    with tempfile.TemporaryDirectory() as td:
        prior_model.save(os.path.join(td, "prior"))
        model = SAC.load(os.path.join(td, "prior"), env=vec_env, device=device)

    model.learn(total_timesteps=total_steps, reset_num_timesteps=False, callback=cb)
    vec_env.close()

    # Grow the replay buffer with new shaped-env experience.
    _collect_transitions(model, env_id, replay_buffer, n_steps=max(1000, total_steps // 4))

    if verbose and cb.ep_rewards:
        last = cb.ep_rewards[-20:]
        print(
            f"  [Phase 3] last-20 avg_shaped_r={np.mean(last):.3f}  "
            f"avg_env_r={np.mean(cb.ep_env_rewards[-20:]):.3f}"
        )

    stats = {
        "ep_rewards":     cb.ep_rewards,
        "ep_env_rewards": cb.ep_env_rewards,
        "ep_shaping":     cb.ep_shaping,
        "ep_lengths":     cb.ep_lengths,
    }
    return model, stats


# ── Collect helpers ────────────────────────────────────────────────────────────

def _collect_transitions(
    model: SAC,
    env_id: str,
    replay_buffer: ReplayBuffer,
    n_steps: int = 2_000,
    task_id: int = 0,
) -> None:
    """Roll out model (deterministic=False) and push raw-reward transitions."""
    env = gym.make(env_id)
    obs, _ = env.reset()
    for _ in range(n_steps):
        action, _ = model.predict(obs, deterministic=False)
        obs_next, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        replay_buffer.push(
            obs, action, float(reward), obs_next, done, task_id,
            terminated=terminated,
        )
        obs = obs_next if not done else env.reset()[0]
    env.close()


# ── Visualisation ──────────────────────────────────────────────────────────────

def _plot_training_curve(all_stats: list, save_dir: str) -> None:
    """Save a plot of episode env returns across all iterations."""
    x_offset = 0
    boundaries = []
    raw_r_segs: list = []

    for it_stats in all_stats:
        ep_env_r = it_stats.get("ep_env_rewards", [])
        n = len(ep_env_r)
        if n == 0:
            continue
        xs = list(range(x_offset, x_offset + n))
        raw_r_segs.append((xs, ep_env_r))
        x_offset += n
        boundaries.append(x_offset)

    if not raw_r_segs:
        return

    all_xs = [x for xs, _ in raw_r_segs for x in xs]
    all_rs = [r for _, rs in raw_r_segs for r in rs]

    fig, ax = plt.subplots(figsize=(9, 4))

    # Raw returns, coloured by iteration.
    for i, (xs, rs) in enumerate(raw_r_segs):
        ax.plot(xs, rs, color=_c(i), alpha=0.25, linewidth=0.8)

    # Rolling mean (window = 5% of total episodes, min 5).
    w = max(5, len(all_rs) // 20)
    if len(all_rs) >= w:
        kernel = np.ones(w) / w
        smooth = np.convolve(all_rs, kernel, mode="valid")
        ax.plot(
            all_xs[w - 1:], smooth,
            color=_P["blue"], linewidth=2.0, label=f"rolling mean (w={w})",
        )

    # Per-iteration mean markers.
    iter_means = [float(np.mean(rs)) for _, rs in raw_r_segs]
    iter_mid_x = [int(np.mean(xs)) for xs, _ in raw_r_segs]
    ax.scatter(iter_mid_x, iter_means, color=_P["gold"], s=60, zorder=5,
               label="iter mean")

    # Iteration boundary lines.
    for b in boundaries[:-1]:
        ax.axvline(b, color=_P["red"], linewidth=0.8, linestyle="--", alpha=0.6)

    # Iteration labels along the top (axes-fraction y so they sit just inside).
    prev = 0
    for i, b in enumerate(boundaries):
        mid = (prev + b) / 2
        ax.text(mid, 0.97, f"iter {i + 1}",
                ha="center", va="top", fontsize=7, color=_c(i),
                transform=ax.get_xaxis_transform())
        prev = b

    ax.set_xlabel("episode")
    ax.set_ylabel("env return")
    ax.set_title("Episode returns across iterations")
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()

    out = os.path.join(save_dir, "returns_evolution.png")
    _save_fig(fig, out)
    print(f"  [Viz] {out}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Topology-guided reward shaping for a single MuJoCo task."
    )
    parser.add_argument("--env",            default="Hopper-v4",
                        help="Gymnasium env ID (default: Hopper-v4)")
    parser.add_argument("--iterations",     type=int, default=4,
                        help="Number of shape→train cycles (default: 4)")
    parser.add_argument("--phase0-steps",   type=int, default=20_000,
                        help="Bootstrap SAC timesteps (default: 20000)")
    parser.add_argument("--phase3-steps",   type=int, default=20_000,
                        help="SAC timesteps per shaped iteration (default: 20000)")
    parser.add_argument("--landmarks",      type=int, default=200,
                        help="FPS landmark count (default: 32)")
    parser.add_argument("--alpha",          type=float, default=0.0,
                        help="Potential mix: 1=pure topology, 0=pure empirical (default: 0.5)")
    parser.add_argument("--shaping-scale",  type=float, default=1.0,
                        help="Intrinsic reward coefficient (default: 1.0)")
    parser.add_argument("--gamma",          type=float, default=0.95,
                        help="Discount factor (default: 0.99)")
    parser.add_argument("--refine-every",   type=int, default=1,
                        help="Rebuild skeleton every N iterations (default: 1=every)")
    parser.add_argument("--save-dir",       default="results/mujoco_single",
                        help="Output directory (default: results/mujoco_single)")
    parser.add_argument("--device",         default="cpu")
    parser.add_argument("--no-verbose",     action="store_true")
    args = parser.parse_args()

    verbose = not args.no_verbose
    os.makedirs(args.save_dir, exist_ok=True)

    state_dim, action_dim = _env_dims(args.env)
    if verbose:
        print("=" * 60)
        print(f"Topology-guided SAC  env={args.env}")
        print(f"  state_dim={state_dim}  action_dim={action_dim}")
        print(f"  landmarks={args.landmarks}  alpha={args.alpha}")
        print(f"  iterations={args.iterations}  save_dir={args.save_dir}")
        print("=" * 60)

    # ── Phase 0 ───────────────────────────────────────────────────────────────
    rb_path = os.path.join(args.save_dir, "replay_buffer.npz")
    model_path = os.path.join(args.save_dir, "phase0_model.zip")

    if os.path.exists(rb_path) and os.path.exists(model_path):
        if verbose:
            print("\n[Phase 0] Loading existing bootstrap data...")
        rb = load_replay_buffer(rb_path, device=args.device)
        model = SAC.load(model_path, device=args.device)
        # Rebuilt rollout buffer is not needed (SAC uses its own off-policy buffer).
        if verbose:
            print(f"  Loaded {len(rb)} transitions.")
    else:
        if verbose:
            print("\n[Phase 0] SAC bootstrap...")
        rb = ReplayBuffer(device=args.device)
        model = phase0_bootstrap(
            args.env, rb,
            total_steps=args.phase0_steps,
            device=args.device,
            verbose=verbose,
        )
        save_replay_buffer(rb, rb_path)
        model.save(model_path)

    metrics = {"iterations": [], "eval_returns": []}
    all_phase3_stats: list = []
    skeleton = None

    for iteration in range(args.iterations):
        if verbose:
            print(f"\n{'─' * 60}")
            print(f"Iteration {iteration + 1}/{args.iterations}  "
                  f"buffer={len(rb)} transitions")
            print(f"{'─' * 60}")

        # ── Phase 1: rebuild skeleton ──────────────────────────────────────
        rebuild = (skeleton is None) or (iteration % args.refine_every == 0)
        if rebuild:
            if verbose:
                print("\n[Phase 1] Building Morse skeleton...")
            try:
                skeleton = phase1_build_skeleton(
                    rb, state_dim, action_dim,
                    num_landmarks=args.landmarks,
                    gamma=args.gamma,
                    device=args.device,
                    verbose=verbose,
                )
                topo_path = os.path.join(
                    args.save_dir, f"topology_iter{iteration + 1:03d}.png"
                )
                plot_skeleton_topology(skeleton, rb, topo_path)
            except Exception as exc:
                print(f"  [Phase 1] WARNING: skeleton build failed ({exc}); "
                      f"skipping shaping this iteration.")
                skeleton = None
        else:
            # Mark potential as stale only when topology was just rebuilt.
            if verbose:
                print("\n[Phase 1] Reusing skeleton from previous iteration.")

        # ── Phase 2: potential ─────────────────────────────────────────────
        n_sub = len((skeleton or {}).get("meta_subgoals")
                    or (skeleton or {}).get("critical_states", {}))
        if skeleton is not None and n_sub > 0:
            if verbose:
                print("\n[Phase 2] Building potential...")
            potential = phase2_build_potential(
                skeleton, rb,
                alpha=args.alpha,
                gamma=args.gamma,
                verbose=verbose,
            )
            if rebuild:
                skeleton["_potential_stale"] = False
        else:
            if verbose:
                print("\n[Phase 2] No subgoals found — training without shaping.")
            potential = None

        # ── Phase 3: shaped SAC ────────────────────────────────────────────
        if verbose:
            print("\n[Phase 3] Continuing SAC with shaped rewards...")
        model, p3_stats = phase3_train(
            args.env,
            potential,
            model,
            rb,
            total_steps=args.phase3_steps,
            shaping_scale=args.shaping_scale,
            device=args.device,
            verbose=verbose,
        )
        all_phase3_stats.append(p3_stats)

        # ── Evaluate ──────────────────────────────────────────────────────
        eval_return = _evaluate(model, args.env, n_episodes=10)
        metrics["eval_returns"].append(eval_return)
        metrics["iterations"].append(iteration + 1)
        if verbose:
            print(f"  [Eval] mean_return={eval_return:.2f}")

        # ── Checkpoint ────────────────────────────────────────────────────
        ckpt_model = os.path.join(args.save_dir, f"model_iter{iteration + 1:03d}.zip")
        model.save(ckpt_model)
        ckpt_rb = os.path.join(args.save_dir, "replay_buffer.npz")
        save_replay_buffer(rb, ckpt_rb)

        _plot_training_curve(all_phase3_stats, args.save_dir)

    # ── Final outputs ─────────────────────────────────────────────────────────
    metrics_path = os.path.join(args.save_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    if verbose:
        print(f"\n[Done] metrics → {metrics_path}")
        print(f"       best eval return: {max(metrics['eval_returns']):.2f}")

    model.save(os.path.join(args.save_dir, "model_final.zip"))


def _evaluate(model: SAC, env_id: str, n_episodes: int = 10) -> float:
    """Run n_episodes deterministically; return mean undiscounted return."""
    env = gym.make(env_id)
    returns = []
    for _ in range(n_episodes):
        obs, _ = env.reset()
        ep_return = 0.0
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            ep_return += float(reward)
            done = terminated or truncated
        returns.append(ep_return)
    env.close()
    return float(np.mean(returns))


if __name__ == "__main__":
    main()
