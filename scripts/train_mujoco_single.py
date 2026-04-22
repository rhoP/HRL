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
from algos.progress import (
    MixedPotential,
    ProgressPotential,
    train_progress_estimator,
)
from utils.checkpoint import load_replay_buffer, save_replay_buffer
from utils.replay_buffer import ReplayBuffer
from utils.skeleton import build_skeleton
from utils.viz import BootstrapCallback, Phase3TrainingCallback, _P, _c, _save_fig, plot_skeleton_topology


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
) -> tuple[SAC, dict]:
    """Train SAC on the raw env and push all collected transitions to replay_buffer.

    Returns (model, stats) where stats has the same shape as Phase 3 stats dicts
    so it can be prepended to the unified all_stats list.
    """
    if verbose:
        print(f"  [Phase 0] SAC bootstrap  env={env_id}  steps={total_steps}  n_envs={n_envs}")

    vec_env = make_vec_env(env_id, n_envs=n_envs)
    cb = BootstrapCallback()
    model = SAC("MlpPolicy", vec_env, verbose=0, device=device)
    model.learn(total_timesteps=total_steps, callback=cb)
    vec_env.close()

    # Fill our replay buffer via the single-env Gymnasium API so that the
    # terminated/truncated distinction is captured accurately.  SB3's VecEnv
    # merges the two flags into a single done boolean and may not reliably
    # populate info["TimeLimit.truncated"] across all Gymnasium versions,
    # which would silently mark every survived episode as failed.
    collect_steps = max(5_000, total_steps // n_envs * 2)
    _collect_transitions(model, env_id, replay_buffer, n_steps=collect_steps)
    if verbose:
        print(f"  [Phase 0] Buffer: {len(replay_buffer)} transitions "
              f"({collect_steps} collection steps)")

    stats = {
        "ep_rewards":     cb.ep_rewards,
        "ep_env_rewards": cb.ep_rewards,   # no shaping in Phase 0
        "ep_shaping":     [0.0] * len(cb.ep_rewards),
        "ep_lengths":     cb.ep_lengths,
        "label":          "bootstrap",
    }
    return model, stats


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
    """Build Morse skeleton from replay buffer using only survived trajectories."""
    if verbose:
        n_eps     = len(replay_buffer._ep_ends)
        n_survived = sum(
            1 for ep in replay_buffer.iter_episodes()
            if not any(np.asarray(ep.get("terminated", ep["dones"]), dtype=bool))
        )
        print(f"  [Phase 1] Building Morse skeleton  landmarks={num_landmarks}"
              f"  episodes={n_eps}  survived={n_survived}")
    skeleton = build_skeleton(
        replay_buffer,
        state_dim=state_dim,
        action_dim=action_dim,
        num_landmarks=num_landmarks,
        # Single task: accept all critical states found in this task.
        min_task_support=0.0,
        device=device,
        verbose=verbose,
        survived_only=True,   # MuJoCo: use only trajectories that ran to completion
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
    gamma: float = 0.9999,
    hit_threshold: float = 10.0,
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


# ── Phase 2b: progress estimator (optional) ───────────────────────────────────

def phase2b_train_progress(
    replay_buffer: ReplayBuffer,
    state_dim: int,
    base_potential,
    latent_dim: int        = 64,
    epochs: int            = 10,
    base_weight: float     = 0.5,
    progress_weight: float = 0.5,
    device: str            = "cpu",
    verbose: bool          = True,
):
    """
    Train a ProgressEstimator and blend it with base_potential.

    Returns a MixedPotential when base_potential is available, a standalone
    ProgressPotential otherwise.  Returns base_potential unchanged if training
    produces no usable pairs.
    """
    if verbose:
        print(f"  [Phase 2b] Training ProgressEstimator  "
              f"latent_dim={latent_dim}  epochs={epochs}  "
              f"base_w={base_weight:.2f}  prog_w={progress_weight:.2f}")

    encoder, estimator = train_progress_estimator(
        replay_buffer,
        state_dim=state_dim,
        latent_dim=latent_dim,
        epochs=epochs,
        survived_only=True,
        device=device,
        verbose=verbose,
    )
    if encoder is None:
        if verbose:
            print("  [Phase 2b] Progress training produced no data — "
                  "using base potential only.")
        return base_potential

    prog_pot = ProgressPotential(encoder, estimator, state_dim, device=device)

    if base_potential is not None:
        mixed = MixedPotential(
            base_potential, prog_pot,
            base_weight=base_weight,
            prog_weight=progress_weight,
        )
        if verbose:
            print("  [Phase 2b] MixedPotential ready.")
        return mixed

    if verbose:
        print("  [Phase 2b] ProgressPotential only (no base potential).")
    return prog_pot


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
    # Budget: enough steps for at least 5-10 complete episodes.
    # total_steps // 4 is too conservative for 1000-step MuJoCo episodes.
    _collect_transitions(model, env_id, replay_buffer, n_steps=max(5_000, total_steps // 2))

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
    """
    Save three separate files as a continuous curve across all phases:
      returns_env.png        — env return (raw task reward)
      returns_shaped.png     — shaped return (env + shaping bonus)
      returns_shaping.png    — shaping bonus only

    all_stats entries may include a "label" key (e.g. "bootstrap", "iter 1").
    The bootstrap segment is shaded distinctly; iteration boundaries are marked
    with dashed vertical lines.
    """
    def _build_segs(key):
        x_off, segs, labels, bounds = 0, [], [], []
        for seg_stats in all_stats:
            vals = seg_stats.get(key, [])
            n = len(vals)
            if n == 0:
                continue
            xs = list(range(x_off, x_off + n))
            segs.append((xs, vals))
            labels.append(seg_stats.get("label", ""))
            x_off += n
            bounds.append(x_off)
        return segs, labels, bounds

    env_segs,     env_labels,  env_bounds  = _build_segs("ep_env_rewards")
    shaped_segs,  _,           _           = _build_segs("ep_rewards")
    shaping_segs, _,           _           = _build_segs("ep_shaping")

    if not env_segs:
        return

    panel_cfg = [
        (env_segs,     env_labels, env_bounds, "env return",    "Env returns (task reward)",    "returns_env.png"),
        (shaped_segs,  env_labels, env_bounds, "shaped return", "Shaped returns (env + bonus)", "returns_shaped.png"),
        (shaping_segs, env_labels, env_bounds, "shaping bonus", "Shaping bonus only",           "returns_shaping.png"),
    ]

    for segs, seg_labels, bounds, ylabel, title, fname in panel_cfg:
        fig, ax = plt.subplots(figsize=(10, 4))

        if not segs:
            ax.text(0.5, 0.5, "no data", ha="center", va="center",
                    transform=ax.transAxes, fontsize=10)
            ax.set_title(title)
            fig.tight_layout()
            _save_fig(fig, os.path.join(save_dir, fname))
            continue

        all_xs = [x for xs, _ in segs for x in xs]
        all_vs = [v for _, vs in segs for v in vs]

        # Draw each phase segment; bootstrap gets grey, iterations get colours.
        for i, (xs, vs) in enumerate(segs):
            is_bootstrap = seg_labels[i] == "bootstrap"
            color = "#888888" if is_bootstrap else _c(i)
            ax.plot(xs, vs, color=color, alpha=0.18, linewidth=0.7)

        # Continuous rolling mean over all episodes.
        w = max(5, len(all_vs) // 20)
        if len(all_vs) >= w:
            smooth = np.convolve(all_vs, np.ones(w) / w, mode="valid")
            ax.plot(all_xs[w - 1:], smooth, color=_P["blue"], linewidth=2.0,
                    label=f"rolling mean (w={w})")

        # Per-segment mean markers.
        seg_means = [float(np.mean(vs)) for _, vs in segs]
        seg_mid_x = [int(np.mean(xs)) for xs, _ in segs]
        ax.scatter(seg_mid_x, seg_means, color=_P["gold"], s=50, zorder=5,
                   label="segment mean")

        # Vertical boundary lines between segments.
        for b in bounds[:-1]:
            ax.axvline(b, color=_P["red"], linewidth=0.8, linestyle="--", alpha=0.5)

        # Segment labels along the top.
        iter_counter = 0
        prev = 0
        for i, b in enumerate(bounds):
            lbl = seg_labels[i]
            if not lbl:
                iter_counter += 1
                lbl = f"iter {iter_counter}"
            elif lbl != "bootstrap":
                iter_counter += 1
            is_bootstrap = lbl == "bootstrap"
            color = "#888888" if is_bootstrap else _c(i)
            mid = (prev + b) / 2
            ax.text(mid, 0.97, lbl, ha="center", va="top",
                    fontsize=6, color=color,
                    transform=ax.get_xaxis_transform())
            prev = b

        ax.set_xlabel("episode")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(fontsize=7, loc="upper left")
        fig.tight_layout()
        out = os.path.join(save_dir, fname)
        _save_fig(fig, out)
        print(f"  [Viz] {out}")


def _log_shaping_diagnostics(all_stats: list, iteration: int) -> None:
    """Print per-iteration shaping diagnostics to stderr to help diagnose stagnation."""
    if not all_stats:
        return
    it_stats = all_stats[-1]
    ep_env  = it_stats.get("ep_env_rewards", [])
    ep_sh   = it_stats.get("ep_shaping", [])
    ep_len  = it_stats.get("ep_lengths", [])
    if not ep_env:
        return
    tail = 20
    mean_env = float(np.mean(ep_env[-tail:]))
    mean_sh  = float(np.mean(ep_sh[-tail:])) if ep_sh else 0.0
    mean_len = float(np.mean(ep_len[-tail:])) if ep_len else 0.0
    sh_frac  = abs(mean_sh) / (abs(mean_env) + abs(mean_sh) + 1e-8)
    print(f"  [Diag iter {iteration}]"
          f"  avg_env_r={mean_env:.2f}"
          f"  avg_shaping={mean_sh:.3f}  ({sh_frac:.1%} of |total|)"
          f"  avg_ep_len={mean_len:.1f}"
          f"  shaping_active={'yes' if mean_sh != 0.0 else 'NO — potential is None'}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Topology-guided reward shaping for a single MuJoCo task."
    )
    parser.add_argument("--env",            default="Hopper-v4",
                        help="Gymnasium env ID (default: Hopper-v4)")
    parser.add_argument("--iterations",     type=int, default=5,
                        help="Number of shape→train cycles (default: 4)")
    parser.add_argument("--phase0-steps",   type=int, default=20_000,
                        help="Bootstrap SAC timesteps (default: 20000)")
    parser.add_argument("--phase3-steps",   type=int, default=10_000,
                        help="SAC timesteps per shaped iteration (default: 20000)")
    parser.add_argument("--landmarks",      type=int, default=200,
                        help="FPS landmark count (default: 32)")
    parser.add_argument("--alpha",          type=float, default=0.5,
                        help="Potential mix: 1=pure topology, 0=pure empirical (default: 0.5)")
    parser.add_argument("--shaping-scale",  type=float, default=1.0,
                        help="Intrinsic reward coefficient (default: 1.0)")
    parser.add_argument("--gamma",          type=float, default=0.999,
                        help="Discount factor (default: 0.95)")
    parser.add_argument("--refine-every",   type=int, default=1,
                        help="Rebuild skeleton every N iterations (default: 1=every)")
    parser.add_argument("--save-dir",       default="results/mujoco_single",
                        help="Output directory (default: results/mujoco_single)")
    parser.add_argument("--device",         default="cpu")
    parser.add_argument("--no-verbose",     action="store_true")
    parser.add_argument("--dbscan-eps",     type=float, default=None,
                        help="DBSCAN eps for meta-subgoal clustering. "
                             "Default: auto (median pairwise distance / 2).")
    # Progress-estimation shaping
    parser.add_argument("--use-progress",        action="store_true",
                        help="Add learned progress-estimation reward shaping")
    parser.add_argument("--progress-latent-dim", type=int,   default=64,
                        help="Encoder latent dim for progress estimator (default: 64)")
    parser.add_argument("--progress-epochs",     type=int,   default=10,
                        help="Training epochs for progress estimator (default: 10)")
    parser.add_argument("--progress-weight",     type=float, default=0.5,
                        help="Fraction of shaping from progress vs base (default: 0.5)")
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

    # all_stats holds one dict per training segment (bootstrap + per-iteration
    # phase 3).  Each dict has ep_rewards / ep_env_rewards / ep_shaping /
    # ep_lengths / label.  Passed to _plot_training_curve for the continuous plot.
    all_stats: list = []

    if os.path.exists(rb_path) and os.path.exists(model_path):
        if verbose:
            print("\n[Phase 0] Loading existing bootstrap data...")
        rb = load_replay_buffer(rb_path, device=args.device)
        model = SAC.load(model_path, device=args.device)
        if verbose:
            print(f"  Loaded {len(rb)} transitions.")
        # No bootstrap curve available when loading from checkpoint.
        all_stats.append({
            "ep_rewards": [], "ep_env_rewards": [],
            "ep_shaping": [], "ep_lengths": [],
            "label": "bootstrap (loaded)",
        })
    else:
        if verbose:
            print("\n[Phase 0] SAC bootstrap...")
        rb = ReplayBuffer(device=args.device)
        model, p0_stats = phase0_bootstrap(
            args.env, rb,
            total_steps=args.phase0_steps,
            device=args.device,
            verbose=verbose,
        )
        all_stats.append(p0_stats)
        save_replay_buffer(rb, rb_path)
        model.save(model_path)

    metrics = {"iterations": [], "eval_returns": []}
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
                    dbscan_eps=args.dbscan_eps,
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

        # ── Phase 2b: progress estimator (optional) ───────────────────────
        if args.use_progress:
            if verbose:
                print("\n[Phase 2b] Training progress estimator...")
            potential = phase2b_train_progress(
                rb, state_dim, potential,
                latent_dim=args.progress_latent_dim,
                epochs=args.progress_epochs,
                base_weight=1.0 - args.progress_weight,
                progress_weight=args.progress_weight,
                device=args.device,
                verbose=verbose,
            )

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
        p3_stats["label"] = f"iter {iteration + 1}"
        all_stats.append(p3_stats)
        _log_shaping_diagnostics(all_stats, iteration + 1)

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

        _plot_training_curve(all_stats, args.save_dir)

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
