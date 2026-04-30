"""
train_mujoco_intersection.py

Intersection Meta-RL on Walker2d-v5 variants (Gymnasium).

Phases per outer iteration:
  Phase 0   — Bootstrap SAC per task (once).
  Phase 1   — Shared landmarks + per-task Morse analysis.
              Each task gets its own critical-state set from the shared topology.
  Intersect — Extract the common skeleton: subgoals present in ALL tasks
              (controllable via --intersection-support; 1.0 = strict ALL tasks).
  Phase 2a  — Build one CombinedPotential per task from that task's own subgoals.
  Phase 2b  — Build a CombinedPotential over the intersection skeleton for Phase 4.
  Phase 3   — Train one SB3 SAC per task, shaped by that task's own potential.
  Phase 4   — Train the shared GRU meta-policy via AWR on the intersection skeleton.
  Eval      — Measure mean undiscounted return per task.
  Collect   — Roll meta-policy to grow the shared replay buffer.
  Refine    — Periodically rebuild per-task skeletons and re-extract intersection.

Key difference from train_mujoco_meta.py:
  MetaPolicy builds one cross-task skeleton via DBSCAN soft intersection
  (controlled by min_task_support).

  This script builds a per-task skeleton from shared landmarks, then hard-intersects
  the critical states (intersection_support; default 1.0 = ALL tasks).  Each task's
  SAC is shaped by its own task-specific potential; the meta-policy sees only the
  intersection subgoals — states that are topologically important across every variant.

Usage:
    python scripts/train_mujoco_intersection.py
    python scripts/train_mujoco_intersection.py \\
        --iterations 5 --landmarks 200 --intersection-support 0.66 \\
        --save-dir results/walker_intersection
"""

import argparse
import os
import sys
import tempfile

import gymnasium as gym
import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

_SCRIPTS = os.path.dirname(os.path.abspath(__file__))
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

torch.set_default_dtype(torch.float32)

from stable_baselines3 import SAC
from stable_baselines3.common.env_util import make_vec_env

from algos.potential import (
    AlphaScheduler,
    CombinedPotential,
    EmpiricalHittingTimePotential,
    ShapedRewardWrapper,
    SkeletonPotential,
)
from algos.progress import blend_with_progress
from algos.meta_policy_gradient import meta_policy_gradient_with_skeleton_shaping
from models.meta_policy_net import MetaPolicy
from utils.replay_buffer import ReplayBuffer
from utils.skeleton import build_skeleton
from utils.checkpoint import (
    BestModelTracker,
    load_replay_buffer,
    save_checkpoint,
    save_replay_buffer,
)
from utils.viz import (
    Phase3TrainingCallback,
    plot_phase3_results,
    plot_skeleton_topology,
    plot_training_curves,
    save_iteration_visuals,
)

# Morse intersection utilities live in scripts/morse.py
from morse import meta_critical_states as _morse_meta_critical_states
from morse import compute_meta_centrality as _morse_compute_meta_centrality

# Task classes and shared Gymnasium helpers from train_mujoco_meta
from train_mujoco_meta import (
    GymTask,
    GymTaskDistribution,
    ParameterizedGymTask,
    _check_dims_consistent,
    _collect_transitions,
    collect_with_meta_policy,
    compute_per_task_shaping_scales,
    evaluate_meta_policy,
    make_walker2d_task_dist,
    phase0_collect_initial_data,
)


# ── Task-filtered replay buffer ────────────────────────────────────────────────

class TaskFilteredReplayBuffer:
    """Read-only view of a ReplayBuffer that exposes only one task's episodes.

    Used so that each task's EmpiricalHittingTimePotential is fitted on only
    that task's hitting trajectories, preventing cross-task contamination of
    the empirical shaping signal.
    """

    def __init__(self, base_buffer: ReplayBuffer, task_id: int):
        self._base    = base_buffer
        self._task_id = task_id

    @property
    def state_dim(self):
        return self._base.state_dim

    @property
    def action_dim(self):
        return self._base.action_dim

    def __len__(self) -> int:
        return sum(
            len(ep["rewards"])
            for ep in self._base.iter_episodes()
            if int(ep.get("task_id", 0)) == self._task_id
        )

    def iter_episodes(self):
        for ep in self._base.iter_episodes():
            if int(ep.get("task_id", 0)) == self._task_id:
                yield ep

    def get_all_states(self) -> torch.Tensor:
        arrays = [
            np.asarray(ep["states"], dtype=np.float32)
            for ep in self.iter_episodes()
        ]
        if not arrays:
            D = self._base.state_dim or 1
            return torch.zeros((0, D), dtype=torch.float32)
        return torch.tensor(np.concatenate(arrays, axis=0), dtype=torch.float32)


# ── Phase 1: per-task skeletons ────────────────────────────────────────────────

def phase1_build_per_task_skeletons(
    replay_buffer: ReplayBuffer,
    state_dim: int,
    action_dim: int,
    num_landmarks: int = 200,
    device: str = "cpu",
    verbose: bool = True,
    **kwargs,
) -> tuple[dict, dict]:
    """Build shared landmarks + per-task Morse analysis.

    Calls build_skeleton once on the full buffer with min_task_support=0.0 and
    survived_only=True so that all task-specific critical states are retained
    before the intersection step.  The shared topology (landmarks, simplices) is
    identical for every task; only critical_states / meta_subgoals differ.

    Returns:
        base_skeleton      — full skeleton dict from build_skeleton()
        per_task_skeletons — {task_id: skeleton_dict} with task-specific subgoals
    """
    if replay_buffer.state_dim is None:
        raise RuntimeError("replay_buffer is empty — run Phase 0 first.")

    kwargs.setdefault("survived_only", True)
    kwargs.setdefault("min_task_support", 0.0)   # keep ALL per-task critical states

    if verbose:
        ep_list = list(replay_buffer.iter_episodes())
        n_surv  = sum(1 for ep in ep_list
                      if not any(np.asarray(ep["terminated"], dtype=bool)))
        lengths = [len(ep["rewards"]) for ep in ep_list]
        print(
            f"  [Phase 1] Building per-task skeletons  landmarks={num_landmarks}"
            f"  episodes={len(ep_list)}  survived={n_surv}"
            + (f"  ep_len min={min(lengths)} median={int(np.median(lengths))}"
               f" max={max(lengths)}" if lengths else "")
        )

    base_skeleton = build_skeleton(
        replay_buffer,
        state_dim=state_dim,
        action_dim=action_dim,
        num_landmarks=num_landmarks,
        device=device,
        verbose=verbose,
        **kwargs,
    )

    lm_np = (
        base_skeleton["landmarks"].cpu().numpy()
        if hasattr(base_skeleton["landmarks"], "cpu")
        else np.asarray(base_skeleton["landmarks"], dtype=np.float32)
    )

    task_critical_states = base_skeleton.get("task_critical_states", {})
    per_task_skeletons: dict = {}

    for task_id, task_crit in task_critical_states.items():
        task_subgoals = {
            f"task{task_id}_sg{v_id}": {"state": np.asarray(state, dtype=np.float32)}
            for v_id, state in task_crit.items()
        }
        per_task_skeletons[task_id] = {
            "landmarks":          base_skeleton["landmarks"],
            "landmark_meta":      base_skeleton.get("landmark_meta", {}),
            "simplices":          base_skeleton["simplices"],
            "simplex_task_ids":   base_skeleton.get("simplex_task_ids", {}),
            "edge_action_labels": base_skeleton.get("edge_action_labels", {}),
            "critical_states":    task_subgoals,
            "meta_subgoals":      task_subgoals,
            "phi_critical":       {},
            "knn_estimator":      base_skeleton.get("knn_estimator"),
            "sa_encoder":         base_skeleton.get("sa_encoder"),
            "_potential_stale":   True,
        }

    if verbose:
        for tid, skel in per_task_skeletons.items():
            print(f"  [Phase 1] Task {tid}: {len(skel['critical_states'])} per-task subgoal(s).")
        if not per_task_skeletons:
            print(
                "  [Phase 1] WARNING: task_critical_states is empty in the skeleton. "
                "build_skeleton may need min_task_support=0.0 to populate it."
            )

    return base_skeleton, per_task_skeletons


# ── Intersection extraction ────────────────────────────────────────────────────

def extract_intersection_skeleton(
    base_skeleton: dict,
    intersection_support: float = 1.0,
    dbscan_eps: float = None,
    centrality_threshold: float = 0.0,
    verbose: bool = True,
) -> dict:
    """Extract critical states shared across tasks.

    intersection_support=1.0  → hard intersection (ALL tasks must share the subgoal).
    intersection_support=0.5  → soft intersection (at least 50 % of tasks suffice).

    Returns a skeleton_data dict compatible with SkeletonPotential and
    meta_policy_gradient_with_skeleton_shaping.
    """
    lm_np = (
        base_skeleton["landmarks"].cpu().numpy()
        if hasattr(base_skeleton["landmarks"], "cpu")
        else np.asarray(base_skeleton["landmarks"], dtype=np.float32)
    )
    task_critical = base_skeleton.get("task_critical_states", {})

    if not task_critical:
        if verbose:
            print("  [Intersection] No per-task critical states — empty intersection.")
        common = dict(base_skeleton)
        common.update({"critical_states": {}, "meta_subgoals": {},
                        "phi_critical": {}, "_potential_stale": True})
        return common

    n_total = sum(len(v) for v in task_critical.values())
    if verbose:
        print(
            f"  [Intersection] {len(task_critical)} task(s), "
            f"{n_total} total critical states, "
            f"support threshold={intersection_support:.0%}..."
        )

    common_subgoals = _morse_meta_critical_states(
        task_critical,
        lm_np,
        min_task_support=intersection_support,
        eps=dbscan_eps,
    )

    if verbose:
        print(f"  [Intersection] {len(common_subgoals)} common subgoal(s) before centrality filter.")

    if common_subgoals and centrality_threshold > 0.0:
        common_subgoals = _morse_compute_meta_centrality(
            base_skeleton["simplices"],
            common_subgoals,
            lm_np,
            centrality_threshold=centrality_threshold,
        )
        if verbose:
            print(f"  [Intersection] {len(common_subgoals)} after centrality filter.")

    common = dict(base_skeleton)
    common["critical_states"]  = {k: v["state"] for k, v in common_subgoals.items()}
    common["meta_subgoals"]    = common_subgoals
    common["phi_critical"]     = {}
    common["_potential_stale"] = True
    return common


# ── Phase 2a: per-task potentials ──────────────────────────────────────────────

def phase2_build_per_task_potentials(
    per_task_skeletons: dict,
    replay_buffer: ReplayBuffer,
    alpha: float = 0.5,
    k: int = 10,
    gamma: float = 0.99,
    hit_threshold: float = 0.5,
    verbose: bool = True,
    use_progress: bool = False,
    progress_latent_dim: int = 128,
    progress_epochs: int = 20,
    progress_weight: float = 0.5,
    device: str = "cpu",
) -> dict:
    """Build one CombinedPotential per task from that task's own subgoals.

    Each task's empirical component is fitted on a TaskFilteredReplayBuffer
    so only that task's hitting trajectories inform its potential.

    Returns {task_id: CombinedPotential | None}.
    """
    task_potentials: dict = {}

    for task_id, task_skel in per_task_skeletons.items():
        task_subgoals = task_skel.get("meta_subgoals", {})
        if not task_subgoals:
            if verbose:
                print(f"  [Phase 2a] Task {task_id}: no subgoals — skipping potential.")
            task_potentials[task_id] = None
            continue

        lm_np = (
            task_skel["landmarks"].cpu().numpy()
            if hasattr(task_skel["landmarks"], "cpu")
            else np.asarray(task_skel["landmarks"], dtype=np.float32)
        )

        if task_skel.get("_potential_stale", True) or "_skel_potential_cached" not in task_skel:
            skel_pot = SkeletonPotential(lm_np, task_skel["simplices"], task_subgoals)
            task_skel["_skel_potential_cached"] = skel_pot
            task_skel["_potential_stale"] = False
            if verbose:
                print(
                    f"  [Phase 2a] Task {task_id}: SkeletonPotential "
                    f"({len(task_subgoals)} subgoal(s), "
                    f"{skel_pot.G.number_of_nodes()} nodes, "
                    f"{skel_pot.G.number_of_edges()} edges)."
                )
        else:
            skel_pot = task_skel["_skel_potential_cached"]

        filtered_rb = TaskFilteredReplayBuffer(replay_buffer, task_id)
        emp_pot     = EmpiricalHittingTimePotential(
            filtered_rb, task_subgoals, k=k, gamma=gamma, hit_threshold=hit_threshold,
        )
        n_covered = sum(len(v) > 0 for v in emp_pot._trajs.values())
        if verbose:
            print(
                f"  [Phase 2a] Task {task_id}: EmpiricalPotential "
                f"{n_covered}/{len(task_subgoals)} subgoal(s) covered."
            )

        combined = CombinedPotential(skel_pot, emp_pot, lm_np, alpha=alpha)

        if use_progress:
            if verbose:
                print(f"  [Phase 2a] Task {task_id}: blending with ProgressEstimator...")
            combined = blend_with_progress(
                combined, filtered_rb,
                state_dim=lm_np.shape[-1],
                latent_dim=progress_latent_dim,
                epochs=progress_epochs,
                base_weight=1.0 - progress_weight,
                progress_weight=progress_weight,
                device=device,
                verbose=verbose,
            )

        task_potentials[task_id] = combined

    return task_potentials


# ── Phase 2b: intersection potential ──────────────────────────────────────────

def phase2_build_intersection_potential(
    intersection_skeleton: dict,
    replay_buffer: ReplayBuffer,
    alpha: float = 0.5,
    k: int = 10,
    gamma: float = 0.99,
    hit_threshold: float = 0.5,
    verbose: bool = True,
) -> CombinedPotential | None:
    """CombinedPotential over the intersection skeleton for Phase 4 shaping.

    Uses the full replay buffer (all tasks) for the empirical component since
    intersection subgoals are reachable from multiple tasks by construction.
    Returns None when the intersection skeleton is empty.
    """
    raw = intersection_skeleton.get("meta_subgoals", {})
    if not raw:
        raw = intersection_skeleton.get("critical_states", {})
        raw = {k: {"state": v} for k, v in raw.items()}
    meta_subgoals = {
        k: {"state": np.asarray(v["state"] if isinstance(v, dict) else v, dtype=np.float32)}
        for k, v in raw.items()
    }

    if not meta_subgoals:
        if verbose:
            print("  [Phase 2b] Intersection skeleton empty — no meta-policy shaping.")
        return None

    lm = intersection_skeleton["landmarks"]
    lm_np = lm.cpu().numpy() if hasattr(lm, "cpu") else np.asarray(lm, dtype=np.float32)

    if (
        intersection_skeleton.get("_potential_stale", True)
        or "_skel_potential_cached" not in intersection_skeleton
    ):
        skel_pot = SkeletonPotential(
            lm_np, intersection_skeleton["simplices"], meta_subgoals
        )
        intersection_skeleton["_skel_potential_cached"] = skel_pot
        intersection_skeleton["_potential_stale"] = False
        if verbose:
            print(
                f"  [Phase 2b] Intersection SkeletonPotential "
                f"({len(meta_subgoals)} subgoal(s), "
                f"{skel_pot.G.number_of_nodes()} nodes, "
                f"{skel_pot.G.number_of_edges()} edges)."
            )
    else:
        skel_pot = intersection_skeleton["_skel_potential_cached"]

    emp_pot   = EmpiricalHittingTimePotential(
        replay_buffer, meta_subgoals, k=k, gamma=gamma, hit_threshold=hit_threshold,
    )
    n_covered = sum(len(v) > 0 for v in emp_pot._trajs.values())
    if verbose:
        print(
            f"  [Phase 2b] Intersection EmpiricalPotential "
            f"{n_covered}/{len(meta_subgoals)} subgoal(s) covered."
        )

    combined = CombinedPotential(skel_pot, emp_pot, lm_np, alpha=alpha)
    if verbose:
        print(
            f"  [Phase 2b] α={alpha:.2f}  "
            f"skel_scale={combined._skel_scale:.4f}  "
            f"emp_scale={combined._emp_scale:.4f}"
        )
    return combined


# ── Phase 3: task policies shaped by per-task potentials ──────────────────────

def phase3_train_task_policies(
    task_dist: GymTaskDistribution,
    per_task_potentials: dict,
    replay_buffer: ReplayBuffer,
    timesteps_per_task: int = 10_000,
    shaping_scales: dict = None,
    device: str = "cpu",
    verbose: bool = True,
    save_dir: str = None,
    iteration: int = 0,
    existing_task_policies: dict = None,
) -> tuple:
    """Train one SAC per task, shaped by that task's own CombinedPotential.

    per_task_potentials: {task_id: CombinedPotential | None}
    shaping_scales:      {task_id: float} from compute_per_task_shaping_scales()

    Returns (task_policies, phase3_stats).
    """
    shaping_scales = shaping_scales or {}
    task_policies: dict = {}
    phase3_stats:  dict = {}

    for task in task_dist.tasks:
        task_pot = per_task_potentials.get(task.id)
        scale    = shaping_scales.get(task.id, 1.0)
        n_sg     = len(task_pot.subgoals) if task_pot is not None else 0

        if verbose:
            print(
                f"  [Phase 3] {task.env_name} (task {task.id})  "
                f"steps={timesteps_per_task}  subgoals={n_sg}  "
                f"shaping_scale={scale:.4g}"
            )

        def _make_env(t=task, pot=task_pot, sc=scale):
            base = t.create_env()
            return ShapedRewardWrapper(base, pot, shaping_scale=sc) if pot is not None else base

        cb      = Phase3TrainingCallback()
        vec_env = make_vec_env(_make_env, n_envs=1)

        prior = (existing_task_policies or {}).get(task.id)
        if prior is not None:
            with tempfile.TemporaryDirectory() as td:
                prior.save(os.path.join(td, "prior"))
                model = SAC.load(os.path.join(td, "prior"), env=vec_env, device=device)
            model.learn(
                total_timesteps=timesteps_per_task,
                reset_num_timesteps=False,
                callback=cb,
            )
        else:
            model = SAC("MlpPolicy", vec_env, verbose=0, device=device)
            model.learn(total_timesteps=timesteps_per_task, callback=cb)

        task_policies[task.id] = model
        vec_env.close()

        _collect_transitions(model, task, replay_buffer,
                             n_steps=max(5_000, timesteps_per_task // 2))

        phase3_stats[task.env_name] = {
            "ep_rewards":     cb.ep_rewards,
            "ep_env_rewards": cb.ep_env_rewards,
            "ep_shaping":     cb.ep_shaping,
            "ep_successes":   getattr(cb, "ep_successes", []),
            "ep_lengths":     cb.ep_lengths,
        }
        if verbose and cb.ep_rewards:
            last20 = cb.ep_rewards[-20:]
            print(
                f"    last-20 avg_shaped_r={np.mean(last20):.3f}  "
                f"avg_env_r={np.mean(cb.ep_env_rewards[-20:]):.3f}"
            )

    if save_dir is not None:
        plot_phase3_results(phase3_stats, save_dir, iteration=iteration)

    return task_policies, phase3_stats


# ── Diagnostics ────────────────────────────────────────────────────────────────

def _subgoal_diagnostic(
    per_task_skeletons: dict,
    intersection_skeleton: dict,
    iteration: int,
) -> int:
    """Log per-task and intersection subgoal counts; return intersection count."""
    for tid, skel in sorted(per_task_skeletons.items()):
        n = len(skel.get("meta_subgoals") or skel.get("critical_states", {}))
        print(f"  [Diag iter {iteration}] Task {tid}: {n} per-task subgoal(s).")

    n_common = len(
        intersection_skeleton.get("meta_subgoals")
        or intersection_skeleton.get("critical_states", {})
    )
    print(f"  [Diag iter {iteration}] Intersection: {n_common} common subgoal(s).")
    if n_common == 0:
        print(
            f"  [Diag iter {iteration}] WARNING: empty intersection — "
            "Phase 4 will run without topology shaping. "
            "Consider lowering --intersection-support."
        )
    return n_common


# ── Main loop ──────────────────────────────────────────────────────────────────

def main_intersection_rl_loop(
    task_dist: GymTaskDistribution,
    state_dim: int,
    action_dim: int,
    action_low: float = -1.0,
    action_high: float = 1.0,
    num_landmarks: int = 200,
    num_iterations: int = 5,
    refine_every: int = 2,
    phase0_steps: int = 30_000,
    n_envs: int = 4,
    task_policy_steps: int = 10_000,
    collect_episodes: int = 20,
    gamma: float = 0.99,
    meta_epochs: int = 300,
    shaping_scale: float = 1.0,
    subgoal_threshold: float = float("inf"),
    potential_alpha: float = 0.5,
    alpha_anneal_iters: int = 3,
    intersection_support: float = 1.0,
    centrality_threshold: float = 0.0,
    dbscan_eps: float = None,
    max_pool_size: int = None,
    eval_episodes_per_task: int = 5,
    use_progress: bool = False,
    progress_latent_dim: int = 128,
    progress_epochs: int = 20,
    progress_weight: float = 0.5,
    save_dir: str = "results/mujoco_intersection",
    device: str = "cpu",
    verbose: bool = True,
):
    """Intersection Meta-RL pipeline for Gymnasium MuJoCo tasks."""
    os.makedirs(save_dir, exist_ok=True)

    _alpha_sched = AlphaScheduler(
        alpha_start=potential_alpha, alpha_end=0.0, anneal_iters=alpha_anneal_iters
    )

    if verbose:
        env_list = [t.env_name for t in task_dist.tasks]
        print("=" * 60)
        print("Intersection Meta-RL  (MuJoCo Gymnasium)")
        print(f"  tasks ({len(env_list)}): {env_list}")
        print(f"  state_dim={state_dim}  action_dim={action_dim}")
        print(f"  landmarks={num_landmarks}  iterations={num_iterations}")
        print(f"  potential_alpha={potential_alpha}  shaping_scale={shaping_scale}")
        print(f"  intersection_support={intersection_support:.0%}  "
              f"(1.0 = all tasks must share a subgoal)")
        print(f"  alpha_schedule: {_alpha_sched}")
        print(f"  save_dir={save_dir}")
        print("=" * 60)

    metrics = {
        "skeleton_train_losses": [],
        "phase3_env_returns":    [],
        "phase4_losses":         [],
        "eval_returns":          [],
    }
    tracker = BestModelTracker(save_dir, higher_is_better=True)

    # ── Phase 0 ───────────────────────────────────────────────────────────────
    rb_path = os.path.join(save_dir, "replay_buffer.npz")
    if os.path.exists(rb_path):
        if verbose:
            print("\n[Phase 0] Loading existing replay buffer...")
        rb = load_replay_buffer(rb_path, device=device)
        task_policies: dict = {}
        if verbose:
            print(f"  Loaded {len(rb)} transitions.")
    else:
        rb = ReplayBuffer(device=device)
        if verbose:
            print("\n[Phase 0] SAC bootstrap per task...")
        task_policies = phase0_collect_initial_data(
            task_dist, rb,
            total_steps=phase0_steps,
            n_envs=n_envs,
            device=device,
            verbose=verbose,
        )
        save_replay_buffer(rb, rb_path)

    # ── Phase 1: per-task skeletons ───────────────────────────────────────────
    if verbose:
        print("\n[Phase 1] Building per-task skeletons...")
    base_skeleton, per_task_skeletons = phase1_build_per_task_skeletons(
        rb, state_dim, action_dim,
        num_landmarks=num_landmarks,
        dbscan_eps=dbscan_eps,
        max_pool_size=max_pool_size,
        device=device,
        verbose=verbose,
    )
    metrics["skeleton_train_losses"].append(base_skeleton.get("train_losses", []))

    # ── Intersection ──────────────────────────────────────────────────────────
    if verbose:
        print("\n[Intersection] Extracting common critical structure...")
    intersection_skeleton = extract_intersection_skeleton(
        base_skeleton,
        intersection_support=intersection_support,
        dbscan_eps=dbscan_eps,
        centrality_threshold=centrality_threshold,
        verbose=verbose,
    )

    for skel in per_task_skeletons.values():
        skel["_potential_stale"] = True
    intersection_skeleton["_potential_stale"] = True

    plot_skeleton_topology(
        base_skeleton, rb, os.path.join(save_dir, "topology_initial.png")
    )

    if not intersection_skeleton.get("critical_states") and verbose:
        print(
            "  WARNING: Intersection skeleton is empty — meta-policy will train "
            "without skeleton shaping.  Consider lowering --intersection-support."
        )

    # Meta-policy created once; persists across iterations
    meta_policy = MetaPolicy(
        state_dim, action_dim,
        discrete=False,
        action_low=action_low,
        action_high=action_high,
    ).to(device)
    training_state = None

    for iteration in range(num_iterations):
        if verbose:
            print(f"\n{'─' * 60}")
            print(f"Iteration {iteration + 1}/{num_iterations}")
            print(f"{'─' * 60}")

        current_alpha = _alpha_sched.alpha(iteration)

        # ── Phase 2a: per-task potentials ─────────────────────────────────────
        if verbose:
            print(f"\n[Phase 2a] Per-task potentials (α={current_alpha:.3f})...")
        per_task_potentials = phase2_build_per_task_potentials(
            per_task_skeletons, rb,
            alpha=current_alpha,
            gamma=gamma,
            verbose=verbose,
            use_progress=use_progress,
            progress_latent_dim=progress_latent_dim,
            progress_epochs=progress_epochs,
            progress_weight=progress_weight,
            device=device,
        )

        # ── Phase 2b: intersection potential ──────────────────────────────────
        if verbose:
            print(f"\n[Phase 2b] Intersection potential (α={current_alpha:.3f})...")
        intersection_potential = phase2_build_intersection_potential(
            intersection_skeleton, rb,
            alpha=current_alpha,
            gamma=gamma,
            verbose=verbose,
        )

        # ── Per-task shaping scales ────────────────────────────────────────────
        # median(|r_env_i|) * shaping_scale so magnitudes are equalised across tasks
        per_task_shaping_scales = compute_per_task_shaping_scales(
            rb, task_dist, desired_shaping_fraction=shaping_scale,
        )
        if verbose:
            scale_str = "  ".join(
                f"{t.env_name}={per_task_shaping_scales.get(t.id, 1.0):.3g}"
                for t in task_dist.tasks
            )
            print(f"  [Scales] {scale_str}")

        # ── Phase 3: task policies with per-task potentials ───────────────────
        if verbose:
            print("\n[Phase 3] Training task policies with per-task potentials...")
        task_policies, p3_stats = phase3_train_task_policies(
            task_dist, per_task_potentials, rb,
            timesteps_per_task=task_policy_steps,
            shaping_scales=per_task_shaping_scales,
            device=device,
            verbose=verbose,
            save_dir=save_dir,
            iteration=iteration,
            existing_task_policies=task_policies,
        )
        p3_env_returns = {
            name: float(np.mean(v["ep_env_rewards"][-20:])) if v["ep_env_rewards"] else 0.0
            for name, v in p3_stats.items()
        }
        metrics["phase3_env_returns"].append(p3_env_returns)

        # ── Intersection diagnostic + fallback ────────────────────────────────
        n_common = _subgoal_diagnostic(per_task_skeletons, intersection_skeleton, iteration + 1)

        if n_common == 0:
            # Hard intersection was too strict — retry with a softer threshold
            fallback_support = max(0.5, 1.0 / max(len(task_dist.tasks), 1))
            print(
                f"\n  [Phase 4] Empty intersection — retrying with "
                f"intersection_support={fallback_support:.0%}..."
            )
            try:
                intersection_skeleton = extract_intersection_skeleton(
                    base_skeleton,
                    intersection_support=fallback_support,
                    dbscan_eps=dbscan_eps,
                    centrality_threshold=centrality_threshold,
                    verbose=verbose,
                )
                intersection_skeleton["_potential_stale"] = True
                n_common = _subgoal_diagnostic(
                    per_task_skeletons, intersection_skeleton, iteration + 1
                )
                if n_common > 0:
                    intersection_potential = phase2_build_intersection_potential(
                        intersection_skeleton, rb,
                        alpha=current_alpha, gamma=gamma, verbose=verbose,
                    )
                else:
                    print(
                        f"  [Phase 4] Still empty at {fallback_support:.0%} — "
                        "Phase 4 will run without intersection shaping."
                    )
            except Exception as exc:
                print(f"  [Phase 4] Fallback extraction failed ({exc}); continuing.")

        # ── Phase 4: meta-policy on intersection skeleton ─────────────────────
        if verbose:
            print(
                f"\n[Phase 4] Training meta-policy on intersection skeleton "
                f"(n_common={n_common})..."
            )
        training_skeleton = dict(intersection_skeleton)
        training_skeleton["skeleton_potential"] = intersection_potential

        meta_policy, _, p4_losses, training_state = (
            meta_policy_gradient_with_skeleton_shaping(
                meta_policy,
                task_dist,
                training_skeleton,
                meta_epochs=meta_epochs,
                gamma=gamma,
                shaping_scale=shaping_scale,
                shaping_scales=per_task_shaping_scales,
                subgoal_threshold=subgoal_threshold,
                max_episode_steps=1000,
                flush_buffer=True,
                device=device,
                verbose=verbose,
                training_state=training_state,
            )
        )
        metrics["phase4_losses"].append(p4_losses)

        # ── Evaluate ──────────────────────────────────────────────────────────
        if verbose:
            print("\n[Eval] Evaluating meta-policy...")
        per_task_returns = evaluate_meta_policy(
            meta_policy, task_dist,
            n_episodes_per_task=eval_episodes_per_task,
            device=device,
            verbose=verbose,
        )
        metrics["eval_returns"].append(per_task_returns)
        mean_return = float(np.mean(list(per_task_returns.values())))
        if verbose:
            print(f"  [Eval] mean_return across tasks: {mean_return:.2f}")

        # ── Checkpoint ────────────────────────────────────────────────────────
        ckpt_dir = save_checkpoint(
            save_dir,
            iteration=iteration,
            meta_policy=meta_policy,
            meta_value_net=None,
            task_policies=task_policies,
            skeleton_data=intersection_skeleton,
            replay_buffer=rb,
            metrics={
                "mean_return":       mean_return,
                "per_task_returns":  per_task_returns,
                "n_common_subgoals": n_common,
                "p4_avg_loss": (
                    float(np.mean([e["total"] for e in p4_losses]))
                    if p4_losses else 0.0
                ),
            },
        )
        improved = tracker.update(mean_return, ckpt_dir)
        if verbose and improved:
            print(f"  ★ New best model (mean_return={mean_return:.2f})")

        save_iteration_visuals(intersection_skeleton, rb, metrics, save_dir, iteration)
        save_replay_buffer(rb, rb_path)

        # ── Collect with meta-policy ───────────────────────────────────────────
        if verbose:
            print("\n[Collect] Rolling out meta-policy...")
        collect_with_meta_policy(
            meta_policy, task_dist, rb,
            num_episodes=collect_episodes,
            device=device,
        )
        save_replay_buffer(rb, rb_path)
        if verbose:
            print(f"  Buffer size: {len(rb)}")

        # ── Periodic skeleton refinement ──────────────────────────────────────
        if (iteration + 1) % refine_every == 0 and iteration < num_iterations - 1:
            if verbose:
                print("\n[Refine] Rebuilding per-task skeletons on enlarged buffer...")
            base_skeleton, per_task_skeletons = phase1_build_per_task_skeletons(
                rb, state_dim, action_dim,
                num_landmarks=num_landmarks,
                dbscan_eps=dbscan_eps,
                max_pool_size=max_pool_size,
                device=device,
                verbose=verbose,
            )
            for skel in per_task_skeletons.values():
                skel["_potential_stale"] = True
            metrics["skeleton_train_losses"].append(base_skeleton.get("train_losses", []))

            if verbose:
                print("[Refine] Re-extracting intersection skeleton...")
            intersection_skeleton = extract_intersection_skeleton(
                base_skeleton,
                intersection_support=intersection_support,
                dbscan_eps=dbscan_eps,
                centrality_threshold=centrality_threshold,
                verbose=verbose,
            )
            intersection_skeleton["_potential_stale"] = True
            n_refined = len(
                intersection_skeleton.get("meta_subgoals")
                or intersection_skeleton.get("critical_states", {})
            )
            if verbose:
                print(f"  Refined intersection: {n_refined} common subgoal(s).")

    plot_training_curves(metrics, save_dir)
    return meta_policy, task_policies, intersection_skeleton, metrics


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Intersection Meta-RL on MuJoCo Gymnasium environments."
    )
    parser.add_argument("--task-set",    default="walker2d",
                        choices=["walker2d"],
                        help="Predefined task set (default: walker2d — "
                             "baseline, heavy-torso, slippery-floor variants)")
    parser.add_argument("--envs",        nargs="+", default=None, metavar="ENV",
                        help="Custom Gymnasium env IDs (overrides --task-set)")
    parser.add_argument("--iterations",  type=int,   default=5)
    parser.add_argument("--landmarks",   type=int,   default=200)
    parser.add_argument("--meta-epochs", type=int,   default=300)
    parser.add_argument("--phase0-steps",      type=int,   default=30_000)
    parser.add_argument("--task-steps",        type=int,   default=10_000)
    parser.add_argument("--n-envs",            type=int,   default=4)
    parser.add_argument("--collect-episodes",  type=int,   default=20)
    parser.add_argument("--shaping-scale",     type=float, default=1.0)
    parser.add_argument("--subgoal-threshold", type=float, default=float("inf"))
    parser.add_argument("--potential-alpha",   type=float, default=0.5)
    parser.add_argument("--alpha-anneal-iters",type=int,   default=3)
    parser.add_argument(
        "--intersection-support", type=float, default=1.0,
        help="Fraction of tasks that must share a subgoal for it to enter the "
             "intersection skeleton (default: 1.0 = all tasks; try 0.67 for 2/3).",
    )
    parser.add_argument(
        "--centrality-threshold", type=float, default=0.0,
        help="Drop intersection subgoals below this centrality score (default: 0.0 = keep all).",
    )
    parser.add_argument("--gamma",             type=float, default=0.99)
    parser.add_argument("--refine-every",      type=int,   default=2)
    parser.add_argument("--eval-episodes",     type=int,   default=5)
    parser.add_argument("--dbscan-eps",        type=float, default=None)
    parser.add_argument("--max-pool-size",     type=int,   default=None)
    parser.add_argument("--use-progress",      action="store_true")
    parser.add_argument("--progress-latent-dim", type=int,   default=128)
    parser.add_argument("--progress-epochs",     type=int,   default=20)
    parser.add_argument("--progress-weight",     type=float, default=0.5)
    parser.add_argument("--save-dir",  default="results/mujoco_intersection")
    parser.add_argument("--device",    default="cpu")
    parser.add_argument("--no-verbose", action="store_true")
    args = parser.parse_args()

    verbose = not args.no_verbose

    if args.envs:
        task_dist = GymTaskDistribution.from_env_ids(args.envs)
    else:
        task_dist = make_walker2d_task_dist()

    state_dim, action_dim, action_low, action_high = _check_dims_consistent(task_dist)

    if verbose:
        print(
            f"Tasks: {[t.env_name for t in task_dist.tasks]}\n"
            f"  state_dim={state_dim}  action_dim={action_dim}"
            f"  action=[{action_low}, {action_high}]"
        )

    main_intersection_rl_loop(
        task_dist=task_dist,
        state_dim=state_dim,
        action_dim=action_dim,
        action_low=action_low,
        action_high=action_high,
        num_landmarks=args.landmarks,
        num_iterations=args.iterations,
        refine_every=args.refine_every,
        phase0_steps=args.phase0_steps,
        n_envs=args.n_envs,
        task_policy_steps=args.task_steps,
        collect_episodes=args.collect_episodes,
        gamma=args.gamma,
        meta_epochs=args.meta_epochs,
        shaping_scale=args.shaping_scale,
        subgoal_threshold=args.subgoal_threshold,
        potential_alpha=args.potential_alpha,
        alpha_anneal_iters=args.alpha_anneal_iters,
        intersection_support=args.intersection_support,
        centrality_threshold=args.centrality_threshold,
        dbscan_eps=args.dbscan_eps,
        max_pool_size=args.max_pool_size,
        eval_episodes_per_task=args.eval_episodes,
        use_progress=args.use_progress,
        progress_latent_dim=args.progress_latent_dim,
        progress_epochs=args.progress_epochs,
        progress_weight=args.progress_weight,
        save_dir=args.save_dir,
        device=args.device,
        verbose=verbose,
    )


if __name__ == "__main__":
    main()
