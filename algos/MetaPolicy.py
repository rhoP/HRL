"""
Meta-RL orchestration for MetaWorld benchmarks.

Phases:
  0. Collect experience per task using SB3 (SAC/PPO).
  1. Build common Meta-Morse skeleton (meta-subgoals from DBSCAN intersection).
  2. Build potential function over the skeleton.
  3. Train one task policy per task with potential-shaped rewards.
  4. Train meta-policy π_θ(a | s, τ) via policy gradient with skeleton shaping.
  5. Repeat.

Entry point:
    python3 algos/MetaPolicy.py --tasks reach-v3 push-v3 \\
        --iterations 3 --landmarks 32 --meta-epochs 500 --save-dir results/run1
"""

import argparse
import sys
import os

import numpy as np
import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import metaworld
import gymnasium as gym
from stable_baselines3 import SAC, PPO
from stable_baselines3.common.env_util import make_vec_env

from models.meta_policy_net import MetaPolicy, MetaValueNetwork
from algos.potential import (
    SkeletonPotential,
    EmpiricalHittingTimePotential,
    CombinedPotential,
    ShapedRewardWrapper,
)
from algos.meta_policy_gradient import (
    meta_policy_gradient_with_skeleton_shaping,
    evaluate_meta_policy,
)
from utils.replay_buffer import ReplayBuffer
from utils.skeleton import build_skeleton, refine_skeleton
from utils.checkpoint import (
    save_checkpoint,
    load_checkpoint,
    restore_models,
    BestModelTracker,
    save_replay_buffer,
    load_replay_buffer,
)
from utils.viz import (
    plot_training_curves,
    plot_skeleton_topology,
    save_iteration_visuals,
    run_demos,
    Phase3TrainingCallback,
    plot_phase3_results,
)


# ── MetaWorld state/action dimensions ─────────────────────────────────────

MW_STATE_DIM = 39
MW_ACTION_DIM = 4


# ── Task / Distribution ────────────────────────────────────────────────────


class MetaWorldTask:
    """Wraps a single MetaWorld (env_name, mw_task) pair."""

    def __init__(self, task_id: int, env_cls, mw_task):
        self.id = task_id
        self.env_name = mw_task.env_name
        self._env_cls = env_cls
        self._mw_task = mw_task

    def create_env(self):
        env = self._env_cls()
        env.set_task(self._mw_task)
        return env


class MetaWorldTaskDistribution:
    """Uniform distribution over a list of MetaWorldTask objects."""

    def __init__(self, tasks: list):
        self.tasks = tasks

    def sample(self) -> MetaWorldTask:
        return self.tasks[np.random.randint(len(self.tasks))]

    @classmethod
    def from_env_names(cls, env_names: list, max_tasks_per_env: int = 5):
        tasks = []
        task_id = 0
        for env_name in env_names:
            ml1 = metaworld.ML1(env_name)
            env_cls = ml1.train_classes[env_name]
            for mw_task in ml1.train_tasks[:max_tasks_per_env]:
                tasks.append(MetaWorldTask(task_id, env_cls, mw_task))
                task_id += 1
        return cls(tasks)

    @classmethod
    def from_benchmark(cls, benchmark, max_tasks_per_env: int = 10):
        tasks = []
        task_id = 0
        env_to_cls = benchmark.train_classes
        env_to_tasks: dict = {}
        for mw_task in benchmark.train_tasks:
            env_to_tasks.setdefault(mw_task.env_name, []).append(mw_task)
        for env_name, mw_task_list in env_to_tasks.items():
            env_cls = env_to_cls[env_name]
            for mw_task in mw_task_list[:max_tasks_per_env]:
                tasks.append(MetaWorldTask(task_id, env_cls, mw_task))
                task_id += 1
        return cls(tasks)


# ── Gymnasium-compatible wrapper ───────────────────────────────────────────


class MetaWorldGymWrapper(gym.Env):
    """Makes a MetaWorld env look like a standard Gymnasium env."""

    metadata = {"render_modes": []}

    def __init__(self, mw_task: MetaWorldTask):
        super().__init__()
        self._task = mw_task
        self._env = mw_task.create_env()
        raw_obs = self._env.observation_space
        raw_act = self._env.action_space
        self.observation_space = gym.spaces.Box(
            low=raw_obs.low.astype(np.float32),
            high=raw_obs.high.astype(np.float32),
            dtype=np.float32,
        )
        self.action_space = gym.spaces.Box(
            low=raw_act.low.astype(np.float32),
            high=raw_act.high.astype(np.float32),
            dtype=np.float32,
        )

    def reset(self, *, seed=None, options=None):
        obs, info = self._env.reset()
        return np.asarray(obs, dtype=np.float32).flatten(), info

    def step(self, action):
        obs, r, terminated, truncated, info = self._env.step(action)
        return (
            np.asarray(obs, dtype=np.float32).flatten(),
            float(r),
            terminated,
            truncated,
            info,
        )

    def close(self):
        self._env.close()


# ── Phase 0 ────────────────────────────────────────────────────────────────


def phase0_collect_initial_data(
    task_distribution: MetaWorldTaskDistribution,
    replay_buffer: ReplayBuffer,
    timesteps_per_task: int = 5_000,
    n_envs: int = 10,
    algo: str = "SAC",
    device: str = "cpu",
    verbose: bool = True,
) -> None:
    """
    Use SAC (or PPO) from stable_baselines3 to fill the replay buffer.

    n_envs parallel copies of each task env are run simultaneously.
    The SB3 model trains for timesteps_per_task total steps across all envs,
    then a separate rollout of the same length collects transitions into
    replay_buffer (steps_per_env = timesteps_per_task // n_envs per env so
    total transitions ≈ timesteps_per_task).
    """
    AlgoCls = SAC if algo.upper() == "SAC" else PPO

    for task in task_distribution.tasks:
        if verbose:
            print(
                f"  [Phase 0] {task.env_name} (task {task.id}) — "
                f"{algo} × {n_envs} envs, {timesteps_per_task} steps..."
            )

        def _make_env(t=task):
            return MetaWorldGymWrapper(t)

        vec_env = make_vec_env(_make_env, n_envs=n_envs)
        model = AlgoCls("MlpPolicy", vec_env, verbose=0, device=device)
        model.learn(total_timesteps=timesteps_per_task)

        # Collect transitions from all envs in parallel.
        # VecEnv auto-resets on episode end — obs_next[i] is the reset obs when
        # done[i] is True, so we never call vec_env.reset() manually here.
        obs = vec_env.reset()
        steps_per_env = max(1, timesteps_per_task // n_envs)
        for _ in range(steps_per_env):
            action, _ = model.predict(obs, deterministic=False)
            obs_next, reward, done, info = vec_env.step(action)
            for i in range(n_envs):
                trunc_i = info[i].get("TimeLimit.truncated", False)
                term_i = bool(done[i])  # and not trunc_i
                replay_buffer.push(
                    obs[i],
                    action[i],
                    float(reward[i]),
                    obs_next[i],
                    bool(done[i]),
                    task.id,
                    terminated=term_i,
                )
            obs = obs_next
        vec_env.close()

    if verbose:
        print(f"  [Phase 0] Buffer size: {len(replay_buffer)}")


# ── Phase 1 ────────────────────────────────────────────────────────────────


def phase1_build_skeleton(
    replay_buffer: ReplayBuffer,
    state_dim: int = MW_STATE_DIM,
    action_dim: int = MW_ACTION_DIM,
    num_landmarks: int = 32,
    device: str = "cpu",
    verbose: bool = True,
    **kwargs,
) -> dict:
    """Build Meta-Morse skeleton; meta-subgoals from trajectory topology."""
    if replay_buffer.state_dim is None:
        raise RuntimeError("replay_buffer is empty — run Phase 0 first.")
    return build_skeleton(
        replay_buffer,
        state_dim=state_dim,
        action_dim=action_dim,
        num_landmarks=num_landmarks,
        device=device,
        verbose=verbose,
        **kwargs,
    )


# ── Phase 2 ────────────────────────────────────────────────────────────────


def phase2_build_potential(
    skeleton_data: dict,
    replay_buffer,
    alpha: float = 0.5,
    k: int = 10,
    gamma: float = 0.99,
    hit_threshold: float = 0.5,
    verbose: bool = True,
):
    """
    Build a CombinedPotential over the meta-skeleton.

    Φ(s; c) = α · Φ̃_skeleton(s; c)  +  (1−α) · Φ̃_empirical(s; c)

    Both components are normalised to unit std over the landmark set before
    mixing.  α=1 → pure graph topology; α=0 → pure empirical returns.

    The SkeletonPotential is cached in skeleton_data["_skel_potential_cached"]
    and rebuilt only when skeleton_data["_potential_stale"] is True, preserving
    the shortest-path cache across iterations.  The EmpiricalHittingTimePotential
    is always rebuilt so the empirical component improves as data accumulates.
    """
    raw = skeleton_data.get("meta_subgoals", {})
    if raw and not isinstance(next(iter(raw.values())), dict):
        raw = {k: {"state": v} for k, v in raw.items()}
    if not raw:
        raw = skeleton_data.get("critical_states", {})
        raw = {k: {"state": v} for k, v in raw.items()}
    meta_subgoals = {
        k: {"state": np.asarray(v["state"], dtype=np.float32)} for k, v in raw.items()
    }

    lm = skeleton_data["landmarks"]
    lm_np = lm.cpu().numpy() if hasattr(lm, "cpu") else np.asarray(lm, dtype=np.float32)

    # Skeleton component — rebuild only when topology is stale
    if (
        skeleton_data.get("_potential_stale", True)
        or "_skel_potential_cached" not in skeleton_data
    ):
        skel_pot = SkeletonPotential(lm_np, skeleton_data["simplices"], meta_subgoals)
        skeleton_data["_skel_potential_cached"] = skel_pot
        skeleton_data["_potential_stale"] = False
        if verbose:
            print(
                f"  [Phase 2] SkeletonPotential ({len(meta_subgoals)} subgoal(s)): "
                f"{skel_pot.G.number_of_nodes()} nodes, "
                f"{skel_pot.G.number_of_edges()} edges"
            )
    else:
        skel_pot = skeleton_data["_skel_potential_cached"]
        if verbose:
            print(f"  [Phase 2] Reusing cached SkeletonPotential.")

    # Empirical component — always rebuilt (buffer grows each iteration)
    emp_pot = EmpiricalHittingTimePotential(
        replay_buffer,
        meta_subgoals,
        k=k,
        gamma=gamma,
        hit_threshold=hit_threshold,
    )
    n_covered = sum(len(v) > 0 for v in emp_pot._trajs.values())
    if verbose:
        print(
            f"  [Phase 2] EmpiricalHittingTimePotential (k={k}, γ={gamma}): "
            f"{n_covered}/{len(meta_subgoals)} subgoal(s) with hitting trajectories"
        )

    combined = CombinedPotential(skel_pot, emp_pot, lm_np, alpha=alpha)
    if verbose:
        print(
            f"  [Phase 2] CombinedPotential α={alpha:.2f}  "
            f"skel_scale={combined._skel_scale:.4f}  "
            f"emp_scale={combined._emp_scale:.4f}"
        )
    return combined


# ── Phase 3 ────────────────────────────────────────────────────────────────


def phase3_train_task_policies(
    skeleton_data: dict,
    task_distribution: MetaWorldTaskDistribution,
    potential=None,
    timesteps_per_task: int = 10_000,
    shaping_scale: float = 1.0,
    algo: str = "SAC",
    device: str = "cpu",
    verbose: bool = True,
    save_dir: str = None,
    iteration: int = 0,
) -> tuple:
    """
    Train one SB3 policy per task with potential-based shaped rewards.

    Returns (task_policies, phase3_stats) where:
        task_policies  — {task_id: SB3 model}
        phase3_stats   — {task_name: {"ep_rewards", "ep_env_rewards",
                                      "ep_shaping", "ep_successes", "ep_lengths"}}
    If save_dir is provided, saves phase3_iter_{N:03d}.png there.
    """
    AlgoCls = SAC if algo.upper() == "SAC" else PPO
    task_policies = {}
    phase3_stats: dict = {}

    for task in task_distribution.tasks:
        if verbose:
            print(
                f"  [Phase 3] Training policy for {task.env_name} (task {task.id})..."
            )

        def _make_env(t=task, pot=potential, scale=shaping_scale):
            env = MetaWorldGymWrapper(t)
            if pot is not None:
                env = ShapedRewardWrapper(env, pot, shaping_scale=scale)
            return env

        cb = Phase3TrainingCallback()
        vec_env = make_vec_env(_make_env, n_envs=1)
        model = AlgoCls("MlpPolicy", vec_env, verbose=0, device=device)
        model.learn(total_timesteps=timesteps_per_task, callback=cb)
        task_policies[task.id] = model
        vec_env.close()

        phase3_stats[task.env_name] = {
            "ep_rewards": cb.ep_rewards,
            "ep_env_rewards": cb.ep_env_rewards,
            "ep_shaping": cb.ep_shaping,
            "ep_successes": cb.ep_successes,
            "ep_lengths": cb.ep_lengths,
        }
        if verbose and cb.ep_rewards:
            last20 = cb.ep_rewards[-20:]
            sr = float(np.mean(cb.ep_successes[-20:])) if cb.ep_successes else 0.0
            print(
                f"    last-20-ep: avg_shaped_r={np.mean(last20):.3f}  "
                f"success_rate={sr:.1%}"
            )

    if save_dir is not None:
        plot_phase3_results(phase3_stats, save_dir, iteration=iteration)

    if verbose:
        print(f"  [Phase 3] Trained {len(task_policies)} task policy/ies.")
    return task_policies, phase3_stats


# ── Collect with meta-policy ───────────────────────────────────────────────


def collect_with_meta_policy(
    meta_policy,
    task_distribution: MetaWorldTaskDistribution,
    replay_buffer: ReplayBuffer,
    num_episodes: int = 20,
    max_steps: int = 500,
    device: str = "cpu",
) -> None:
    """Roll out trained meta-policy; push transitions into replay_buffer."""
    meta_policy.eval()
    for _ in range(num_episodes):
        task = task_distribution.sample()
        env = task.create_env()
        result = env.reset()
        s = result[0] if isinstance(result, tuple) else result
        tau = []
        done = False
        t = 0

        while not done and t < max_steps:
            with torch.no_grad():
                a_dist = meta_policy(s, tau)
                a = a_dist.sample()
            a_np = a.cpu().numpy().flatten()
            a_env = int(a_np[0]) if meta_policy.discrete else a_np
            s_next, r, terminated, truncated, _ = env.step(a_env)
            done = terminated or truncated
            s_arr = np.asarray(s, dtype=np.float32)
            replay_buffer.push(
                s_arr,
                a_np,
                r,
                np.asarray(s_next, dtype=np.float32),
                done,
                task.id,
                terminated=terminated,
            )
            tau.append((s_arr, a_np, r))
            s = s_next
            t += 1
        env.close()
    meta_policy.train()


# ── Main orchestration ─────────────────────────────────────────────────────


def main_meta_rl_loop(
    task_distribution: MetaWorldTaskDistribution,
    state_dim: int = MW_STATE_DIM,
    action_dim: int = MW_ACTION_DIM,
    num_landmarks: int = 32,
    num_iterations: int = 3,
    refine_every: int = 2,
    timesteps_per_task: int = 5_000,
    n_envs: int = 10,
    task_policy_steps: int = 10_000,
    collect_episodes: int = 20,
    gamma: float = 0.97,
    meta_epochs: int = 1000,
    shaping_scale: float = 1.0,
    subgoal_threshold: float = float("inf"),
    potential_alpha: float = 0.5,
    state_projection_fn=None,
    algo: str = "SAC",
    eval_episodes: int = 10,
    n_demos: int = 5,
    save_dir: str = "results/meta_rl",
    device: str = "cpu",
    verbose: bool = True,
):
    """
    Full meta-RL pipeline:
        Phase 0 → Phase 1 → (Phase 2 → Phase 3 → Phase 4 →
            evaluate → checkpoint → collect → refine)*
    """
    os.makedirs(save_dir, exist_ok=True)
    discrete = False  # MetaWorld is continuous

    if verbose:
        print("=" * 60)
        print("Meta-RL pipeline  (MetaWorld)")
        print(
            f"  tasks: {len(task_distribution.tasks)}, "
            f"landmarks: {num_landmarks}, iterations: {num_iterations}"
        )
        print(f"  potential_alpha: {potential_alpha}  shaping_scale: {shaping_scale}")
        print(f"  save_dir: {save_dir}")
        print("=" * 60)

    metrics = {
        "skeleton_train_losses": [],
        "phase3_success_rates": [],
        "phase4_losses": [],
        "eval_success_rates": [],
    }

    tracker = BestModelTracker(save_dir, higher_is_better=True)

    # Phase 0
    rb_path = os.path.join(save_dir, "replay_buffer.npz")
    if os.path.exists(rb_path):
        if verbose:
            print("\n[Phase 0] Loading existing replay buffer...")
        rb = load_replay_buffer(rb_path, device=device)
        if verbose:
            print(f"  Loaded {len(rb)} transitions.")
    else:
        rb = ReplayBuffer(device=device)
        if verbose:
            print("\n[Phase 0] Collecting initial data with SB3...")
        phase0_collect_initial_data(
            task_distribution,
            rb,
            timesteps_per_task=timesteps_per_task,
            n_envs=n_envs,
            algo=algo,
            device=device,
            verbose=verbose,
        )
        save_replay_buffer(rb, rb_path)

    # Phase 1
    if verbose:
        print("\n[Phase 1] Building Meta-Morse skeleton...")
    skeleton = phase1_build_skeleton(
        rb,
        state_dim=state_dim,
        action_dim=action_dim,
        num_landmarks=num_landmarks,
        state_projection_fn=state_projection_fn,
        min_task_support=0.4,
        device=device,
        verbose=verbose,
    )
    metrics["skeleton_train_losses"].append(skeleton.get("train_losses", []))
    n_sub = len(skeleton["critical_states"])
    if verbose:
        print(f"  Found {n_sub} meta-subgoal(s).")

    skeleton["_potential_stale"] = True
    plot_skeleton_topology(skeleton, rb, os.path.join(save_dir, "topology_initial.png"))

    if n_sub == 0:
        print("No meta-subgoals found; aborting.")
        plot_training_curves(metrics, save_dir)
        return None, None, skeleton, metrics

    # Create meta-policy once; reuse and refine across iterations
    meta_policy = MetaPolicy(state_dim, action_dim, discrete=discrete).to(device)
    meta_value_net = None
    training_state = None

    for iteration in range(num_iterations):
        if verbose:
            print(f"\n{'─' * 60}")
            print(f"Iteration {iteration + 1}/{num_iterations}")
            print(f"{'─' * 60}")

        # Phase 2 — combined potential (skeleton + empirical, normalised)
        if verbose:
            print(f"[Phase 2] Building combined potential (α={potential_alpha:.2f})...")
        potential = phase2_build_potential(
            skeleton,
            rb,
            alpha=potential_alpha,
            gamma=gamma,
            verbose=verbose,
        )

        # Phase 3
        if verbose:
            print("[Phase 3] Training task policies with shaped rewards...")
        task_policies, p3_stats = phase3_train_task_policies(
            skeleton,
            task_distribution,
            potential=potential,
            timesteps_per_task=task_policy_steps,
            shaping_scale=shaping_scale,
            algo=algo,
            device=device,
            verbose=verbose,
            save_dir=save_dir,
            iteration=iteration,
        )
        p3_sr = float(
            np.mean(
                [
                    np.mean(v["ep_successes"][-20:])
                    for v in p3_stats.values()
                    if v["ep_successes"]
                ]
            )
            if p3_stats
            else 0.0
        )
        metrics["phase3_success_rates"].append(p3_sr)

        # Phase 4 — meta-policy gradient with skeleton shaping
        if verbose:
            print("[Phase 4] Training meta-policy via policy gradient...")
        training_skeleton = dict(skeleton)
        training_skeleton["skeleton_potential"] = potential

        meta_policy, meta_value_net, p4_losses, training_state = (
            meta_policy_gradient_with_skeleton_shaping(
                meta_policy,
                task_distribution,
                training_skeleton,
                meta_epochs=meta_epochs,
                gamma=gamma,
                shaping_scale=shaping_scale,
                subgoal_threshold=subgoal_threshold,
                flush_buffer=True,
                device=device,
                verbose=verbose,
                training_state=training_state,
            )
        )
        metrics["phase4_losses"].append(p4_losses)

        # Evaluate
        if verbose:
            print("[Eval] Evaluating meta-policy...")
        sr = evaluate_meta_policy(
            meta_policy,
            task_distribution,
            n_episodes=eval_episodes,
            device=device,
        )
        metrics["eval_success_rates"].append(sr)
        if verbose:
            print(f"  success_rate={sr:.1%}")

        # Checkpoint
        ckpt_dir = save_checkpoint(
            save_dir,
            iteration=iteration,
            meta_policy=meta_policy,
            meta_value_net=meta_value_net,
            task_policies=task_policies,
            skeleton_data=skeleton,
            replay_buffer=rb,
            metrics={
                "success_rate": sr,
                "p4_avg_loss": float(np.mean(p4_losses)) if p4_losses else 0.0,
            },
        )
        improved = tracker.update(sr, ckpt_dir)
        if verbose and improved:
            print(f"  ★ New best model (success_rate={sr:.1%})")

        save_iteration_visuals(skeleton, rb, metrics, save_dir, iteration)

        # Collect more data
        if verbose:
            print("[Collect] Rolling out meta-policy for additional data...")
        collect_with_meta_policy(
            meta_policy,
            task_distribution,
            rb,
            num_episodes=collect_episodes,
            device=device,
        )
        save_replay_buffer(rb, rb_path)
        if verbose:
            print(f"  Buffer size: {len(rb)}")

        # Periodic skeleton refinement
        if (iteration + 1) % refine_every == 0 and iteration < num_iterations - 1:
            if verbose:
                print("[Refine] Rebuilding skeleton on enlarged buffer...")
            skeleton = refine_skeleton(
                skeleton,
                rb,
                num_landmarks=num_landmarks,
                state_projection_fn=state_projection_fn,
                min_task_support=0.4,
                device=device,
                verbose=verbose,
            )
            skeleton["_potential_stale"] = True
            metrics["skeleton_train_losses"].append(skeleton.get("train_losses", []))
            n_sub = len(skeleton["critical_states"])
            if verbose:
                print(f"  Refined skeleton: {n_sub} meta-subgoal(s).")
            if n_sub == 0:
                print("  No subgoals after refinement; stopping.")
                break

    plot_training_curves(metrics, save_dir)

    best_dir = os.path.join(save_dir, "best")
    if os.path.isdir(best_dir) and n_demos > 0:
        if verbose:
            print(f"\n[Demo] Running {n_demos} demos with best model...")
        try:
            best_ckpt = load_checkpoint(best_dir, device=device)
            demo_mp, _, demo_tp, demo_skel = restore_models(
                best_ckpt,
                state_dim,
                action_dim,
                discrete=discrete,
                device=device,
            )
            run_demos(
                demo_mp,
                demo_tp,
                demo_skel,
                task_distribution,
                save_dir=os.path.join(save_dir, "demos"),
                n_demos=n_demos,
                render=True,
                gamma=gamma,
                device=device,
            )
        except Exception as e:
            print(f"  [Demo] Could not run demos: {e}")

    if verbose:
        print("\nMeta-RL pipeline complete.")

    return meta_policy, task_policies, skeleton, metrics


# ── Entry point ────────────────────────────────────────────────────────────

_DEFAULT_TASKS = ["reach-v3", "push-v3", "pick-place-v3"]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Meta-RL on MetaWorld")
    parser.add_argument("--tasks", nargs="+", default=_DEFAULT_TASKS, metavar="ENV")
    parser.add_argument("--max-tasks", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--landmarks", type=int, default=100)
    parser.add_argument("--meta-epochs", type=int, default=500)
    parser.add_argument("--task-steps", type=int, default=10_000)
    parser.add_argument("--timesteps", type=int, default=50_000)
    parser.add_argument("--shaping-scale", type=float, default=1.0)
    parser.add_argument("--subgoal-threshold", type=float, default=float("inf"))
    parser.add_argument(
        "--potential-alpha",
        type=float,
        default=0.5,
        help="α for combined potential: α·skeleton + (1−α)·empirical (default: 0.5)",
    )
    parser.add_argument(
        "--n-envs",
        type=int,
        default=10,
        help="parallel envs for Phase 0 data collection (default: 10)",
    )
    parser.add_argument("--algo", default="PPO", choices=["SAC", "PPO"])
    parser.add_argument("--eval-episodes", type=int, default=5)
    parser.add_argument("--n-demos", type=int, default=10)
    parser.add_argument("--save-dir", default="results/meta_rl")
    parser.add_argument("--load", default=None, metavar="CKPT_DIR")
    parser.add_argument("--demo-only", action="store_true")
    parser.add_argument("--no-render", action="store_true")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    dist = MetaWorldTaskDistribution.from_env_names(
        args.tasks, max_tasks_per_env=args.max_tasks
    )
    print(f"Task distribution: {len(dist.tasks)} tasks  ({', '.join(args.tasks)})")

    if args.demo_only:
        if args.load is None:
            parser.error("--demo-only requires --load <checkpoint_dir>")
        ckpt = load_checkpoint(args.load, device=args.device)
        mp, _, tp, skel = restore_models(
            ckpt,
            MW_STATE_DIM,
            MW_ACTION_DIM,
            discrete=False,
            device=args.device,
        )
        run_demos(
            mp,
            tp,
            skel,
            dist,
            save_dir=os.path.join(args.save_dir, "demos"),
            n_demos=args.n_demos,
            render=not args.no_render,
            device=args.device,
        )
    else:
        main_meta_rl_loop(
            task_distribution=dist,
            state_dim=MW_STATE_DIM,
            action_dim=MW_ACTION_DIM,
            num_landmarks=args.landmarks,
            num_iterations=args.iterations,
            timesteps_per_task=args.timesteps,
            n_envs=args.n_envs,
            task_policy_steps=args.task_steps,
            gamma=0.99,
            meta_epochs=args.meta_epochs,
            shaping_scale=args.shaping_scale,
            subgoal_threshold=args.subgoal_threshold,
            potential_alpha=args.potential_alpha,
            algo=args.algo,
            eval_episodes=args.eval_episodes,
            n_demos=args.n_demos,
            save_dir=args.save_dir,
            device=args.device,
            verbose=True,
        )
