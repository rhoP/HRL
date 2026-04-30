"""
Meta-RL on MuJoCo environments (default: HumanoidStandup-v5 and Humanoid-v5).

Follows the same phase structure as MetaPolicy.py but replaces MetaWorld with
plain Gymnasium environments.  The shared meta-policy π_θ(a | s, τ) is trained
across all tasks simultaneously; a common topology-guided potential provides the
reward shaping signal.

Phases per outer iteration:
  Phase 0  — Bootstrap SAC per task to fill the shared replay buffer (once).
  Phase 1  — Build cross-task Morse skeleton (meta-subgoals from DBSCAN on
              trajectories that appear in *both* task buffers).
  Phase 2  — CombinedPotential: α·Φ_skeleton + (1−α)·Φ_empirical_hitting_time.
  Phase 2b — Optional: blend with learned ProgressEstimator.
  Phase 3  — Continue one SB3 SAC policy per task with shaped rewards.
  Phase 4  — Train the shared GRU meta-policy via advantage-weighted PG.
  Eval     — Measure mean undiscounted return per task.
  Collect  — Roll meta-policy to grow the shared replay buffer.
  Refine   — Periodically rebuild skeleton on the enlarged buffer.

Usage:
    python scripts/train_mujoco_meta.py
    python scripts/train_mujoco_meta.py \\
        --envs HumanoidStandup-v5 Humanoid-v5 \\
        --iterations 5 --landmarks 200 --meta-epochs 300 --save-dir results/humanoid_meta
    python scripts/train_mujoco_meta.py \\
        --envs Hopper-v4 Walker2d-v4 \\
        --phase0-steps 30000 --task-steps 15000 --device cuda
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

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

torch.set_default_dtype(torch.float32)

from stable_baselines3 import SAC
from stable_baselines3.common.env_util import make_vec_env

from algos.potential import (
    AlphaScheduler,
    calibrate_hit_threshold,
    CombinedPotential,
    EmpiricalHittingTimePotential,
    ShapedRewardWrapper,
    SkeletonPotential,
)
from algos.progress import blend_with_progress
from algos.meta_policy_gradient import (
    meta_policy_gradient_with_skeleton_shaping,
)
from models.meta_policy_net import MetaPolicy, MetaValueNetwork
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
    Phase3TrainingCallback,
    plot_phase3_results,
)


# ── Task wrappers ──────────────────────────────────────────────────────────────

class GymTask:
    """A single Gymnasium env wrapped to match the task interface used by
    MetaPolicy.py and meta_policy_gradient.py."""

    def __init__(self, task_id: int, env_id: str):
        self.id       = task_id
        self.env_name = env_id

    def create_env(self, **make_kwargs) -> gym.Env:
        return gym.make(self.env_name, **make_kwargs)


class ParameterizedGymTask(GymTask):
    """GymTask whose MuJoCo model parameters are modified after gym.make().

    Modifications (mass, friction, gravity) are applied once to the MjModel
    object right after construction; because MuJoCo model arrays are mutable
    and persist for the model's lifetime, the changes survive all subsequent
    env.reset() calls without any extra bookkeeping.

    Args:
        task_id:             integer task identifier
        base_env_id:         Gymnasium env string passed to gym.make()
        label:               human-readable name used for logging and stats keys;
                             defaults to base_env_id
        body_mass_scale:     {body_name: float} — multiply each named body's
                             mass by the given factor (e.g. {"torso": 3.0})
        geom_friction_scale: multiply every geom's friction coefficients by
                             this factor (e.g. 0.2 for a slippery floor)
        gravity:             override gravity vector as (gx, gy, gz) in m/s²
                             (e.g. (2.0, 0, -9.81) simulates an ~11° incline)
    """

    def __init__(
        self,
        task_id: int,
        base_env_id: str,
        *,
        label: str = None,
        body_mass_scale: dict = None,
        geom_friction_scale: float = 1.0,
        gravity: tuple = None,
    ):
        super().__init__(task_id, base_env_id)
        self._base_env_id         = base_env_id
        self.env_name             = label or base_env_id
        self._body_mass_scale     = body_mass_scale or {}
        self._geom_friction_scale = geom_friction_scale
        self._gravity             = gravity

    def create_env(self, **make_kwargs) -> gym.Env:
        import mujoco as _mj
        env   = gym.make(self._base_env_id, **make_kwargs)
        model = env.unwrapped.model

        for body_name, scale in self._body_mass_scale.items():
            bid = _mj.mj_name2id(model, _mj.mjtObj.mjOBJ_BODY, body_name)
            if bid >= 0:
                model.body_mass[bid] *= scale

        if self._geom_friction_scale != 1.0:
            model.geom_friction[:] *= self._geom_friction_scale

        if self._gravity is not None:
            model.opt.gravity[:] = self._gravity

        return env


def make_walker2d_task_dist() -> "GymTaskDistribution":
    """Three Walker2d-v5 variants with identical obs/action spaces.

    Variant 0 — baseline:  standard Walker2d-v5
    Variant 1 — heavy:     torso mass × 3  (harder balance; requires more torque)
    Variant 2 — slippery:  all geom friction × 0.2  (feet slip; requires careful gait)
    """
    return GymTaskDistribution([
        ParameterizedGymTask(0, "Walker2d-v5",
                             label="Walker2d_baseline"),
        ParameterizedGymTask(1, "Walker2d-v5",
                             label="Walker2d_heavy",
                             body_mass_scale={"torso": 3.0}),
        ParameterizedGymTask(2, "Walker2d-v5",
                             label="Walker2d_slippery",
                             geom_friction_scale=0.2),
    ])


class GymTaskDistribution:
    """Uniform distribution over a list of GymTask objects."""

    def __init__(self, tasks: list):
        self.tasks = tasks

    def sample(self) -> GymTask:
        return self.tasks[np.random.randint(len(self.tasks))]

    @classmethod
    def from_env_ids(cls, env_ids: list) -> "GymTaskDistribution":
        return cls([GymTask(i, eid) for i, eid in enumerate(env_ids)])


# ── Environment helpers ────────────────────────────────────────────────────────

def _env_dims(env_id: str) -> tuple[int, int]:
    env = gym.make(env_id)
    s = int(np.prod(env.observation_space.shape))
    a = int(np.prod(env.action_space.shape))
    env.close()
    return s, a


def _check_dims_consistent(task_dist: GymTaskDistribution) -> tuple[int, int, float, float]:
    """Assert all tasks share state/action dims and bounds; return (state_dim, action_dim, low, high).

    All four values must be identical across tasks — a shared meta-policy and
    TanhNormal squashing require a common action space.
    """
    dims   = []
    bounds = []
    for t in task_dist.tasks:
        env = t.create_env()
        s   = int(np.prod(env.observation_space.shape))
        a   = int(np.prod(env.action_space.shape))
        low  = float(env.action_space.low.min())
        high = float(env.action_space.high.max())
        dims.append((s, a))
        bounds.append((low, high))
        env.close()

    if len(set(dims)) != 1:
        raise ValueError(
            f"All tasks must have identical state/action dims for a shared "
            f"meta-policy, got: {dict(zip([t.env_name for t in task_dist.tasks], dims))}"
        )
    if len(set(bounds)) != 1:
        raise ValueError(
            f"All tasks must have identical action bounds for TanhNormal squashing, "
            f"got: {dict(zip([t.env_name for t in task_dist.tasks], bounds))}"
        )
    s_dim, a_dim = dims[0]
    low, high    = bounds[0]
    return s_dim, a_dim, low, high


# ── Phase 0: bootstrap ─────────────────────────────────────────────────────────

def phase0_collect_initial_data(
    task_dist: GymTaskDistribution,
    replay_buffer: ReplayBuffer,
    total_steps: int = 30_000,
    n_envs: int = 4,
    device: str = "cpu",
    verbose: bool = True,
) -> dict:
    """SAC bootstrap per task; push all collected transitions to replay_buffer.

    Returns {task_id: model} so Phase 3 can warm-start from these weights.
    """
    task_policies: dict = {}

    for task in task_dist.tasks:
        if verbose:
            print(
                f"  [Phase 0] {task.env_name} (task {task.id})  "
                f"SAC × {n_envs} envs, {total_steps} steps..."
            )
        vec_env = make_vec_env(task.create_env, n_envs=n_envs)
        model = SAC("MlpPolicy", vec_env, verbose=0, device=device)
        model.learn(total_timesteps=total_steps)
        task_policies[task.id] = model

        # Collect via single-env API for accurate terminated/truncated split.
        collect_steps = max(5_000, total_steps // n_envs * 2)
        _collect_transitions(model, task, replay_buffer, n_steps=collect_steps)
        vec_env.close()
        if verbose:
            print(
                f"    buffer size after task {task.id}: {len(replay_buffer)}"
            )

    return task_policies


# ── Phase 1: skeleton ──────────────────────────────────────────────────────────

def phase1_build_skeleton(
    replay_buffer: ReplayBuffer,
    state_dim: int,
    action_dim: int,
    num_landmarks: int = 200,
    min_task_support: float = 0.4,
    device: str = "cpu",
    verbose: bool = True,
    **kwargs,
) -> dict:
    """Build cross-task Morse skeleton; meta-subgoals must appear in both tasks."""
    if replay_buffer.state_dim is None:
        raise RuntimeError("replay_buffer is empty — run Phase 0 first.")
    if verbose:
        n_eps = len(replay_buffer._ep_ends)
        ep_list = list(replay_buffer.iter_episodes())
        n_survived = sum(
            1 for ep in ep_list
            if not any(np.asarray(ep["terminated"], dtype=bool))
        )
        ep_lengths = [len(ep["rewards"]) for ep in ep_list]
        if ep_lengths:
            print(
                f"  [Phase 1] Building Morse skeleton  landmarks={num_landmarks}"
                f"  episodes={n_eps}  survived={n_survived}"
                f"  ep_len min={min(ep_lengths)}"
                f" median={int(np.median(ep_lengths))}"
                f" max={max(ep_lengths)}"
            )
    skeleton = build_skeleton(
        replay_buffer,
        state_dim=state_dim,
        action_dim=action_dim,
        num_landmarks=num_landmarks,
        min_task_support=min_task_support,
        device=device,
        verbose=verbose,
        survived_only=True,
        **kwargs,
    )
    n_sub = len(skeleton.get("meta_subgoals") or skeleton.get("critical_states", {}))
    if verbose:
        print(f"  [Phase 1] Found {n_sub} meta-subgoal(s).")
    skeleton["_potential_stale"] = True
    return skeleton


# ── Phase 2: potential ─────────────────────────────────────────────────────────

def phase2_build_potential(
    skeleton_data: dict,
    replay_buffer: ReplayBuffer,
    alpha: float = 0.5,
    k: int = 10,
    gamma: float = 0.99,
    hit_threshold: float = None,
    verbose: bool = True,
):
    """Build CombinedPotential: α·Φ_skeleton + (1−α)·Φ_empirical."""
    if hit_threshold is None:
        hit_threshold = calibrate_hit_threshold(replay_buffer)
        if verbose:
            print(f"  [Phase 2] hit_threshold auto-calibrated → {hit_threshold:.4g}")
    raw = skeleton_data.get("meta_subgoals", {})
    if raw and not isinstance(next(iter(raw.values())), dict):
        raw = {k: {"state": v} for k, v in raw.items()}
    if not raw:
        raw = skeleton_data.get("critical_states", {})
        raw = {k: {"state": v} for k, v in raw.items()}
    meta_subgoals = {
        k: {"state": np.asarray(v["state"], dtype=np.float32)} for k, v in raw.items()
    }

    if not meta_subgoals:
        return None

    lm = skeleton_data["landmarks"]
    lm_np = lm.cpu().numpy() if hasattr(lm, "cpu") else np.asarray(lm, dtype=np.float32)

    if skeleton_data.get("_potential_stale", True) or "_skel_potential_cached" not in skeleton_data:
        skel_pot = SkeletonPotential(lm_np, skeleton_data["simplices"], meta_subgoals)
        skeleton_data["_skel_potential_cached"] = skel_pot
        skeleton_data["_potential_stale"] = False
        if verbose:
            print(
                f"  [Phase 2] SkeletonPotential  "
                f"{skel_pot.G.number_of_nodes()} nodes  "
                f"{skel_pot.G.number_of_edges()} edges"
            )
    else:
        skel_pot = skeleton_data["_skel_potential_cached"]
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

    combined = CombinedPotential(skel_pot, emp_pot, lm_np, alpha=alpha,
                                 replay_buffer=replay_buffer)
    if verbose:
        print(
            f"  [Phase 2] CombinedPotential  α={alpha:.2f}  "
            f"skel_scale={combined._skel_scale:.4f}  "
            f"emp_scale={combined._emp_scale:.4f}"
        )
    return combined


# ── Per-task shaping scale ────────────────────────────────────────────────────

def compute_per_task_shaping_scales(
    replay_buffer: ReplayBuffer,
    task_dist: "GymTaskDistribution",
    desired_shaping_fraction: float = 1.0,
    fallback_scale: float = 1.0,
) -> dict:
    """Compute a per-task shaping_scale from replay buffer env rewards.

    shaping_scale_i = median(|env_reward_i|) * desired_shaping_fraction

    This equalises the shaping signal magnitude across tasks whose raw env
    rewards differ by orders of magnitude (e.g. HumanoidStandup ~50 000/ep
    vs Humanoid ~100/ep).  The scale is recomputed each time the replay
    buffer grows so it tracks the current policy's reward distribution.

    Returns {task_id: float}.  Tasks with no buffer data use fallback_scale.
    """
    rewards_by_task: dict = {}
    for r, tid in zip(replay_buffer._rewards, replay_buffer._task_ids):
        rewards_by_task.setdefault(int(tid), []).append(float(r))

    scales: dict = {}
    for task in task_dist.tasks:
        rs = rewards_by_task.get(task.id, [])
        if rs:
            median_abs = float(np.percentile(np.abs(rs), 50))
            scales[task.id] = max(median_abs * desired_shaping_fraction, 1e-3)
        else:
            scales[task.id] = fallback_scale
    return scales


def calibrate_scales_by_phi_range(
    potential,
    rb: ReplayBuffer,
    task_dist: "GymTaskDistribution",
    base_scales: dict,
    n_sample: int = 300,
) -> dict:
    """Refine per-task shaping scales using the actual Φ range of the potential.

    base_scales (from compute_per_task_shaping_scales) set the target shaping
    magnitude in env-reward units.  But Φ values from SkeletonPotential /
    EmpiricalHittingTimePotential vary by orders of magnitude depending on the
    skeleton topology, so the same shaping_scale * ΔΦ can be tiny for one task
    and enormous for another.

    This function samples `n_sample` states per task from rb, evaluates
    potential.get_potential(s, sg_id) across all subgoals, and rescales so
    that the maximum possible shaped bonus (shaping_scale * Φ_range) equals
    base_scales[task_id] regardless of Φ topology.

        calibrated_scale_i = base_scale_i / max(Φ_max_i − Φ_min_i, 1e-4)

    Falls back to base_scales when the potential is None or states are missing.
    """
    if potential is None or not hasattr(potential, "get_potential"):
        return base_scales
    sg_ids = list(potential.subgoals.keys()) if potential.subgoals else []
    if not sg_ids:
        return base_scales

    states_by_task: dict = {}
    for i in range(len(rb)):
        tid    = int(rb._task_ids[i])
        bucket = states_by_task.setdefault(tid, [])
        if len(bucket) < n_sample:
            bucket.append(rb._states[i])

    calibrated: dict = {}
    for task in task_dist.tasks:
        base   = base_scales.get(task.id, 1.0)
        states = states_by_task.get(task.id, [])
        if not states:
            calibrated[task.id] = base
            continue

        phi_vals: list = []
        for s in states[:n_sample]:
            for sg_id in sg_ids:
                try:
                    phi_vals.append(potential.get_potential(s, sg_id))
                except Exception:
                    pass

        if len(phi_vals) < 2:
            calibrated[task.id] = base
            continue

        phi_range = float(np.max(phi_vals) - np.min(phi_vals))
        phi_range = max(phi_range, 1e-4)
        calibrated[task.id] = base / phi_range

    return calibrated


# ── Phase 3: task-policy training ─────────────────────────────────────────────

def phase3_train_task_policies(
    task_dist: GymTaskDistribution,
    potential,
    replay_buffer: ReplayBuffer,
    timesteps_per_task: int = 10_000,
    shaping_scales: dict = None,
    device: str = "cpu",
    verbose: bool = True,
    save_dir: str = None,
    iteration: int = 0,
    existing_task_policies: dict = None,
) -> tuple:
    """Train one SAC policy per task with potential-shaped rewards.

    shaping_scales: {task_id: float} — per-task scale factor applied to the
    potential-shaping term.  Use compute_per_task_shaping_scales() to derive
    these from the replay buffer so reward magnitudes are equalised across tasks.

    Returns (task_policies, phase3_stats).
    """
    shaping_scales = shaping_scales or {}
    task_policies: dict = {}
    phase3_stats: dict = {}

    for task in task_dist.tasks:
        scale = shaping_scales.get(task.id, 1.0)
        if verbose:
            print(
                f"  [Phase 3] {task.env_name} (task {task.id})  "
                f"steps={timesteps_per_task}  shaping_scale={scale:.4g}"
            )

        def _make_env(t=task, pot=potential, scale=scale):
            base = t.create_env()
            return ShapedRewardWrapper(base, pot, shaping_scale=scale) if pot is not None else base

        cb = Phase3TrainingCallback()
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


# ── Collect helpers ────────────────────────────────────────────────────────────

def _collect_transitions(
    model: SAC,
    task: GymTask,
    replay_buffer: ReplayBuffer,
    n_steps: int = 5_000,
    deterministic: bool = False,
) -> None:
    """Roll out model on task's raw env and push transitions."""
    env = task.create_env()
    obs, _ = env.reset()
    for _ in range(n_steps):
        action, _ = model.predict(obs, deterministic=deterministic)
        obs_next, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        replay_buffer.push(
            obs, action, float(reward), obs_next, done, task.id,
            terminated=terminated,
        )
        obs = obs_next if not done else env.reset()[0]
    env.close()


def collect_with_meta_policy(
    meta_policy,
    task_dist: GymTaskDistribution,
    replay_buffer: ReplayBuffer,
    num_episodes: int = 20,
    max_steps: int = 1000,
    device: str = "cpu",
) -> None:
    """Roll out the trained meta-policy and push transitions into replay_buffer."""
    meta_policy.eval()
    for _ in range(num_episodes):
        task = task_dist.sample()
        env  = task.create_env()
        obs, _ = env.reset()
        h    = meta_policy.init_hidden(device)
        done = False
        t    = 0

        while not done and t < max_steps:
            s_arr = np.asarray(obs, dtype=np.float32)
            with torch.no_grad():
                a_dist = meta_policy.forward_with_hidden(s_arr, h)
                a      = a_dist.sample()
            a_np = a.cpu().numpy().flatten()

            obs_next, r_env, terminated, truncated, _ = env.step(a_np)
            done = terminated or truncated

            replay_buffer.push(
                s_arr, a_np, float(r_env),
                np.asarray(obs_next, dtype=np.float32),
                done, task.id, terminated=terminated,
            )
            h   = meta_policy.update_hidden(s_arr, a_np, float(r_env), h)
            obs = obs_next
            t  += 1

        env.close()
    meta_policy.train()


# ── Subgoal diagnostics ────────────────────────────────────────────────────────

def _subgoal_diagnostic(skeleton: dict, iteration: int) -> int:
    """Print a breakdown of subgoal key population and return the count.

    Both keys should always be in sync (critical_states is derived from
    meta_subgoals inside morse.py), but we surface both explicitly so that a
    key-name mismatch is immediately visible rather than silently causing zero
    shaping.
    """
    ms  = skeleton.get("meta_subgoals",  {}) or {}
    cs  = skeleton.get("critical_states", {}) or {}
    n_ms = len(ms)
    n_cs = len(cs)

    mismatch = abs(n_ms - n_cs)
    print(
        f"  [Diag iter {iteration}] subgoals:"
        f"  meta_subgoals={n_ms}  critical_states={n_cs}"
        + (f"  *** KEY MISMATCH ({mismatch}) ***" if mismatch else "")
    )

    if n_ms == 0 and n_cs == 0:
        print(
            f"  [Diag iter {iteration}] ZERO subgoals — "
            "meta-policy Phase 4 will receive no topology shaping signal."
        )
    elif n_ms == 0:
        # critical_states populated but meta_subgoals missing — shouldn't happen
        # with current morse.py but guard against it.
        print(
            f"  [Diag iter {iteration}] WARNING: critical_states has {n_cs} entries "
            "but meta_subgoals is empty. Shaping will use critical_states fallback only."
        )

    return max(n_ms, n_cs)


# ── Evaluation ─────────────────────────────────────────────────────────────────

def evaluate_meta_policy(
    meta_policy,
    task_dist: GymTaskDistribution,
    n_episodes_per_task: int = 5,
    max_steps: int = 1000,
    device: str = "cpu",
    verbose: bool = True,
) -> dict:
    """Run n_episodes_per_task episodes per task; return {task_id: mean_return}."""
    meta_policy.eval()
    per_task: dict = {}

    for task in task_dist.tasks:
        returns = []
        for _ in range(n_episodes_per_task):
            env = task.create_env()
            obs, _ = env.reset()
            h        = meta_policy.init_hidden(device)
            ep_ret   = 0.0
            done     = False
            t        = 0

            with torch.no_grad():
                while not done and t < max_steps:
                    s_arr  = np.asarray(obs, dtype=np.float32)
                    a_dist = meta_policy.forward_with_hidden(s_arr, h)
                    a      = a_dist.sample()
                    a_np   = a.cpu().numpy().flatten()

                    obs_next, r_env, terminated, truncated, _ = env.step(a_np)
                    done    = terminated or truncated
                    ep_ret += float(r_env)
                    h = meta_policy.update_hidden(s_arr, a_np, float(r_env), h)
                    obs = obs_next
                    t  += 1

            env.close()
            returns.append(ep_ret)

        per_task[task.id] = float(np.mean(returns))
        if verbose:
            print(
                f"  [Eval] {task.env_name} (task {task.id}): "
                f"mean_return={per_task[task.id]:.2f}  "
                f"(n={n_episodes_per_task})"
            )

    meta_policy.train()
    return per_task


# ── Main loop ──────────────────────────────────────────────────────────────────

def main_meta_rl_loop(
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
    dbscan_eps: float = None,
    max_pool_size: int = None,
    min_task_support: float = 0.4,
    eval_episodes_per_task: int = 5,
    use_progress: bool = False,
    progress_latent_dim: int = 128,
    progress_epochs: int = 20,
    progress_weight: float = 0.5,
    save_dir: str = "results/mujoco_meta",
    device: str = "cpu",
    verbose: bool = True,
):
    """Full meta-RL pipeline over Gymnasium MuJoCo tasks."""
    os.makedirs(save_dir, exist_ok=True)

    _alpha_sched = AlphaScheduler(
        alpha_start=potential_alpha, alpha_end=0.0, anneal_iters=alpha_anneal_iters
    )

    if verbose:
        env_list = [t.env_name for t in task_dist.tasks]
        print("=" * 60)
        print("Meta-RL pipeline  (MuJoCo Gymnasium)")
        print(f"  tasks ({len(env_list)}): {env_list}")
        print(f"  state_dim={state_dim}  action_dim={action_dim}")
        print(f"  landmarks={num_landmarks}  iterations={num_iterations}")
        print(f"  potential_alpha={potential_alpha}  shaping_scale={shaping_scale}")
        print(f"  alpha_schedule: {_alpha_sched}")
        print(f"  save_dir={save_dir}")
        print("=" * 60)

    metrics = {
        "skeleton_train_losses": [],
        "phase3_env_returns":    [],   # list of {env_name: mean_env_return} per iter
        "phase4_losses":         [],
        "eval_returns":          [],   # list of {task_id: mean_return} per iter
    }

    tracker = BestModelTracker(save_dir, higher_is_better=True)

    # ── Phase 0 ───────────────────────────────────────────────────────────────
    rb_path    = os.path.join(save_dir, "replay_buffer.npz")
    model_path = os.path.join(save_dir, "phase0_models.json")   # manifest only

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
        # Persist model paths so they can be located if needed
        model_paths = {}
        for tid, model in task_policies.items():
            mp = os.path.join(save_dir, f"phase0_task{tid}.zip")
            model.save(mp)
            model_paths[str(tid)] = mp
        with open(model_path, "w") as f:
            json.dump(model_paths, f)

    # ── Phase 1 (initial skeleton) ────────────────────────────────────────────
    if verbose:
        print("\n[Phase 1] Building cross-task Morse skeleton...")
    skeleton = phase1_build_skeleton(
        rb, state_dim, action_dim,
        num_landmarks=num_landmarks,
        min_task_support=min_task_support,
        gamma=gamma,
        dbscan_eps=dbscan_eps,
        max_pool_size=max_pool_size,
        device=device,
        verbose=verbose,
    )
    metrics["skeleton_train_losses"].append(skeleton.get("train_losses", []))
    n_sub = len(skeleton.get("meta_subgoals") or skeleton.get("critical_states", {}))
    if verbose:
        print(f"  Found {n_sub} meta-subgoal(s).")
    skeleton["_potential_stale"] = True
    plot_skeleton_topology(skeleton, rb, os.path.join(save_dir, "topology_initial.png"))

    if n_sub == 0:
        print("No meta-subgoals found — aborting.")
        plot_training_curves(metrics, save_dir)
        return None, None, skeleton, metrics

    meta_policy   = MetaPolicy(
        state_dim, action_dim, discrete=False,
        action_low=action_low, action_high=action_high,
    ).to(device)
    training_state = None

    for iteration in range(num_iterations):
        if verbose:
            print(f"\n{'─' * 60}")
            print(
                f"Iteration {iteration + 1}/{num_iterations}  "
                f"buffer={len(rb)} transitions"
            )
            print(f"{'─' * 60}")

        # ── Per-task shaping scales (recomputed each iteration) ───────────────
        # median(|r_env_i|) is drawn from the full replay buffer as it stands at
        # the start of this iteration — grows with each collect phase, so the
        # scale tracks the improving policy rather than being frozen at Phase-0
        # quality. desired_shaping_fraction=shaping_scale keeps the global knob:
        #   shaping_scale_i = median(|r_env_i|) * shaping_scale
        per_task_shaping_scales = compute_per_task_shaping_scales(
            rb, task_dist, desired_shaping_fraction=shaping_scale,
        )
        if verbose:
            scale_str = "  ".join(
                f"{task.env_name}={per_task_shaping_scales.get(task.id, 1.0):.3g}"
                for task in task_dist.tasks
            )
            print(f"  [shaping_scales iter={iteration + 1}] {scale_str}")

        # ── Phase 2 ───────────────────────────────────────────────────────────
        current_alpha = _alpha_sched.alpha(iteration)
        if verbose:
            print(
                f"\n[Phase 2] Building combined potential  "
                f"α={current_alpha:.3f}..."
            )
        potential = phase2_build_potential(
            skeleton, rb,
            alpha=current_alpha,
            gamma=gamma,
            verbose=verbose,
        )

        # ── Phase 2b: optional progress estimator ─────────────────────────────
        if use_progress and potential is not None:
            if verbose:
                print("\n[Phase 2b] Blending with progress estimator...")
            potential = blend_with_progress(
                potential, rb,
                state_dim=state_dim,
                latent_dim=progress_latent_dim,
                epochs=progress_epochs,
                base_weight=1.0 - progress_weight,
                progress_weight=progress_weight,
                device=device,
                verbose=verbose,
            )

        # ── Calibrate shaping scales by actual Φ range ────────────────────────
        # base_scales used median(|r_env|) as the target magnitude, but Φ values
        # vary with skeleton topology independently of env reward scale.
        # Rescale so that max_possible(shaping_scale * ΔΦ) ≈ base_scale per task.
        if potential is not None:
            per_task_shaping_scales = calibrate_scales_by_phi_range(
                potential, rb, task_dist, per_task_shaping_scales,
            )
            if verbose:
                cal_str = "  ".join(
                    f"{task.env_name}={per_task_shaping_scales.get(task.id, 1.0):.3g}"
                    for task in task_dist.tasks
                )
                print(f"  [shaping_scales calibrated iter={iteration + 1}] {cal_str}")

        # ── Phase 3 ───────────────────────────────────────────────────────────
        if verbose:
            print("\n[Phase 3] Training task policies with shaped rewards...")
        task_policies, p3_stats = phase3_train_task_policies(
            task_dist, potential, rb,
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

        # ── Phase 4 pre-check: verify subgoals exist ──────────────────────────
        n_subgoals = _subgoal_diagnostic(skeleton, iteration + 1)

        if n_subgoals == 0:
            # Cross-task intersection (min_task_support) was too strict.  Fall
            # back to per-task subgoals (min_task_support=0.0) so Phase 4 has
            # at least some topology signal in the GRU reward channel.
            print(
                f"\n  [Phase 4] Rebuilding skeleton with min_task_support=0.0 "
                f"(was {min_task_support}) to recover subgoals..."
            )
            try:
                skeleton = phase1_build_skeleton(
                    rb, state_dim, action_dim,
                    num_landmarks=num_landmarks,
                    min_task_support=0.0,
                    gamma=gamma,
                    dbscan_eps=dbscan_eps,
                    max_pool_size=max_pool_size,
                    device=device,
                    verbose=verbose,
                )
                skeleton["_potential_stale"] = True
                n_subgoals = _subgoal_diagnostic(skeleton, iteration + 1)
                if n_subgoals > 0:
                    # Rebuild potential on the new skeleton before Phase 4.
                    potential = phase2_build_potential(
                        skeleton, rb,
                        alpha=current_alpha,
                        gamma=gamma,
                        verbose=verbose,
                    )
                else:
                    print(
                        "  [Phase 4] Still zero subgoals after min_task_support=0.0 "
                        "— Phase 4 will run with env reward only (no shaping)."
                    )
            except Exception as exc:
                print(f"  [Phase 4] Fallback skeleton rebuild failed ({exc}); "
                      "continuing without shaping.")

        # ── Phase 4: meta-policy gradient ─────────────────────────────────────
        if verbose:
            print(f"\n[Phase 4] Training meta-policy via policy gradient "
                  f"(n_subgoals={n_subgoals})...")
        training_skeleton = dict(skeleton)
        training_skeleton["skeleton_potential"] = potential

        meta_policy, meta_value_net, p4_losses, training_state = (
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
            meta_value_net=meta_value_net,
            task_policies=task_policies,
            skeleton_data=skeleton,
            replay_buffer=rb,
            metrics={
                "mean_return": mean_return,
                "per_task_returns": per_task_returns,
                "p4_avg_loss": (
                    float(np.mean([e["total"] for e in p4_losses]))
                    if p4_losses else 0.0
                ),
            },
        )
        improved = tracker.update(mean_return, ckpt_dir)
        if verbose and improved:
            print(f"  ★ New best model (mean_return={mean_return:.2f})")

        save_iteration_visuals(skeleton, rb, metrics, save_dir, iteration)
        save_replay_buffer(rb, rb_path)

        # ── Collect with meta-policy ───────────────────────────────────────────
        if verbose:
            print("\n[Collect] Rolling out meta-policy for additional data...")
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
                print("\n[Refine] Rebuilding skeleton on enlarged buffer...")
            skeleton = refine_skeleton(
                skeleton, rb,
                num_landmarks=num_landmarks,
                dbscan_eps=dbscan_eps,
                max_pool_size=max_pool_size,
                min_task_support=min_task_support,
                device=device,
                verbose=verbose,
            )
            skeleton["_potential_stale"] = True
            metrics["skeleton_train_losses"].append(skeleton.get("train_losses", []))
            n_sub = len(skeleton.get("meta_subgoals") or skeleton.get("critical_states", {}))
            if verbose:
                print(f"  Refined skeleton: {n_sub} meta-subgoal(s).")
            topo_path = os.path.join(
                save_dir, f"topology_iter{iteration + 1:03d}.png"
            )
            plot_skeleton_topology(skeleton, rb, topo_path)
            if n_sub == 0:
                print("  No subgoals after refinement — stopping.")
                break

    plot_training_curves(metrics, save_dir)
    if verbose:
        print("\nMeta-RL pipeline complete.")

    return meta_policy, task_policies, skeleton, metrics


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Meta-RL over MuJoCo Gymnasium environments."
    )
    parser.add_argument(
        "--task-set",
        default="walker2d",
        choices=["walker2d"],
        help="Predefined task set (default: walker2d — three Walker2d-v5 "
             "variants: baseline, heavy torso, slippery floor). "
             "Ignored when --envs is provided.",
    )
    parser.add_argument(
        "--envs", nargs="+",
        default=None,
        metavar="ENV",
        help="Custom Gymnasium env IDs (must share obs/action dims). "
             "Overrides --task-set. Plain GymTask (no param modifications).",
    )
    parser.add_argument("--iterations",         type=int,   default=5)
    parser.add_argument("--landmarks",          type=int,   default=200)
    parser.add_argument("--meta-epochs",        type=int,   default=300)
    parser.add_argument("--phase0-steps",       type=int,   default=30_000,
                        help="SAC bootstrap steps per task (default: 30000)")
    parser.add_argument("--task-steps",         type=int,   default=10_000,
                        help="Phase 3 SAC steps per task per iteration (default: 10000)")
    parser.add_argument("--n-envs",             type=int,   default=4,
                        help="Parallel envs for Phase 0 (default: 4)")
    parser.add_argument("--collect-episodes",   type=int,   default=20,
                        help="Meta-policy rollout episodes per iteration (default: 20)")
    parser.add_argument("--shaping-scale",      type=float, default=1.0)
    parser.add_argument("--subgoal-threshold",  type=float, default=float("inf"))
    parser.add_argument("--potential-alpha",    type=float, default=0.5,
                        help="α: 1=pure topology, 0=pure empirical (default: 0.5)")
    parser.add_argument("--alpha-anneal-iters", type=int,   default=3,
                        help="Iterations to anneal α→0 (default: 3)")
    parser.add_argument("--gamma",              type=float, default=0.99)
    parser.add_argument("--refine-every",       type=int,   default=2,
                        help="Rebuild skeleton every N iterations (default: 2)")
    parser.add_argument("--min-task-support",   type=float, default=0.4,
                        help="Min fraction of tasks a subgoal must appear in (default: 0.4)")
    parser.add_argument("--eval-episodes",      type=int,   default=5,
                        help="Evaluation episodes per task (default: 5)")
    parser.add_argument("--dbscan-eps",         type=float, default=None,
                        help="DBSCAN eps for meta-subgoal clustering (default: auto)")
    parser.add_argument("--max-pool-size",      type=int,   default=None,
                        help="Max states passed to FPS (default: landmarks × 20)")
    parser.add_argument("--use-progress",       action="store_true",
                        help="Blend Phase-2 potential with progress estimator")
    parser.add_argument("--progress-latent-dim",type=int,   default=128)
    parser.add_argument("--progress-epochs",    type=int,   default=20)
    parser.add_argument("--progress-weight",    type=float, default=0.5)
    parser.add_argument("--save-dir",           default="results/mujoco_meta")
    parser.add_argument("--device",             default="cpu")
    parser.add_argument("--no-verbose",         action="store_true")
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

    main_meta_rl_loop(
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
        dbscan_eps=args.dbscan_eps,
        max_pool_size=args.max_pool_size,
        min_task_support=args.min_task_support,
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
