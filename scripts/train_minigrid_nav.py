"""
Meta-RL pipeline for MiniGrid navigation tasks.

Four tasks on the FourRooms 19×19 grid; each task fixes the goal in one
quadrant.  The shared topological bottleneck is the doorway crossing at the
grid centre — the skeleton should identify it as the critical state.

State  : (agent_x/19, agent_y/19, agent_dir/3, goal_x/19, goal_y/19)
         — float32, dim=5
Actions: 3 discrete  (0=left, 1=right, 2=forward; sufficient for navigation)
Tasks  : NW  NE  SW  SE — goal in each of the four rooms

Run as:
    python3 scripts/train_minigrid_nav.py --iterations 5 --landmarks 64
    python3 scripts/train_minigrid_nav.py --demo-only --load results/minigrid_nav/best
"""

import argparse
import os
import sys
import json

import numpy as np
import torch
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from utils.replay_buffer import ReplayBuffer
from utils.checkpoint    import (save_checkpoint, load_checkpoint,
                                  restore_models, BestModelTracker,
                                  save_replay_buffer, load_replay_buffer)
from utils.viz           import (plot_training_curves, plot_skeleton_topology,
                                  save_iteration_visuals,
                                  Phase3TrainingCallback, plot_phase3_results,
                                  _P, _c, _save_fig)
from utils.skeleton      import build_skeleton, refine_skeleton
from models.meta_policy_net import MetaPolicy
from algos.MetaPolicy import (
    phase1_build_skeleton, phase2_build_potential, ShapedRewardWrapper,
)
from algos.meta_policy_gradient import meta_policy_gradient_with_skeleton_shaping

try:
    from minigrid.envs import FourRoomsEnv as _MG_FourRooms
    from minigrid.core.world_object import Goal as _MG_Goal
    _MINIGRID_OK = True
except ImportError:
    _MINIGRID_OK = False

STATE_DIM  = 5
ACTION_DIM = 3     # left, right, forward
GRID_SIZE  = 19    # MiniGrid FourRooms default

# Fixed goal cell in each quadrant (clear of walls and doorways)
TASK_GOALS = {
    "NW": (4,  4),
    "NE": (14, 4),
    "SW": (4,  14),
    "SE": (14, 14),
}
TASK_IDS = {"NW": 0, "NE": 1, "SW": 2, "SE": 3}


# ── Wrapper ────────────────────────────────────────────────────────────────

class MiniGridNavWrapper(gym.Wrapper):
    """
    Wraps MiniGrid FourRooms into the pipeline interface:
      - Compact 5-dim float32 observation
      - Restricted to 3 actions (left / right / forward)
      - Dense reward: +10 on goal reached, −0.01/step
      - info["success"] = 1.0 on termination
    """

    def __init__(self, env: gym.Env, grid_size: int = GRID_SIZE):
        super().__init__(env)
        self._gs   = float(grid_size)
        self._goal = np.zeros(2, dtype=np.float32)
        self.observation_space = spaces.Box(
            0.0, 1.0, shape=(STATE_DIM,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(ACTION_DIM)

    def reset(self, **kwargs):
        _, info = self.env.reset(**kwargs)
        self._goal = self._locate_goal()
        return self._obs(), info

    def step(self, action):
        _, r, terminated, truncated, info = self.env.step(int(action))
        success = bool(terminated) and (r > 0.0)
        r_out   = 10.0 if success else -0.01
        info["success"] = float(success)
        return self._obs(), r_out, terminated, truncated, info

    # ── helpers ───────────────────────────────────────────────────────────

    def _obs(self) -> np.ndarray:
        ax, ay = self.unwrapped.agent_pos
        ad     = float(self.unwrapped.agent_dir)
        gx, gy = self._goal
        return np.array(
            [ax / self._gs, ay / self._gs, ad / 3.0,
             gx / self._gs, gy / self._gs],
            dtype=np.float32,
        )

    def _locate_goal(self) -> np.ndarray:
        grid = self.unwrapped.grid
        for y in range(self.unwrapped.height):
            for x in range(self.unwrapped.width):
                cell = grid.get(x, y)
                if isinstance(cell, _MG_Goal):
                    return np.array([float(x), float(y)], dtype=np.float32)
        return np.zeros(2, dtype=np.float32)


# ── Task / Distribution ────────────────────────────────────────────────────

class MiniGridNavTask:
    def __init__(self, task_key: str):
        if not _MINIGRID_OK:
            raise RuntimeError("minigrid not installed.  Run: pip install minigrid")
        self.id       = TASK_IDS[task_key]
        self.env_name = f"MiniGridNav-{task_key}"
        self._key     = task_key
        self._goal    = TASK_GOALS[task_key]

    def create_env(self, max_steps: int = 500) -> gym.Env:
        base = _MG_FourRooms(goal_pos=self._goal, max_steps=max_steps)
        return MiniGridNavWrapper(base)


class MiniGridNavTaskDistribution:
    def __init__(self, task_keys=None):
        keys       = task_keys or list(TASK_GOALS.keys())
        self.tasks = [MiniGridNavTask(k) for k in keys]

    def sample(self) -> MiniGridNavTask:
        return self.tasks[np.random.randint(len(self.tasks))]


# ── Phase 0 ────────────────────────────────────────────────────────────────

def phase0_collect(
    task_distribution: MiniGridNavTaskDistribution,
    replay_buffer: ReplayBuffer,
    timesteps_per_task: int = 20_000,
    n_envs: int             = 10,
    device: str             = "cpu",
    verbose: bool           = True,
) -> dict:
    """Train one PPO per task, collect into replay_buffer, return {task_id: model}."""
    task_policies: dict = {}
    for task in task_distribution.tasks:
        if verbose:
            print(f"  [Phase 0] {task.env_name}  PPO × {n_envs} envs  "
                  f"{timesteps_per_task} steps...")

        def _make(t=task):
            return t.create_env()

        vec_env = make_vec_env(_make, n_envs=n_envs)
        model   = PPO("MlpPolicy", vec_env, verbose=0, device=device)
        model.learn(total_timesteps=timesteps_per_task)
        task_policies[task.id] = model

        obs           = vec_env.reset()
        steps_per_env = max(1, timesteps_per_task // n_envs)
        for _ in range(steps_per_env):
            action, _ = model.predict(obs, deterministic=False)
            obs_next, reward, done, info = vec_env.step(action)
            for i in range(n_envs):
                trunc_i = info[i].get("TimeLimit.truncated", False)
                term_i  = bool(done[i]) and not trunc_i
                replay_buffer.push(
                    obs[i], action[i], float(reward[i]),
                    obs_next[i], bool(done[i]), task.id,
                    terminated=term_i,
                )
            obs = obs_next
        vec_env.close()

    if verbose:
        print(f"  [Phase 0] Buffer: {len(replay_buffer)} transitions")
    return task_policies


# ── Phase 3 ────────────────────────────────────────────────────────────────

def phase3_train_task_policies(
    skeleton_data: dict,
    task_distribution: MiniGridNavTaskDistribution,
    potential=None,
    timesteps_per_task: int      = 20_000,
    shaping_scale: float         = 1.0,
    device: str                  = "cpu",
    verbose: bool                = True,
    save_dir: str                = None,
    iteration: int               = 0,
    existing_task_policies: dict = None,
) -> tuple:
    task_policies: dict = {}
    phase3_stats:  dict = {}

    for task in task_distribution.tasks:
        if verbose:
            print(f"  [Phase 3] {task.env_name} (task {task.id})...")

        def _make_env(t=task, pot=potential, scale=shaping_scale):
            env = t.create_env()
            if pot is not None:
                env = ShapedRewardWrapper(env, pot, shaping_scale=scale)
            return env

        cb      = Phase3TrainingCallback()
        vec_env = make_vec_env(_make_env, n_envs=1)

        prior = (existing_task_policies or {}).get(task.id)
        if prior is not None:
            import tempfile
            with tempfile.TemporaryDirectory() as _td:
                prior.save(os.path.join(_td, "prior"))
                model = type(prior).load(os.path.join(_td, "prior"),
                                         env=vec_env, device=device)
            model.learn(total_timesteps=timesteps_per_task,
                        reset_num_timesteps=False, callback=cb)
        else:
            model = PPO("MlpPolicy", vec_env, verbose=0, device=device)
            model.learn(total_timesteps=timesteps_per_task, callback=cb)

        task_policies[task.id] = model
        vec_env.close()

        phase3_stats[task.env_name] = {
            "ep_rewards":     cb.ep_rewards,
            "ep_env_rewards": cb.ep_env_rewards,
            "ep_shaping":     cb.ep_shaping,
            "ep_successes":   cb.ep_successes,
            "ep_lengths":     cb.ep_lengths,
        }
        if verbose and cb.ep_rewards:
            sr = float(np.mean(cb.ep_successes[-20:])) if cb.ep_successes else 0.0
            print(f"    last-20-ep avg_r={np.mean(cb.ep_rewards[-20:]):.3f}  "
                  f"success={sr:.1%}")

    if save_dir is not None:
        plot_phase3_results(phase3_stats, save_dir, iteration=iteration)

    if verbose:
        print(f"  [Phase 3] Trained {len(task_policies)} policies.")
    return task_policies, phase3_stats


# ── Evaluation ─────────────────────────────────────────────────────────────

def evaluate_policy(
    meta_policy,
    task_distribution: MiniGridNavTaskDistribution,
    n_episodes: int = 20,
    max_steps: int  = 500,
    gamma: float    = 0.99,
    device: str     = "cpu",
) -> dict:
    meta_policy.eval()
    successes:   list = []
    returns:     list = []
    env_results: dict = {}

    for _ in range(n_episodes):
        task    = task_distribution.sample()
        env     = task.create_env()
        obs, _  = env.reset()
        obs_arr = np.asarray(obs, dtype=np.float32).flatten()
        tau     = []
        done    = False
        ep_ret  = 0.0
        t       = 0
        success = False

        while not done and t < max_steps:
            with torch.no_grad():
                a_dist = meta_policy(obs_arr, tau)
                a      = a_dist.sample()
            a_np  = a.cpu().numpy().flatten()
            a_env = int(a_np[0])
            s_next, r, terminated, truncated, info = env.step(a_env)
            done    = terminated or truncated
            success = success or bool(info.get("success", 0.0) > 0.5)
            ep_ret += (gamma ** t) * r
            tau.append((obs_arr, a_np, r))
            obs_arr = np.asarray(s_next, dtype=np.float32).flatten()
            t += 1

        env.close()
        successes.append(float(success))
        returns.append(ep_ret)
        env_results.setdefault(task.env_name, []).append(float(success))

    meta_policy.train()
    return {
        "success_rate": float(np.mean(successes)),
        "avg_return":   float(np.mean(returns)),
        "per_env":      {k: float(np.mean(v)) for k, v in env_results.items()},
    }


# ── Demo runner ────────────────────────────────────────────────────────────

def run_demos(
    meta_policy,
    skeleton_data: dict,
    task_distribution: MiniGridNavTaskDistribution,
    save_dir: str,
    n_demos: int   = 5,
    max_steps: int = 500,
    gamma: float   = 0.99,
    device: str    = "cpu",
) -> list:
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA

    os.makedirs(save_dir, exist_ok=True)
    c_states = [np.asarray(v, dtype=np.float32)
                for v in skeleton_data["critical_states"].values()]
    lm_np    = skeleton_data["landmarks"].cpu().numpy()
    D        = lm_np.shape[1]
    pca      = PCA(n_components=2).fit(lm_np) if D > 2 else None

    def _proj(s):
        arr = np.asarray(s, dtype=np.float32).flatten()
        return pca.transform(arr.reshape(1, -1))[0] if pca else arr[:2]

    lm_2d   = pca.transform(lm_np) if pca else lm_np[:, :2]
    crit_2d = (pca.transform(np.stack(c_states)) if (pca and c_states)
               else (np.stack(c_states)[:, :2] if c_states else np.empty((0, 2))))

    meta_policy.eval()
    results = []

    for demo_i in range(n_demos):
        task    = task_distribution.sample()
        env     = task.create_env()
        obs, _  = env.reset()
        obs_arr = np.asarray(obs, dtype=np.float32).flatten()
        tau     = []
        grid_pts = []   # (agent_x, agent_y) in grid coords
        pca_pts  = []
        ep_ret  = 0.0
        t       = 0
        done    = False
        success = False

        while not done and t < max_steps:
            ax = obs_arr[0] * GRID_SIZE
            ay = obs_arr[1] * GRID_SIZE
            grid_pts.append([ax, ay])
            pca_pts.append(_proj(obs_arr))

            with torch.no_grad():
                a_dist = meta_policy(obs_arr, tau)
                a      = a_dist.sample()
            a_np  = a.cpu().numpy().flatten()
            s_next, r, terminated, truncated, info = env.step(int(a_np[0]))
            done    = terminated or truncated
            success = success or bool(info.get("success", 0.0) > 0.5)
            ep_ret += (gamma ** t) * r
            tau.append((obs_arr, a_np, r))
            obs_arr = np.asarray(s_next, dtype=np.float32).flatten()
            t += 1

        env.close()
        grid_pts.append([obs_arr[0] * GRID_SIZE, obs_arr[1] * GRID_SIZE])
        pca_pts.append(_proj(obs_arr))

        results.append({
            "demo": demo_i, "task": task.env_name,
            "success": success, "return": ep_ret, "steps": t,
        })

        grid_arr = np.array(grid_pts)
        pca_arr  = np.array(pca_pts)
        status   = "success" if success else "fail"

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Left: grid view
        ax = axes[0]
        gs = GRID_SIZE
        c  = gs // 2  # wall centre
        ax.set_xlim(-0.5, gs - 0.5)
        ax.set_ylim(-0.5, gs - 0.5)
        ax.set_aspect("equal")
        # four-rooms wall outline
        for seg in [
            ([c - 0.5, c - 0.5], [-0.5, c - 0.5]),
            ([c - 0.5, c - 0.5], [c + 0.5, gs - 0.5]),
            ([c + 0.5, c + 0.5], [-0.5, c - 0.5]),
            ([c + 0.5, c + 0.5], [c + 0.5, gs - 0.5]),
        ]:
            ax.plot(seg[0], seg[1], "k-", lw=1.5)
        # goal marker
        gx = obs_arr[3] * GRID_SIZE
        gy = obs_arr[4] * GRID_SIZE
        ax.plot(gx, gy, marker="*", ms=14, color=_P["gold"], zorder=4, label="goal")
        for k in range(len(grid_arr) - 1):
            ax.plot(grid_arr[k:k+2, 0], grid_arr[k:k+2, 1],
                    color=_c(k), lw=1.0, alpha=0.7)
        ax.plot(*grid_arr[0],  marker="^", ms=9, color=_P["blue"], zorder=5, label="start")
        ax.plot(*grid_arr[-1], marker="*", ms=11,
                color=_P["gold"] if success else _P["red"],
                zorder=5, label="end (✓)" if success else "end (✗)")
        ax.legend(fontsize=7, loc="upper right")
        ax.axis("off")

        # Right: PCA skeleton
        ax = axes[1]
        ax.scatter(lm_2d[:, 0], lm_2d[:, 1], s=20, c=_P["mid"], alpha=0.5)
        if len(crit_2d):
            ax.scatter(crit_2d[:, 0], crit_2d[:, 1], s=150,
                       facecolors="none", edgecolors=_P["red"], lw=1.5)
        for k in range(len(pca_arr) - 1):
            ax.plot(pca_arr[k:k+2, 0], pca_arr[k:k+2, 1],
                    color=_P["blue"], lw=1.0, alpha=0.6)
        ax.plot(*pca_arr[0],  marker="^", ms=8, color=_P["blue"], zorder=5)
        ax.plot(*pca_arr[-1], marker="*", ms=10,
                color=_P["gold"] if success else _P["red"], zorder=5)
        ax.axis("off")

        fig.tight_layout()
        out = os.path.join(save_dir,
                           f"demo_{demo_i:02d}_{task.env_name}_{status}.png")
        _save_fig(fig, out)
        print(f"  [Demo {demo_i}] {task.env_name}  {status}  "
              f"return={ep_ret:.3f}  steps={t}")

    meta_policy.train()
    with open(os.path.join(save_dir, "demo_summary.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"  [Demo] success_rate={np.mean([r['success'] for r in results]):.1%}")
    return results


# ── Main pipeline ──────────────────────────────────────────────────────────

def main(
    task_distribution: MiniGridNavTaskDistribution = None,
    num_landmarks: int       = 64,
    num_iterations: int      = 5,
    refine_every: int        = 2,
    timesteps_per_task: int  = 20_000,
    n_envs: int              = 10,
    task_steps: int          = 20_000,
    collect_episodes: int    = 50,
    gamma: float             = 0.97,
    shaping_scale: float     = 1.0,
    shaping_scale_start: float = 0.1,
    subgoal_threshold: float = float("inf"),
    potential_alpha: float   = 0.5,
    meta_epochs: int         = 1000,
    episodes_per_update: int = 4,
    entropy_coef: float      = 0.01,
    is_buffer_size: int      = 16,
    is_clip_epsilon: float   = 0.2,
    eval_episodes: int       = 20,
    n_demos: int             = 5,
    save_dir: str            = "results/minigrid_nav",
    device: str              = "cpu",
    verbose: bool            = True,
):
    if not _MINIGRID_OK:
        raise RuntimeError("minigrid not installed.  Run: pip install minigrid")
    if task_distribution is None:
        task_distribution = MiniGridNavTaskDistribution()

    os.makedirs(save_dir, exist_ok=True)

    if verbose:
        print("=" * 60)
        print("Meta-RL pipeline  (MiniGrid FourRooms Navigation)")
        print(f"  tasks: {[t.env_name for t in task_distribution.tasks]}")
        print(f"  landmarks: {num_landmarks}  iterations: {num_iterations}")
        print(f"  save_dir: {save_dir}")
        print("=" * 60)

    metrics = {
        "skeleton_train_losses": [],
        "phase3_success_rates":  [],
        "phase4_losses":         [],
        "eval_success_rates":    [],
        "eval_returns":          [],
    }
    tracker = BestModelTracker(save_dir, higher_is_better=True)

    # Phase 0
    rb_path = os.path.join(save_dir, "replay_buffer.npz")
    if os.path.exists(rb_path):
        if verbose:
            print("\n[Phase 0] Loading existing replay buffer...")
        rb = load_replay_buffer(rb_path, device=device)
        task_policies = {}   # no Phase-0 models available; Phase 3 will create fresh
    else:
        rb = ReplayBuffer(device=device)
        if verbose:
            print("\n[Phase 0] Collecting initial data with PPO...")
        task_policies = phase0_collect(task_distribution, rb,
                                       timesteps_per_task=timesteps_per_task,
                                       n_envs=n_envs,
                                       device=device, verbose=verbose)
        save_replay_buffer(rb, rb_path)
    if verbose:
        print(f"  Buffer: {len(rb)} transitions")

    # Phase 1
    if verbose:
        print("\n[Phase 1] Building Morse skeleton...")
    # Goal coordinates (s[3:5]) dominate inter-task distances, splitting the
    # complex into disconnected manifolds.  Use only agent (x, y) for spectral
    # landmark selection and witness-complex distances; full state is used for KNN phi estimates.
    _pos_proj = lambda s: s[:2]  # noqa: E731

    skeleton = phase1_build_skeleton(rb,
                                     state_dim=STATE_DIM, action_dim=ACTION_DIM,
                                     num_landmarks=num_landmarks,
                                     state_projection_fn=_pos_proj,
                                     sa_epochs=0,
                                     min_task_support=0.4,
                                     device=device, verbose=verbose)
    skeleton["_potential_stale"] = True
    metrics["skeleton_train_losses"].append(skeleton.get("train_losses", []))
    n_sub = len(skeleton["critical_states"])
    if verbose:
        print(f"  Found {n_sub} critical state(s).")
        knn_est = skeleton.get("knn_estimator")
        if knn_est is not None:
            for tid, s in knn_est.back_ret_stats().items():
                flag = "  ← FLAT phi" if s["std"] < 0.01 else ""
                print(f"  [KNN] task {tid}: n={s['n']:5d}  "
                      f"mean={s['mean']:+.4f}  std={s['std']:.4f}{flag}")

    plot_skeleton_topology(skeleton, rb,
                           os.path.join(save_dir, "topology_initial.png"))

    if n_sub == 0:
        print("No subgoals found; aborting.")
        plot_training_curves(metrics, save_dir)
        return None, None, skeleton, metrics

    meta_policy    = MetaPolicy(STATE_DIM, ACTION_DIM, discrete=True, gru_hidden=64).to(device)
    meta_value_net = None
    training_state = None

    for iteration in range(num_iterations):
        if verbose:
            print(f"\n{'─'*60}")
            print(f"Iteration {iteration + 1}/{num_iterations}")
            print(f"{'─'*60}")

        # Cross-iteration shaping annealing: ramp from shaping_scale_start to
        # shaping_scale over the course of all iterations.  This prevents the
        # large distribution jump at iteration boundaries that occurs when the
        # potential is rebuilt with a full-amplitude shaped reward from epoch 0.
        if num_iterations > 1:
            t = iteration / (num_iterations - 1)
        else:
            t = 1.0
        iter_shaping_scale = shaping_scale_start + (shaping_scale - shaping_scale_start) * t
        if verbose:
            print(f"  shaping_scale for this iteration: {iter_shaping_scale:.3f}")

        # Phase 2
        if verbose:
            print(f"[Phase 2] Combined potential (α={potential_alpha:.2f})...")
        potential = phase2_build_potential(
            skeleton, replay_buffer=rb,
            alpha=potential_alpha, gamma=gamma, verbose=verbose,
        )

        # Phase 3
        if verbose:
            print("[Phase 3] Training task policies (continuing from previous)...")
        task_policies, p3_stats = phase3_train_task_policies(
            skeleton, task_distribution,
            potential=potential,
            timesteps_per_task=task_steps,
            shaping_scale=iter_shaping_scale,
            device=device, verbose=verbose,
            save_dir=save_dir, iteration=iteration,
            existing_task_policies=task_policies,
        )
        p3_sr = float(np.mean([np.mean(v["ep_successes"][-20:])
                                for v in p3_stats.values()
                                if v["ep_successes"]]) if p3_stats else 0.0)
        metrics["phase3_success_rates"].append(p3_sr)

        # Phase 4
        if verbose:
            print("[Phase 4] Training meta-policy...")
        training_skeleton = dict(skeleton)
        training_skeleton["skeleton_potential"] = potential
        meta_policy, meta_value_net, p4_losses, training_state = \
            meta_policy_gradient_with_skeleton_shaping(
                meta_policy, task_distribution, training_skeleton,
                meta_epochs=meta_epochs,
                episodes_per_update=episodes_per_update,
                entropy_coef=entropy_coef,
                is_buffer_size=is_buffer_size,
                is_clip_epsilon=is_clip_epsilon,
                replay_buffer=rb,
                gamma=gamma,
                shaping_scale=iter_shaping_scale,
                subgoal_threshold=subgoal_threshold,
                flush_buffer=True,
                device=device, verbose=verbose,
                training_state=training_state,
            )
        metrics["phase4_losses"].append(p4_losses)

        # Evaluate
        eval_result = evaluate_policy(
            meta_policy, task_distribution,
            n_episodes=eval_episodes, gamma=gamma, device=device,
        )
        metrics["eval_success_rates"].append(eval_result["success_rate"])
        metrics["eval_returns"].append(eval_result["avg_return"])
        if verbose:
            print(f"  [Eval] success_rate={eval_result['success_rate']:.1%}  "
                  f"avg_return={eval_result['avg_return']:.4f}")
            for env_name, sr in eval_result["per_env"].items():
                print(f"    {env_name}: {sr:.1%}")

        # Checkpoint
        ckpt_dir = save_checkpoint(
            save_dir, iteration=iteration,
            meta_policy=meta_policy, meta_value_net=meta_value_net,
            task_policies=task_policies,
            skeleton_data=skeleton, replay_buffer=rb,
            metrics={"eval": eval_result,
                     "p4_avg_loss": float(np.mean([e["total"] for e in p4_losses])) if p4_losses else 0.0},
        )
        if tracker.update(eval_result["success_rate"], ckpt_dir) and verbose:
            print(f"  ★ New best (success_rate={eval_result['success_rate']:.1%})")

        save_iteration_visuals(skeleton, rb, metrics, save_dir, iteration)

        # Collect more data
        meta_policy.eval()
        for _ in range(collect_episodes):
            task    = task_distribution.sample()
            env     = task.create_env()
            obs, _  = env.reset()
            obs_arr = np.asarray(obs, dtype=np.float32).flatten()
            tau     = []
            done    = False
            while not done:
                with torch.no_grad():
                    a_dist = meta_policy(obs_arr, tau)
                    a      = a_dist.sample()
                a_np = a.cpu().numpy().flatten()
                s_next, r, terminated, truncated, _ = env.step(int(a_np[0]))
                done         = terminated or truncated
                obs_next_arr = np.asarray(s_next, dtype=np.float32).flatten()
                rb.push(obs_arr, a_np, r, obs_next_arr, done, task.id,
                        terminated=terminated)
                tau.append((obs_arr, a_np, r))
                obs_arr = obs_next_arr
            env.close()
        meta_policy.train()

        save_replay_buffer(rb, rb_path)
        if verbose:
            print(f"  Buffer: {len(rb)} transitions")

        # Periodic refinement
        if (iteration + 1) % refine_every == 0 and iteration < num_iterations - 1:
            if verbose:
                print("[Refine] Rebuilding skeleton...")
            skeleton = refine_skeleton(skeleton, rb,
                                       num_landmarks=num_landmarks,
                                       state_projection_fn=_pos_proj,
                                       sa_epochs=0,
                                       min_task_support=0.4,
                                       device=device, verbose=verbose)
            skeleton["_potential_stale"] = True
            metrics["skeleton_train_losses"].append(skeleton.get("train_losses", []))
            n_sub = len(skeleton["critical_states"])
            if verbose:
                print(f"  Refined skeleton: {n_sub} critical state(s).")
                knn_est = skeleton.get("knn_estimator")
                if knn_est is not None:
                    for tid, s in knn_est.back_ret_stats().items():
                        flag = "  ← FLAT phi" if s["std"] < 0.01 else ""
                        print(f"  [KNN] task {tid}: n={s['n']:5d}  "
                              f"mean={s['mean']:+.4f}  std={s['std']:.4f}{flag}")
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
            demo_mp, _, _, demo_skel = restore_models(
                best_ckpt, STATE_DIM, ACTION_DIM, discrete=True, device=device,
            )
            run_demos(demo_mp, demo_skel, task_distribution,
                      save_dir=os.path.join(save_dir, "demos"),
                      n_demos=n_demos, gamma=gamma, device=device)
        except Exception as e:
            print(f"  [Demo] Error: {e}")

    if verbose:
        print("\nMiniGrid Nav pipeline complete.")
    return meta_policy, task_policies, skeleton, metrics


# ── Entry point ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Meta-RL on MiniGrid FourRooms navigation")
    parser.add_argument("--tasks",        nargs="+", default=list(TASK_GOALS.keys()),
                        choices=list(TASK_GOALS.keys()),
                        help="Task quadrants to include (default: all NW NE SW SE)")
    parser.add_argument("--iterations",   type=int,   default=5)
    parser.add_argument("--landmarks",    type=int,   default=64)
    parser.add_argument("--meta-epochs",  type=int,   default=200)
    parser.add_argument("--episodes-per-update", type=int, default=4)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--is-buffer-size",  type=int,   default=16)
    parser.add_argument("--is-clip-epsilon", type=float, default=0.2)
    parser.add_argument("--task-steps",   type=int,   default=20_000)
    parser.add_argument("--timesteps",    type=int,   default=50_000,
                        help="PPO timesteps per task in Phase 0")
    parser.add_argument("--n-envs",       type=int,   default=10)
    parser.add_argument("--shaping-scale",       type=float, default=1.0)
    parser.add_argument("--shaping-scale-start", type=float, default=0.1,
                        help="Shaping scale at iteration 0; linearly ramps to "
                             "--shaping-scale by the final iteration (default: 0.1)")
    parser.add_argument("--subgoal-threshold", type=float, default=float("inf"))
    parser.add_argument("--potential-alpha",   type=float, default=0.5)
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--n-demos",      type=int, default=5)
    parser.add_argument("--save-dir",     default="results/minigrid_nav")
    parser.add_argument("--load",         default=None, metavar="CKPT_DIR")
    parser.add_argument("--demo-only",    action="store_true")
    parser.add_argument("--device",       default="cpu")
    args = parser.parse_args()

    dist = MiniGridNavTaskDistribution(task_keys=args.tasks)

    if args.demo_only:
        if args.load is None:
            parser.error("--demo-only requires --load <checkpoint_dir>")
        ckpt = load_checkpoint(args.load, device=args.device)
        mp, _, _, skel = restore_models(
            ckpt, STATE_DIM, ACTION_DIM, discrete=True, device=args.device,
        )
        run_demos(mp, skel, dist,
                  save_dir=os.path.join(args.save_dir, "demos"),
                  n_demos=args.n_demos, device=args.device)
    else:
        main(
            task_distribution=dist,
            num_landmarks=args.landmarks,
            num_iterations=args.iterations,
            timesteps_per_task=args.timesteps,
            n_envs=args.n_envs,
            task_steps=args.task_steps,
            gamma=0.99,
            shaping_scale=args.shaping_scale,
            shaping_scale_start=args.shaping_scale_start,
            subgoal_threshold=args.subgoal_threshold,
            potential_alpha=args.potential_alpha,
            meta_epochs=args.meta_epochs,
            episodes_per_update=args.episodes_per_update,
            entropy_coef=args.entropy_coef,
            is_buffer_size=args.is_buffer_size,
            is_clip_epsilon=args.is_clip_epsilon,
            eval_episodes=args.eval_episodes,
            n_demos=args.n_demos,
            save_dir=args.save_dir,
            device=args.device,
            verbose=True,
        )
