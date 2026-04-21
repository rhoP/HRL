"""
Meta-RL pipeline for the Four-Rooms gridworld with moving goals.

State  : (agent_x, agent_y, goal_x, goal_y)  — float32, dim=4
Actions: 4 discrete directions (0=up 1=right 2=down 3=left)
Tasks  : A, B, C, D — differ in start/goal room pair

Run as:
    python3 scripts/train_four_rooms.py --iterations 5 --landmarks 16
    python3 scripts/train_four_rooms.py --demo-only --load results/four_rooms/best
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
                                  Phase3TrainingCallback, plot_phase3_results)
from utils.skeleton      import build_skeleton, refine_skeleton

from models.meta_policy_net import MetaPolicy
from algos.MetaPolicy import (
    phase1_build_skeleton, phase2_build_potential,
    ShapedRewardWrapper,
)
from algos.meta_policy_gradient import meta_policy_gradient_with_skeleton_shaping

STATE_DIM  = 4
ACTION_DIM = 4   # discrete; stored as int but dim=4 for one-hot consistency

TASK_CONFIGS = {
    "A": {"start": (1, 1), "goal": (9, 9)},
    "B": {"start": (9, 1), "goal": (1, 9)},
    "C": {"start": (1, 1), "goal": (9, 1)},
    "D": {"start": (1, 9), "goal": (9, 9)},
}
TASK_IDS = {"A": 0, "B": 1, "C": 2, "D": 3}


# ── Environment ────────────────────────────────────────────────────────────

class FourRoomsEnv(gym.Env):
    """
    Four-rooms gridworld (11×11). Horizontal wall at row center=5 with a
    doorway only at column 5. The agent must cross the wall to reach goals
    in the opposite half.

    Gymnasium-compatible: reset() → (obs, info), step() → (obs, r, term, trunc, info).
    info["success"] = 1.0 when the goal is reached.
    """

    metadata = {"render_modes": []}

    def __init__(self, task_key: str = "A", grid_size: int = 11, noise: float = 0.1,
                 max_steps: int = 200):
        super().__init__()
        self.task_key   = task_key
        self.grid_size  = grid_size
        self.center     = grid_size // 2
        self.noise      = noise
        self._max_steps  = max_steps
        self._step_count = 0

        self.observation_space = spaces.Box(
            low=0.0, high=float(grid_size - 1),
            shape=(STATE_DIM,), dtype=np.float32,
        )
        self.action_space = spaces.Discrete(4)

        self._moves = np.array([(0, -1), (1, 0), (0, 1), (-1, 0)], dtype=np.float32)
        self._config = TASK_CONFIGS[task_key]
        self.agent_pos = None
        self.goal_pos  = None

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.agent_pos   = np.array(self._config["start"], dtype=np.float32)
        self.goal_pos    = np.array(self._config["goal"],  dtype=np.float32)
        self._step_count = 0
        return self._obs(), {}

    def step(self, action):
        action = int(action)
        move = self._moves[action].copy()

        if np.random.random() < self.noise:
            move = move[::-1].copy()

        new_pos = self.agent_pos + move

        if self._is_valid(self.agent_pos, new_pos):
            self.agent_pos = new_pos

        self._step_count += 1
        dist      = np.linalg.norm(self.agent_pos - self.goal_pos)
        success   = dist < 0.5
        reward    = 10.0 if success else -0.1
        truncated = (not success) and self._step_count >= self._max_steps

        return self._obs(), reward, success, truncated, {"success": float(success)}

    def _obs(self):
        return np.concatenate([self.agent_pos, self.goal_pos]).astype(np.float32)

    def _is_valid(self, old_pos, new_pos):
        x, y = new_pos
        gs = self.grid_size
        c  = self.center

        if x < 0 or x >= gs or y < 0 or y >= gs:
            return False

        if old_pos[1] < c and y >= c:
            if int(round(x)) != c:
                return False

        if old_pos[1] > c and y <= c:
            if int(round(x)) != c:
                return False

        return True


# ── Task / Distribution ────────────────────────────────────────────────────

class FourRoomsTask:
    def __init__(self, task_key: str):
        self.id       = TASK_IDS[task_key]
        self.env_name = f"FourRooms-{task_key}"
        self._key     = task_key

    def create_env(self) -> FourRoomsEnv:
        return FourRoomsEnv(self._key)


class FourRoomsTaskDistribution:
    def __init__(self, task_keys=None):
        keys       = task_keys or list(TASK_CONFIGS.keys())
        self.tasks = [FourRoomsTask(k) for k in keys]

    def sample(self) -> FourRoomsTask:
        return self.tasks[np.random.randint(len(self.tasks))]


# ── Phase 0: PPO-based data collection ────────────────────────────────────

def phase0_collect(
    task_distribution: FourRoomsTaskDistribution,
    replay_buffer: ReplayBuffer,
    timesteps_per_task: int = 10_000,
    device: str = "cpu",
    verbose: bool = True,
) -> None:
    """Use SB3 PPO (supports Discrete action spaces) to fill the replay buffer."""
    for task in task_distribution.tasks:
        if verbose:
            print(f"  [Phase 0] Collecting from {task.env_name} "
                  f"for {timesteps_per_task} steps with PPO...")

        def _make():
            return task.create_env()

        vec_env = make_vec_env(_make, n_envs=1)
        model   = PPO("MlpPolicy", vec_env, verbose=0, device=device)
        model.learn(total_timesteps=timesteps_per_task)

        obs = vec_env.reset()
        for _ in range(timesteps_per_task):
            action, _ = model.predict(obs, deterministic=False)
            obs_next, reward, done, info = vec_env.step(action)
            truncated  = info[0].get("TimeLimit.truncated", False)
            terminated = bool(done[0]) and not truncated
            replay_buffer.push(
                obs[0], action[0], float(reward[0]),
                obs_next[0], bool(done[0]), task.id,
                terminated=terminated,
            )
            obs = obs_next if not done[0] else vec_env.reset()

        vec_env.close()

    if verbose:
        print(f"  [Phase 0] Buffer size: {len(replay_buffer)}")


# ── Phase 3: Per-task policies with shaped rewards ─────────────────────────

def phase3_train_task_policies(
    skeleton_data: dict,
    task_distribution: FourRoomsTaskDistribution,
    potential=None,
    timesteps_per_task: int = 10_000,
    shaping_scale: float    = 1.0,
    device: str             = "cpu",
    verbose: bool           = True,
    save_dir: str           = None,
    iteration: int          = 0,
) -> tuple:
    """
    Train one PPO policy per task with potential-based shaped rewards.

    Returns (task_policies, phase3_stats) where:
        task_policies — {task_id: PPO model}
        phase3_stats  — {task_name: {"ep_rewards", "ep_env_rewards",
                                     "ep_shaping", "ep_successes", "ep_lengths"}}
    If save_dir is provided, saves phase3_iter_{N:03d}.png there.
    """
    task_policies: dict = {}
    phase3_stats:  dict = {}

    for task in task_distribution.tasks:
        if verbose:
            print(f"  [Phase 3] Training policy for {task.env_name} (task {task.id})...")

        def _make_env(t=task, pot=potential, scale=shaping_scale):
            env = t.create_env()
            if pot is not None:
                env = ShapedRewardWrapper(env, pot, shaping_scale=scale)
            return env

        cb      = Phase3TrainingCallback()
        vec_env = make_vec_env(_make_env, n_envs=1)
        model   = PPO("MlpPolicy", vec_env, verbose=0, device=device)
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
            last20 = cb.ep_rewards[-20:]
            sr     = float(np.mean(cb.ep_successes[-20:])) if cb.ep_successes else 0.0
            print(f"    last-20-ep: avg_shaped_r={np.mean(last20):.3f}  "
                  f"success_rate={sr:.1%}")

    if save_dir is not None:
        plot_phase3_results(phase3_stats, save_dir, iteration=iteration)

    if verbose:
        print(f"  [Phase 3] Trained {len(task_policies)} task policy/ies.")
    return task_policies, phase3_stats


# ── Evaluation ─────────────────────────────────────────────────────────────

def evaluate_policy(
    meta_policy,
    task_distribution: FourRoomsTaskDistribution,
    n_episodes: int = 20,
    max_steps: int  = 200,
    gamma: float    = 0.99,
    device: str     = "cpu",
) -> dict:
    meta_policy.eval()
    successes: list = []
    returns:   list = []
    env_results: dict = {}

    for _ in range(n_episodes):
        task   = task_distribution.sample()
        env    = task.create_env()
        result = env.reset()
        s      = result[0] if isinstance(result, tuple) else result
        tau    = []
        done   = False
        ep_return = 0.0
        t      = 0
        success = False

        while not done and t < max_steps:
            with torch.no_grad():
                a_dist = meta_policy(s, tau)
                a      = a_dist.sample()
            a_np  = a.cpu().numpy().flatten()
            a_env = int(a_np[0]) if meta_policy.discrete else a_np
            s_next, r, terminated, truncated, info = env.step(a_env)
            done    = terminated or truncated
            success = success or bool(info.get("success", 0.0) > 0.5)
            ep_return += (gamma ** t) * r
            tau.append((np.asarray(s, dtype=np.float32), a_np, r))
            s = s_next
            t += 1

        env.close()
        successes.append(float(success))
        returns.append(ep_return)
        env_results.setdefault(task.env_name, []).append(float(success))

    meta_policy.train()
    per_env = {k: float(np.mean(v)) for k, v in env_results.items()}
    return {
        "success_rate": float(np.mean(successes)),
        "avg_return":   float(np.mean(returns)),
        "per_env":      per_env,
    }


# ── Demo runner ────────────────────────────────────────────────────────────

def run_demos(
    meta_policy,
    skeleton_data: dict,
    task_distribution: FourRoomsTaskDistribution,
    save_dir: str,
    n_demos: int   = 5,
    max_steps: int = 200,
    gamma: float   = 0.99,
    device: str    = "cpu",
) -> list:
    """
    Run demo episodes with the GRU meta-policy and save grid renders + JSON summary.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA

    os.makedirs(save_dir, exist_ok=True)
    c_states = [np.asarray(v, dtype=np.float32)
                for v in skeleton_data["critical_states"].values()]
    lm_np    = skeleton_data["landmarks"].cpu().numpy()
    D        = lm_np.shape[1]
    pca      = PCA(n_components=2).fit(lm_np) if D > 2 else None
    proj     = lambda s: pca.transform(np.asarray(s).reshape(1, -1))[0] if pca else np.asarray(s)[:2]
    lm_2d    = pca.transform(lm_np) if pca else lm_np[:, :2]
    crit_2d  = (pca.transform(np.stack(c_states)) if (pca and c_states)
                else (np.stack(c_states)[:, :2] if c_states else np.empty((0, 2))))
    cmap     = plt.get_cmap("tab10")

    meta_policy.eval()
    results = []
    for demo_i in range(n_demos):
        task   = task_distribution.sample()
        env    = task.create_env()
        result = env.reset()
        s      = result[0] if isinstance(result, tuple) else result
        s_arr  = np.asarray(s, dtype=np.float32).flatten()
        tau    = []
        traj_pts = [proj(s_arr)]
        ep_return = 0.0
        t     = 0
        done  = False
        success = False

        while not done and t < max_steps:
            with torch.no_grad():
                a_dist = meta_policy(s_arr, tau)
                a      = a_dist.sample()
            a_np  = a.cpu().numpy().flatten()
            a_env = int(a_np[0]) if meta_policy.discrete else a_np
            s_next, r, terminated, truncated, info = env.step(a_env)
            done    = terminated or truncated
            success = success or bool(info.get("success", 0.0) > 0.5)
            ep_return += (gamma ** t) * r
            tau.append((s_arr, a_np, r))
            s_arr = np.asarray(s_next, dtype=np.float32).flatten()
            traj_pts.append(proj(s_arr))
            t += 1

        env.close()
        result_d = {
            "demo": demo_i, "task": task.env_name,
            "success": success, "return": ep_return, "steps": t,
        }
        results.append(result_d)

        traj = np.array(traj_pts)
        fig, axes = plt.subplots(1, 2, figsize=(11, 5))

        # Left: grid trajectory
        ax = axes[0]
        grid_size = env.grid_size if hasattr(env, "grid_size") else 11
        ax.set_xlim(-0.5, grid_size - 0.5)
        ax.set_ylim(-0.5, grid_size - 0.5)
        ax.set_aspect("equal")
        ax.set_title(f"Grid trajectory — {task.env_name}")
        c = grid_size // 2
        ax.plot([c - 0.5, c - 0.5], [-0.5, c - 0.5], "k-", lw=2)
        ax.plot([c - 0.5, c - 0.5], [c + 0.5, grid_size - 0.5], "k-", lw=2)
        ax.plot([c + 0.5, c + 0.5], [-0.5, c - 0.5], "k-", lw=2)
        ax.plot([c + 0.5, c + 0.5], [c + 0.5, grid_size - 0.5], "k-", lw=2)
        for k in range(len(traj_pts) - 1):
            ax.plot([traj_pts[k][0], traj_pts[k+1][0]],
                    [traj_pts[k][1], traj_pts[k+1][1]],
                    color=cmap(k % 10), lw=1.2, alpha=0.8)
        ax.plot(*traj_pts[0][:2],  "g^", ms=10, label="start", zorder=5)
        ax.plot(*traj_pts[-1][:2], "*",
                color="gold" if success else "red", ms=12,
                label="end (✓)" if success else "end (✗)", zorder=5)
        ax.legend(fontsize=8)

        # Right: PCA skeleton projection
        ax = axes[1]
        ax.scatter(lm_2d[:, 0], lm_2d[:, 1], s=20, c="grey", alpha=0.5)
        if len(crit_2d):
            ax.scatter(crit_2d[:, 0], crit_2d[:, 1], s=150,
                       facecolors="none", edgecolors="crimson", lw=1.5)
        ax.set_title("PCA skeleton + trajectory")
        for k in range(len(traj) - 1):
            ax.plot(traj[k:k+2, 0], traj[k:k+2, 1],
                    color=cmap(0), lw=1.0, alpha=0.6)
        ax.plot(*traj[0],  "g^", ms=8, zorder=5)
        ax.plot(*traj[-1], "*", color="gold" if success else "red", ms=10, zorder=5)

        status = "SUCCESS" if success else "fail"
        fig.suptitle(f"Demo {demo_i} | {task.env_name} | {status} | "
                     f"return={ep_return:.2f}  steps={t}", fontsize=11)
        fig.tight_layout()
        fig.savefig(os.path.join(save_dir, f"demo_{demo_i:02d}_{task.env_name}.png"),
                    dpi=100, bbox_inches="tight")
        plt.close(fig)
        print(f"  [Demo {demo_i}] {task.env_name}  {status}  "
              f"return={ep_return:.3f}  steps={t}")

    meta_policy.train()
    with open(os.path.join(save_dir, "demo_summary.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"  [Demo] success_rate={np.mean([r['success'] for r in results]):.1%}")
    return results


# ── Main pipeline ──────────────────────────────────────────────────────────

def main(
    task_distribution: FourRoomsTaskDistribution = None,
    num_landmarks: int        = 16,
    num_iterations: int       = 5,
    refine_every: int         = 2,
    timesteps_per_task: int   = 10_000,
    task_steps: int           = 10_000,
    collect_episodes: int     = 50,
    gamma: float              = 0.99,
    shaping_scale: float      = 1.0,
    subgoal_threshold: float  = float("inf"),
    phase2_method: str        = "skeleton",
    meta_epochs: int          = 200,
    episodes_per_update: int  = 4,
    entropy_coef: float       = 0.01,
    is_buffer_size: int       = 16,
    is_clip_epsilon: float    = 0.2,
    eval_episodes: int        = 20,
    n_demos: int              = 5,
    save_dir: str             = "results/four_rooms",
    device: str               = "cpu",
    verbose: bool             = True,
):
    if task_distribution is None:
        task_distribution = FourRoomsTaskDistribution()

    os.makedirs(save_dir, exist_ok=True)

    if verbose:
        print("=" * 60)
        print("Meta-RL pipeline  (Four Rooms)")
        print(f"  tasks: {[t.env_name for t in task_distribution.tasks]}")
        print(f"  landmarks: {num_landmarks}  iterations: {num_iterations}")
        print(f"  save_dir: {save_dir}")
        print("=" * 60)

    metrics = {
        "skeleton_train_losses": [],
        "phase3_success_rates":  [],
        "phase4_returns":        [],
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
    else:
        rb = ReplayBuffer(device=device)
        if verbose:
            print("\n[Phase 0] Collecting initial data with PPO...")
        phase0_collect(task_distribution, rb,
                       timesteps_per_task=timesteps_per_task,
                       device=device, verbose=verbose)
        save_replay_buffer(rb, rb_path)
    if verbose:
        print(f"  Buffer: {len(rb)} transitions")

    # Phase 1
    if verbose:
        print("\n[Phase 1] Building Morse skeleton...")
    skeleton = phase1_build_skeleton(rb,
                                     state_dim=STATE_DIM, action_dim=ACTION_DIM,
                                     num_landmarks=num_landmarks,
                                     device=device, verbose=verbose)
    metrics["skeleton_train_losses"].append(skeleton.get("train_losses", []))
    n_sub = len(skeleton["critical_states"])
    if verbose:
        print(f"  Found {n_sub} critical state(s).")
        knn_est = skeleton.get("knn_estimator")
        if knn_est is not None:
            for tid, s in knn_est.back_ret_stats().items():
                flag = "  ← FLAT phi" if s["std"] < 0.01 else ""
                print(f"  [KNN] task {tid}: n={s['n']:5d}  "
                      f"back_ret mean={s['mean']:+.4f}  std={s['std']:.4f}{flag}")

    plot_skeleton_topology(skeleton, rb,
                           os.path.join(save_dir, "topology_initial.png"))

    if n_sub == 0:
        print("No subgoals found; aborting.")
        plot_training_curves(metrics, save_dir)
        return None, None, skeleton, metrics

    meta_policy    = MetaPolicy(STATE_DIM, ACTION_DIM, discrete=True).to(device)
    meta_value_net = None
    task_policies  = {}

    for iteration in range(num_iterations):
        if verbose:
            print(f"\n{'─'*60}")
            print(f"Iteration {iteration + 1}/{num_iterations}")
            print(f"{'─'*60}")

        # Phase 2 — build potential function
        if verbose:
            print(f"[Phase 2] Building {phase2_method} potential...")
        potential = phase2_build_potential(
            skeleton,
            replay_buffer=rb if phase2_method == "empirical" else None,
            method=phase2_method,
            gamma=gamma,
            verbose=verbose,
        )

        # Phase 3 — one PPO policy per task with shaped rewards
        if verbose:
            print("[Phase 3] Training task policies with shaped rewards...")
        task_policies, p3_stats = phase3_train_task_policies(
            skeleton, task_distribution,
            potential=potential,
            timesteps_per_task=task_steps,
            shaping_scale=shaping_scale,
            device=device, verbose=verbose,
            save_dir=save_dir, iteration=iteration,
        )
        p3_sr = float(np.mean([np.mean(v["ep_successes"][-20:])
                                for v in p3_stats.values()
                                if v["ep_successes"]]) if p3_stats else 0.0)
        metrics["phase3_success_rates"].append(p3_sr)

        # Phase 4 — meta-policy gradient with skeleton shaping
        if verbose:
            print("[Phase 4] Training meta-policy via policy gradient...")
        training_skeleton = dict(skeleton)
        training_skeleton["skeleton_potential"] = potential
        meta_policy, meta_value_net, p4_losses = \
            meta_policy_gradient_with_skeleton_shaping(
                meta_policy, task_distribution, training_skeleton,
                meta_epochs=meta_epochs,
                episodes_per_update=episodes_per_update,
                entropy_coef=entropy_coef,
                is_buffer_size=is_buffer_size,
                is_clip_epsilon=is_clip_epsilon,
                replay_buffer=rb,
                gamma=gamma,
                shaping_scale=shaping_scale,
                subgoal_threshold=subgoal_threshold,
                device=device, verbose=verbose,
            )
        metrics["phase4_returns"].append(p4_losses)

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
            metrics={
                "eval": eval_result,
                "p4_avg_return": float(np.mean(p4_losses)) if p4_losses else 0.0,
            },
        )
        improved = tracker.update(eval_result["success_rate"], ckpt_dir)
        if verbose and improved:
            print(f"  ★ New best (success_rate={eval_result['success_rate']:.1%})")

        save_iteration_visuals(skeleton, rb, metrics, save_dir, iteration)

        # Collect more data with trained meta-policy
        meta_policy.eval()
        for _ in range(collect_episodes):
            task    = task_distribution.sample()
            env     = task.create_env()
            result  = env.reset()
            obs_arr = np.asarray(result[0] if isinstance(result, tuple) else result,
                                 dtype=np.float32).flatten()
            tau  = []
            done = False
            while not done:
                with torch.no_grad():
                    a_dist = meta_policy(obs_arr, tau)
                    a      = a_dist.sample()
                a_np  = a.cpu().numpy().flatten()
                a_env = int(a_np[0]) if meta_policy.discrete else a_np
                obs_next, r, terminated, truncated, _ = env.step(a_env)
                done         = terminated or truncated
                obs_next_arr = np.asarray(obs_next, dtype=np.float32).flatten()
                rb.push(obs_arr, a_np, r, obs_next_arr, done, task.id,
                        terminated=terminated)
                tau.append((obs_arr, a_np, r))
                obs_arr = obs_next_arr
            env.close()
        meta_policy.train()

        save_replay_buffer(rb, rb_path)
        if verbose:
            print(f"  Buffer: {len(rb)} transitions")

        # Periodic skeleton refinement
        if (iteration + 1) % refine_every == 0 and iteration < num_iterations - 1:
            if verbose:
                print("[Refine] Rebuilding skeleton on enlarged buffer...")
            skeleton = refine_skeleton(skeleton, rb,
                                       num_landmarks=num_landmarks,
                                       device=device, verbose=verbose)
            metrics["skeleton_train_losses"].append(skeleton.get("train_losses", []))
            n_sub = len(skeleton["critical_states"])
            if verbose:
                print(f"  Refined skeleton: {n_sub} critical state(s).")
                knn_est = skeleton.get("knn_estimator")
                if knn_est is not None:
                    for tid, s in knn_est.back_ret_stats().items():
                        flag = "  ← FLAT phi" if s["std"] < 0.01 else ""
                        print(f"  [KNN] task {tid}: n={s['n']:5d}  "
                              f"back_ret mean={s['mean']:+.4f}  std={s['std']:.4f}{flag}")
            if n_sub == 0:
                print("  No subgoals after refinement; stopping.")
                break

    plot_training_curves(metrics, save_dir)

    # Demos from best model
    best_dir = os.path.join(save_dir, "best")
    if os.path.isdir(best_dir) and n_demos > 0:
        if verbose:
            print(f"\n[Demo] Running {n_demos} demos with best model...")
        try:
            best_ckpt = load_checkpoint(best_dir, device=device)
            demo_mp, _, demo_tp, demo_skel = restore_models(
                best_ckpt, STATE_DIM, ACTION_DIM, discrete=True, device=device,
            )
            run_demos(demo_mp, demo_skel, task_distribution,
                      save_dir=os.path.join(save_dir, "demos"),
                      n_demos=n_demos, gamma=gamma, device=device)
        except Exception as e:
            print(f"  [Demo] Error: {e}")

    if verbose:
        print("\nFour-Rooms pipeline complete.")

    return meta_policy, task_policies, skeleton, metrics


# ── Entry point ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Meta-RL on Four-Rooms gridworld")
    parser.add_argument("--tasks",        nargs="+", default=list(TASK_CONFIGS.keys()),
                        choices=list(TASK_CONFIGS.keys()),
                        help="Task keys to include (default: all A B C D)")
    parser.add_argument("--iterations",   type=int, default=2)
    parser.add_argument("--landmarks",    type=int, default=200)
    parser.add_argument("--meta-epochs",        type=int, default=200)
    parser.add_argument("--episodes-per-update", type=int, default=4,
                        help="Episodes collected per gradient update in Phase 4 (default: 4)")
    parser.add_argument("--entropy-coef", type=float, default=0.01,
                        help="Entropy bonus coefficient for meta-policy loss (default: 0.01)")
    parser.add_argument("--is-buffer-size", type=int, default=16,
                        help="Number of recent episodes in IS trajectory buffer (default: 16)")
    parser.add_argument("--is-clip-epsilon", type=float, default=0.2,
                        help="PPO-style IS ratio clip epsilon (default: 0.2)")
    parser.add_argument("--task-steps",   type=int, default=10_000,
                        help="PPO timesteps per task in Phase 3")
    parser.add_argument("--shaping-scale",     type=float, default=1.0,
                        help="Potential shaping scale")
    parser.add_argument("--subgoal-threshold", type=float, default=float("inf"),
                        help="Distance threshold for sparse shaping (inf = always shape)")
    parser.add_argument("--phase2-method", default="skeleton",
                        choices=["skeleton", "empirical"],
                        help="Phase 2 potential method")
    parser.add_argument("--timesteps",    type=int,   default=10_000,
                        help="PPO timesteps per task in Phase 0")
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--n-demos",      type=int, default=5)
    parser.add_argument("--save-dir",     default="results/four_rooms")
    parser.add_argument("--load",         default=None, metavar="CKPT_DIR")
    parser.add_argument("--demo-only",    action="store_true")
    parser.add_argument("--device",       default="cpu")
    args = parser.parse_args()

    dist = FourRoomsTaskDistribution(task_keys=args.tasks)

    if args.demo_only:
        if args.load is None:
            parser.error("--demo-only requires --load <checkpoint_dir>")
        ckpt = load_checkpoint(args.load, device=args.device)
        mp, _, tp, skel = restore_models(
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
            task_steps=args.task_steps,
            gamma=0.99,
            shaping_scale=args.shaping_scale,
            subgoal_threshold=args.subgoal_threshold,
            phase2_method=args.phase2_method,
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
