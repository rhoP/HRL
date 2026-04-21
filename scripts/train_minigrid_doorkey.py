"""
Meta-RL pipeline for MiniGrid DoorKey tasks.

Three tasks with the same key→door→goal structure at different grid scales
(5×5, 6×6, 8×8).  The skeleton should consistently identify the door passage
as the critical state across all three scales.

State  : (agent_x/N, agent_y/N, agent_dir/3, has_key,
          key_x/N, key_y/N, door_x/N, door_y/N, goal_x/N, goal_y/N)
         — float32, dim=10  (positions normalised by each env's own grid size N)
         key_x/key_y = −1/N once the key is carried
Actions: 7 discrete  (standard MiniGrid: left right forward pickup drop toggle done)
Tasks  : DK5 (5×5)  DK6 (6×6)  DK8 (8×8)

Run as:
    python3 scripts/train_minigrid_doorkey.py --iterations 5 --landmarks 64
    python3 scripts/train_minigrid_doorkey.py --demo-only --load results/minigrid_doorkey/best
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
    from minigrid.envs import DoorKeyEnv as _MG_DoorKey
    from minigrid.core.world_object import Key as _MG_Key, Door as _MG_Door, Goal as _MG_Goal
    _MINIGRID_OK = True
except ImportError:
    _MINIGRID_OK = False

STATE_DIM  = 10
ACTION_DIM = 7   # full MiniGrid action set

TASK_SIZES = {"DK5": 5, "DK6": 6, "DK8": 8}
TASK_IDS   = {"DK5": 0, "DK6": 1, "DK8": 2}


# ── Wrapper ────────────────────────────────────────────────────────────────

def _locate(env, obj_type):
    """Return (x, y) of the first object of obj_type in the grid, or (-1,-1)."""
    for y in range(env.height):
        for x in range(env.width):
            cell = env.grid.get(x, y)
            if isinstance(cell, obj_type):
                return float(x), float(y)
    return -1.0, -1.0


class MiniGridDoorKeyWrapper(gym.Wrapper):
    """
    Wraps MiniGrid DoorKeyEnv into the pipeline interface:
      - 10-dim compact float32 observation (positions normalised by grid size N)
      - Dense reward: +1 key pickup, +2 door open, +10 goal reached, −0.01/step
      - info["success"] = 1.0 when the goal is reached (terminated=True)
      - Exposes grid_size for the demo renderer
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)
        N = float(self.unwrapped.width)
        self._N = N
        self._door_pos  = np.array([-1.0, -1.0])
        self._key_pos   = np.array([-1.0, -1.0])
        self._goal_pos  = np.array([-1.0, -1.0])
        self.grid_size  = int(N)
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(STATE_DIM,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(ACTION_DIM)

    def reset(self, **kwargs):
        _, info = self.env.reset(**kwargs)
        self._refresh_object_positions()
        return self._obs(), info

    def step(self, action):
        _, r, terminated, truncated, info = self.env.step(int(action))
        success = bool(terminated) and (r > 0.0)

        # Richer reward signal so the skeleton can distinguish sub-goals
        carrying = self.unwrapped.carrying
        if success:
            r_out = 10.0
        elif r > 1.5:          # door opened (MiniGrid gives 0.0 but we can check state)
            r_out = 2.0
        elif r > 0.0:          # something rewarded (unlikely in DoorKey apart from goal)
            r_out = r
        else:
            r_out = -0.01

        # Detect key pickup and door opening via state changes
        if carrying is not None and isinstance(carrying, _MG_Key):
            # Key just picked up: mark key position as "carried"
            self._key_pos = np.array([-1.0, -1.0])
            if not hasattr(self, "_key_reward_given"):
                self._key_reward_given = False
            if not self._key_reward_given:
                r_out = max(r_out, 1.0)
                self._key_reward_given = True
        elif not hasattr(self, "_key_reward_given"):
            self._key_reward_given = False

        info["success"] = float(success)
        return self._obs(), r_out, terminated, truncated, info

    # ── helpers ───────────────────────────────────────────────────────────

    def _obs(self) -> np.ndarray:
        ax, ay = self.unwrapped.agent_pos
        ad     = float(self.unwrapped.agent_dir)
        N      = self._N
        carrying = self.unwrapped.carrying
        has_key  = float(carrying is not None and isinstance(carrying, _MG_Key))
        kx, ky   = self._key_pos if has_key == 0.0 else (-1.0, -1.0)
        dx, dy   = self._door_pos
        gx, gy   = self._goal_pos
        return np.array([
            ax / N, ay / N, ad / 3.0, has_key,
            kx / N, ky / N,
            dx / N, dy / N,
            gx / N, gy / N,
        ], dtype=np.float32)

    def _refresh_object_positions(self):
        kx, ky = _locate(self.unwrapped, _MG_Key)
        dx, dy = _locate(self.unwrapped, _MG_Door)
        gx, gy = _locate(self.unwrapped, _MG_Goal)
        self._key_pos  = np.array([kx, ky])
        self._door_pos = np.array([dx, dy])
        self._goal_pos = np.array([gx, gy])
        self._key_reward_given = False


# ── Task / Distribution ────────────────────────────────────────────────────

class MiniGridDoorKeyTask:
    def __init__(self, task_key: str):
        if not _MINIGRID_OK:
            raise RuntimeError("minigrid not installed.  Run: pip install minigrid")
        self.id        = TASK_IDS[task_key]
        self.env_name  = f"MiniGridDoorKey-{task_key}"
        self._key      = task_key
        self._size     = TASK_SIZES[task_key]

    def create_env(self, max_steps: int = None) -> gym.Env:
        if max_steps is None:
            max_steps = self._size * self._size * 10
        base = _MG_DoorKey(size=self._size, max_steps=max_steps)
        return MiniGridDoorKeyWrapper(base)


class MiniGridDoorKeyTaskDistribution:
    def __init__(self, task_keys=None):
        keys       = task_keys or list(TASK_SIZES.keys())
        self.tasks = [MiniGridDoorKeyTask(k) for k in keys]

    def sample(self) -> MiniGridDoorKeyTask:
        return self.tasks[np.random.randint(len(self.tasks))]


# ── Phase 0 ────────────────────────────────────────────────────────────────

def phase0_collect(
    task_distribution: MiniGridDoorKeyTaskDistribution,
    replay_buffer: ReplayBuffer,
    timesteps_per_task: int = 30_000,
    n_envs: int             = 10,
    device: str             = "cpu",
    verbose: bool           = True,
) -> None:
    for task in task_distribution.tasks:
        if verbose:
            print(f"  [Phase 0] {task.env_name}  PPO × {n_envs} envs  "
                  f"{timesteps_per_task} steps...")

        def _make(t=task):
            return t.create_env()

        vec_env = make_vec_env(_make, n_envs=n_envs)
        model   = PPO("MlpPolicy", vec_env, verbose=0, device=device)
        model.learn(total_timesteps=timesteps_per_task)

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


# ── Phase 3 ────────────────────────────────────────────────────────────────

def phase3_train_task_policies(
    skeleton_data: dict,
    task_distribution: MiniGridDoorKeyTaskDistribution,
    potential=None,
    timesteps_per_task: int = 30_000,
    shaping_scale: float    = 1.0,
    device: str             = "cpu",
    verbose: bool           = True,
    save_dir: str           = None,
    iteration: int          = 0,
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
    task_distribution: MiniGridDoorKeyTaskDistribution,
    n_episodes: int = 20,
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
        max_steps = task._size * task._size * 10
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
            s_next, r, terminated, truncated, info = env.step(int(a_np[0]))
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
    task_distribution: MiniGridDoorKeyTaskDistribution,
    save_dir: str,
    n_demos: int   = 5,
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
        task      = task_distribution.sample()
        env       = task.create_env()
        max_steps = task._size * task._size * 10
        obs, _    = env.reset()
        obs_arr   = np.asarray(obs, dtype=np.float32).flatten()
        N         = float(task._size)
        tau       = []
        map_pts   = []   # (agent_x, agent_y) in grid coords
        pca_pts   = []
        ep_ret    = 0.0
        t         = 0
        done      = False
        success   = False

        while not done and t < max_steps:
            ax = obs_arr[0] * N
            ay = obs_arr[1] * N
            map_pts.append([ax, ay])
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

        # Capture final position
        map_pts.append([obs_arr[0] * N, obs_arr[1] * N])
        pca_pts.append(_proj(obs_arr))
        env.close()

        results.append({
            "demo": demo_i, "task": task.env_name,
            "success": success, "return": ep_ret, "steps": t,
        })

        map_arr = np.array(map_pts)
        pca_arr = np.array(pca_pts)
        status  = "success" if success else "fail"

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        gs = task._size

        # Left: grid map with key / door / goal markers
        ax = axes[0]
        ax.set_xlim(-0.5, gs - 0.5)
        ax.set_ylim(-0.5, gs - 0.5)
        ax.set_aspect("equal")
        # Objects from last obs (positions normalised)
        kx_n, ky_n = obs_arr[4] * N, obs_arr[5] * N
        dx_n, dy_n = obs_arr[6] * N, obs_arr[7] * N
        gx_n, gy_n = obs_arr[8] * N, obs_arr[9] * N
        if kx_n >= 0:
            ax.plot(kx_n, ky_n, marker="s", ms=10, color=_P["gold"],
                    zorder=4, label="key")
        ax.plot(dx_n, dy_n, marker="^", ms=10, color=_P["mid"],
                zorder=4, label="door")
        ax.plot(gx_n, gy_n, marker="*", ms=14, color=_P["gold"],
                zorder=4, label="goal")
        for k in range(len(map_arr) - 1):
            ax.plot(map_arr[k:k+2, 0], map_arr[k:k+2, 1],
                    color=_c(k), lw=1.0, alpha=0.7)
        ax.plot(*map_arr[0],  marker="^", ms=9, color=_P["blue"], zorder=5, label="start")
        ax.plot(*map_arr[-1], marker="*", ms=11,
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
    task_distribution: MiniGridDoorKeyTaskDistribution = None,
    num_landmarks: int       = 64,
    num_iterations: int      = 5,
    refine_every: int        = 2,
    timesteps_per_task: int  = 30_000,
    n_envs: int              = 10,
    task_steps: int          = 30_000,
    collect_episodes: int    = 50,
    gamma: float             = 0.99,
    shaping_scale: float     = 1.0,
    subgoal_threshold: float = float("inf"),
    potential_alpha: float   = 0.5,
    meta_epochs: int         = 200,
    episodes_per_update: int = 4,
    entropy_coef: float      = 0.01,
    is_buffer_size: int      = 16,
    is_clip_epsilon: float   = 0.2,
    eval_episodes: int       = 20,
    n_demos: int             = 5,
    save_dir: str            = "results/minigrid_doorkey",
    device: str              = "cpu",
    verbose: bool            = True,
):
    if not _MINIGRID_OK:
        raise RuntimeError("minigrid not installed.  Run: pip install minigrid")
    if task_distribution is None:
        task_distribution = MiniGridDoorKeyTaskDistribution()

    os.makedirs(save_dir, exist_ok=True)

    if verbose:
        print("=" * 60)
        print("Meta-RL pipeline  (MiniGrid DoorKey)")
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
                       n_envs=n_envs,
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

    meta_policy    = MetaPolicy(STATE_DIM, ACTION_DIM, discrete=True).to(device)
    meta_value_net = None
    task_policies  = {}
    training_state = None

    for iteration in range(num_iterations):
        if verbose:
            print(f"\n{'─'*60}")
            print(f"Iteration {iteration + 1}/{num_iterations}")
            print(f"{'─'*60}")

        # Phase 2
        if verbose:
            print(f"[Phase 2] Combined potential (α={potential_alpha:.2f})...")
        potential = phase2_build_potential(
            skeleton, replay_buffer=rb,
            alpha=potential_alpha, gamma=gamma, verbose=verbose,
        )

        # Phase 3
        if verbose:
            print("[Phase 3] Training task policies...")
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
                shaping_scale=shaping_scale,
                subgoal_threshold=subgoal_threshold,
                device=device, verbose=verbose,
                training_state=training_state,
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
            metrics={"eval": eval_result,
                     "p4_avg_return": float(np.mean(p4_losses)) if p4_losses else 0.0},
        )
        if tracker.update(eval_result["success_rate"], ckpt_dir) and verbose:
            print(f"  ★ New best (success_rate={eval_result['success_rate']:.1%})")

        save_iteration_visuals(skeleton, rb, metrics, save_dir, iteration)

        # Collect more data
        meta_policy.eval()
        for _ in range(collect_episodes):
            task      = task_distribution.sample()
            env       = task.create_env()
            max_steps = task._size * task._size * 10
            obs, _    = env.reset()
            obs_arr   = np.asarray(obs, dtype=np.float32).flatten()
            tau       = []
            done      = False
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
                                       device=device, verbose=verbose)
            skeleton["_potential_stale"] = True
            if training_state is not None:
                training_state["is_buffer"] = None
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
        print("\nMiniGrid DoorKey pipeline complete.")
    return meta_policy, task_policies, skeleton, metrics


# ── Entry point ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Meta-RL on MiniGrid DoorKey tasks")
    parser.add_argument("--tasks",        nargs="+", default=list(TASK_SIZES.keys()),
                        choices=list(TASK_SIZES.keys()),
                        help="Task keys to include (default: DK5 DK6 DK8)")
    parser.add_argument("--iterations",   type=int,   default=5)
    parser.add_argument("--landmarks",    type=int,   default=64)
    parser.add_argument("--meta-epochs",  type=int,   default=200)
    parser.add_argument("--episodes-per-update", type=int, default=4)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--is-buffer-size",  type=int,   default=16)
    parser.add_argument("--is-clip-epsilon", type=float, default=0.2)
    parser.add_argument("--task-steps",   type=int,   default=30_000)
    parser.add_argument("--timesteps",    type=int,   default=30_000,
                        help="PPO timesteps per task in Phase 0")
    parser.add_argument("--n-envs",       type=int,   default=10)
    parser.add_argument("--shaping-scale",     type=float, default=1.0)
    parser.add_argument("--subgoal-threshold", type=float, default=float("inf"))
    parser.add_argument("--potential-alpha",   type=float, default=0.5)
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--n-demos",      type=int, default=5)
    parser.add_argument("--save-dir",     default="results/minigrid_doorkey")
    parser.add_argument("--load",         default=None, metavar="CKPT_DIR")
    parser.add_argument("--demo-only",    action="store_true")
    parser.add_argument("--device",       default="cpu")
    args = parser.parse_args()

    dist = MiniGridDoorKeyTaskDistribution(task_keys=args.tasks)

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
