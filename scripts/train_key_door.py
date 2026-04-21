"""
Meta-RL pipeline for the Key-Door maze.

State  : (agent_x, agent_y, has_key, door_open, key_a_exists, key_b_exists)
         — float32, dim=6
Actions: 4 discrete directions (0=up 1=right 2=down 3=left), continuous scale=0.5
Tasks  : A (key A only), B (key B only), C (both keys, must use A)

The critical state discovered by the skeleton should correspond to the door
passage at (5, 5) — the topological bottleneck every task must pass through.

Run as:
    python3 scripts/train_key_door.py --iterations 5 --landmarks 16
    python3 scripts/train_key_door.py --demo-only --load results/key_door/best
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
                                  save_iteration_visuals)
from utils.skeleton      import build_skeleton, refine_skeleton

from algos.MetaPolicy import (
    SubPolicy, MetaPolicy, MetaValueNetwork,
    phase1_build_skeleton, phase2_train_value_net,
    phase3_train_sub_policies, phase4_train_meta_policy,
    check_sub_policy_convergence,
)

STATE_DIM  = 6
ACTION_DIM = 4   # discrete

TASK_CONFIGS = {
    "A": {"key_a_exists": True,  "key_b_exists": False, "correct_key": None},
    "B": {"key_a_exists": False, "key_b_exists": True,  "correct_key": None},
    "C": {"key_a_exists": True,  "key_b_exists": True,  "correct_key": "A"},
}
TASK_IDS = {"A": 0, "B": 1, "C": 2}

_START_POS  = np.array([5.0, 1.0])
_KEY_A_POS  = np.array([3.0, 3.0])
_KEY_B_POS  = np.array([7.0, 3.0])
_DOOR_POS   = np.array([5.0, 5.0])
_GOAL_POS   = np.array([5.0, 8.0])
_MOVE_SCALE = 0.5
_PICKUP_R   = 0.5
_INTERACT_R = 0.5
_BOUNDS     = (0.0, 10.0)


# ── Environment ────────────────────────────────────────────────────────────

class KeyDoorEnv(gym.Env):
    """
    Key-Door maze. The agent must pick up the correct key, pass through
    the door (critical bottleneck state), and reach the goal.

    Gymnasium-compatible. info["success"] = 1.0 when goal is reached.
    """

    metadata = {"render_modes": []}

    def __init__(self, task_key: str = "A"):
        super().__init__()
        self.task_key = task_key
        self._cfg     = TASK_CONFIGS[task_key]

        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, 0, 0, 0], dtype=np.float32),
            high=np.array([10, 10, 1, 1, 1, 1], dtype=np.float32),
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(4)
        self._moves = np.array([
            (0,  -_MOVE_SCALE),
            (_MOVE_SCALE, 0),
            (0,   _MOVE_SCALE),
            (-_MOVE_SCALE, 0),
        ], dtype=np.float32)

        # State variables — initialised in reset()
        self.agent_pos    = None
        self.has_key      = False
        self.door_open    = False
        self.key_a_taken  = False
        self.key_b_taken  = False
        self.key_a_exists = False
        self.key_b_exists = False
        self.correct_key  = None

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.agent_pos    = _START_POS.copy()
        self.has_key      = False
        self.door_open    = False
        self.key_a_taken  = False
        self.key_b_taken  = False
        self.key_a_exists = self._cfg["key_a_exists"]
        self.key_b_exists = self._cfg["key_b_exists"]
        self.correct_key  = self._cfg["correct_key"]
        return self._obs(), {}

    def step(self, action):
        action = int(action)
        self.agent_pos = np.clip(
            self.agent_pos + self._moves[action], _BOUNDS[0], _BOUNDS[1]
        )

        reward = -0.1
        info   = {"success": 0.0, "reached_key": False,
                  "reached_door": False, "reached_goal": False}

        # Key pickup
        if self.key_a_exists and not self.key_a_taken:
            if np.linalg.norm(self.agent_pos - _KEY_A_POS) < _PICKUP_R:
                self.has_key     = True
                self.key_a_taken = True
                reward = 1.0
                info["reached_key"] = True

        if self.key_b_exists and not self.key_b_taken:
            if np.linalg.norm(self.agent_pos - _KEY_B_POS) < _PICKUP_R:
                self.has_key     = True
                self.key_b_taken = True
                reward = 1.0
                info["reached_key"] = True

        # Door interaction
        if np.linalg.norm(self.agent_pos - _DOOR_POS) < _INTERACT_R:
            info["reached_door"] = True
            if self.has_key and not self.door_open:
                correct = True
                if self.correct_key == "A":
                    correct = self.key_a_taken
                elif self.correct_key == "B":
                    correct = self.key_b_taken
                if correct:
                    self.door_open = True
                    reward = 2.0

        # Goal
        done    = False
        success = False
        if np.linalg.norm(self.agent_pos - _GOAL_POS) < _INTERACT_R and self.door_open:
            success = True
            done    = True
            reward  = 10.0
            info["reached_goal"] = True

        info["success"] = float(success)
        return self._obs(), reward, done, False, info

    def _obs(self):
        return np.array([
            self.agent_pos[0], self.agent_pos[1],
            float(self.has_key),
            float(self.door_open),
            float(self.key_a_exists),
            float(self.key_b_exists),
        ], dtype=np.float32)


# ── Task / Distribution ────────────────────────────────────────────────────

class KeyDoorTask:
    def __init__(self, task_key: str):
        self.id       = TASK_IDS[task_key]
        self.env_name = f"KeyDoor-{task_key}"
        self._key     = task_key

    def create_env(self) -> KeyDoorEnv:
        return KeyDoorEnv(self._key)


class KeyDoorTaskDistribution:
    def __init__(self, task_keys=None):
        keys       = task_keys or list(TASK_CONFIGS.keys())
        self.tasks = [KeyDoorTask(k) for k in keys]

    def sample(self) -> KeyDoorTask:
        return self.tasks[np.random.randint(len(self.tasks))]


# ── Phase 0: PPO-based data collection ────────────────────────────────────

def phase0_collect(
    task_distribution: KeyDoorTaskDistribution,
    replay_buffer: ReplayBuffer,
    timesteps_per_task: int = 15_000,
    device: str = "cpu",
    verbose: bool = True,
) -> None:
    """Use SB3 PPO (Discrete action spaces) to fill the replay buffer."""
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
            obs_next, reward, done, _ = vec_env.step(action)
            replay_buffer.push(
                obs[0], action[0], float(reward[0]),
                obs_next[0], bool(done[0]), task.id,
            )
            obs = obs_next if not done[0] else vec_env.reset()

        vec_env.close()

    if verbose:
        print(f"  [Phase 0] Buffer size: {len(replay_buffer)}")


# ── Evaluation ─────────────────────────────────────────────────────────────

def evaluate_policy(
    meta_policy,
    sub_policies: dict,
    skeleton_data: dict,
    task_distribution: KeyDoorTaskDistribution,
    n_episodes: int = 20,
    max_steps: int  = 300,
    gamma: float    = 0.99,
    device: str     = "cpu",
) -> dict:
    c_list    = list(skeleton_data["critical_states"].keys())
    successes = []
    returns   = []
    env_results: dict = {}

    for _ in range(n_episodes):
        task = task_distribution.sample()
        env  = task.create_env()
        obs, _ = env.reset()
        obs_t  = torch.tensor(obs, dtype=torch.float32, device=device)

        done      = False
        ep_return = 0.0
        t         = 0
        success   = False

        while not done and t < max_steps:
            with torch.no_grad():
                c_idx = meta_policy(obs_t).sample().item()
            c_id = c_list[c_idx]
            sp   = sub_policies.get(c_id)
            T_c  = 0

            while sp is not None and not sp.is_terminated(obs_t, done, T_c):
                with torch.no_grad():
                    a = sp.get_action(obs_t)
                a_np = int(a.cpu().numpy()) if a.numel() == 1 else a.cpu().numpy()
                obs_next, r, terminated, truncated, info = env.step(a_np)
                done    = terminated or truncated
                success = success or bool(info.get("success", 0.0) > 0.5)
                ep_return += (gamma ** t) * r
                obs_t = torch.tensor(obs_next, dtype=torch.float32, device=device)
                t += 1; T_c += 1
                if done or t >= max_steps:
                    break

        env.close()
        successes.append(float(success))
        returns.append(ep_return)
        env_results.setdefault(task.env_name, []).append(float(success))

    per_env = {k: float(np.mean(v)) for k, v in env_results.items()}
    return {
        "success_rate": float(np.mean(successes)),
        "avg_return":   float(np.mean(returns)),
        "per_env":      per_env,
    }


# ── Demo runner ────────────────────────────────────────────────────────────

def run_demos(
    meta_policy,
    sub_policies: dict,
    skeleton_data: dict,
    task_distribution: KeyDoorTaskDistribution,
    save_dir: str,
    n_demos: int   = 5,
    max_steps: int = 300,
    gamma: float   = 0.99,
    device: str    = "cpu",
) -> list:
    """
    Save per-demo 2D map plots (showing agent path, key/door/goal positions)
    and PCA skeleton projection side-by-side, plus a JSON summary.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA

    os.makedirs(save_dir, exist_ok=True)
    c_list   = list(skeleton_data["critical_states"].keys())
    lm_np    = skeleton_data["landmarks"].cpu().numpy()
    D        = lm_np.shape[1]
    pca      = PCA(n_components=2).fit(lm_np) if D > 2 else None
    lm_2d    = pca.transform(lm_np) if pca else lm_np[:, :2]
    crit_ids = list(skeleton_data["critical_states"].keys())
    crit_2d  = lm_2d[crit_ids] if crit_ids else np.empty((0, 2))
    cmap     = plt.get_cmap("tab10")

    def _proj(s_np):
        if pca is not None:
            return pca.transform(np.array(s_np[:D]).reshape(1, -1))[0]
        return np.array(s_np[:2])

    results = []
    for demo_i in range(n_demos):
        task   = task_distribution.sample()
        env    = task.create_env()
        obs, _ = env.reset()
        obs_t  = torch.tensor(obs, dtype=torch.float32, device=device)

        map_pts     = [obs[:2].copy()]
        pca_pts     = [_proj(obs)]
        sg_sequence = []
        ep_return   = 0.0
        t           = 0
        done        = False
        success     = False

        while not done and t < max_steps:
            with torch.no_grad():
                c_idx = meta_policy(obs_t).sample().item()
            c_id = c_list[c_idx]
            sp   = sub_policies.get(c_id)
            sg_sequence.append(int(c_id))
            T_c  = 0

            while sp is not None and not sp.is_terminated(obs_t, done, T_c):
                with torch.no_grad():
                    a = sp.get_action(obs_t)
                a_np = int(a.cpu().numpy()) if a.numel() == 1 else a.cpu().numpy()
                obs_next, r, terminated, truncated, info = env.step(a_np)
                done    = terminated or truncated
                success = success or bool(info.get("success", 0.0) > 0.5)
                ep_return += (gamma ** t) * r
                map_pts.append(obs_next[:2].copy())
                pca_pts.append(_proj(obs_next))
                obs_t = torch.tensor(obs_next, dtype=torch.float32, device=device)
                t += 1; T_c += 1
                if done or t >= max_steps:
                    break

        env.close()
        result = {
            "demo": demo_i, "task": task.env_name,
            "success": success, "return": ep_return, "steps": t,
            "subgoals": sg_sequence,
        }
        results.append(result)

        map_pts = np.array(map_pts)
        pca_pts = np.array(pca_pts)

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Left: 2D map
        ax = axes[0]
        ax.set_xlim(-0.5, 10.5)
        ax.set_ylim(-0.5, 10.5)
        ax.set_aspect("equal")
        cfg = TASK_CONFIGS[task._key]
        # Landmarks (key/door/goal)
        if cfg["key_a_exists"]:
            ax.plot(*_KEY_A_POS, "bs", ms=12, label="key A", zorder=3)
        if cfg["key_b_exists"]:
            ax.plot(*_KEY_B_POS, "gs", ms=12, label="key B", zorder=3)
        ax.plot(*_DOOR_POS, "k^", ms=12, label="door", zorder=3)
        ax.plot(*_GOAL_POS, "r*", ms=14, label="goal", zorder=3)
        # Trajectory
        for k in range(len(map_pts) - 1):
            sg_c = cmap(sg_sequence[min(k, len(sg_sequence) - 1)] % 10
                        if sg_sequence else 0)
            ax.plot(map_pts[k:k+2, 0], map_pts[k:k+2, 1],
                    color=sg_c, lw=1.2, alpha=0.8)
        ax.plot(*map_pts[0],  "g^", ms=10, label="start", zorder=5)
        ax.plot(*map_pts[-1], "*",
                color="gold" if success else "red", ms=12,
                label="end (✓)" if success else "end (✗)", zorder=5)
        ax.legend(fontsize=7, loc="upper right")
        ax.set_title(f"Map — {task.env_name}")

        # Right: PCA skeleton
        ax = axes[1]
        ax.scatter(lm_2d[:, 0], lm_2d[:, 1], s=20, c="grey", alpha=0.5)
        if len(crit_2d):
            ax.scatter(crit_2d[:, 0], crit_2d[:, 1], s=150,
                       facecolors="none", edgecolors="crimson", lw=1.5,
                       label="subgoals")
        for k in range(len(pca_pts) - 1):
            ax.plot(pca_pts[k:k+2, 0], pca_pts[k:k+2, 1],
                    color=cmap(0), lw=1.0, alpha=0.7)
        ax.plot(*pca_pts[0],  "g^", ms=8, zorder=5)
        ax.plot(*pca_pts[-1], "*", color="gold" if success else "red",
                ms=10, zorder=5)
        ax.legend(fontsize=7)
        ax.set_title("PCA skeleton + trajectory")

        status = "SUCCESS" if success else "fail"
        fig.suptitle(f"Demo {demo_i} | {task.env_name} | {status} | "
                     f"return={ep_return:.2f}  steps={t}", fontsize=11)
        fig.tight_layout()
        fig.savefig(os.path.join(save_dir, f"demo_{demo_i:02d}_{task.env_name}.png"),
                    dpi=100, bbox_inches="tight")
        plt.close(fig)
        print(f"  [Demo {demo_i}] {task.env_name}  {status}  "
              f"return={ep_return:.3f}  steps={t}")

    with open(os.path.join(save_dir, "demo_summary.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"  [Demo] success_rate={np.mean([r['success'] for r in results]):.1%}")
    return results


# ── Main pipeline ──────────────────────────────────────────────────────────

def main(
    task_distribution: KeyDoorTaskDistribution = None,
    num_landmarks: int      = 16,
    num_iterations: int     = 5,
    refine_every: int       = 2,
    timesteps_per_task: int = 15_000,
    collect_episodes: int   = 50,
    gamma: float            = 0.99,
    sub_epochs: int         = 50,
    meta_epochs: int        = 200,
    eval_episodes: int      = 20,
    n_demos: int            = 5,
    save_dir: str           = "results/key_door",
    device: str             = "cpu",
    verbose: bool           = True,
):
    if task_distribution is None:
        task_distribution = KeyDoorTaskDistribution()

    os.makedirs(save_dir, exist_ok=True)

    if verbose:
        print("=" * 60)
        print("Meta-RL pipeline  (Key-Door)")
        print(f"  tasks: {[t.env_name for t in task_distribution.tasks]}")
        print(f"  landmarks: {num_landmarks}  iterations: {num_iterations}")
        print(f"  save_dir: {save_dir}")
        print("=" * 60)

    metrics = {
        "skeleton_train_losses": [],
        "phase2_losses":         [],
        "phase3_pi_losses":      [],
        "phase3_v_losses":       [],
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
    skeleton = phase1_build_skeleton(rb, num_landmarks=num_landmarks,
                                     device=device, verbose=verbose)
    metrics["skeleton_train_losses"].append(skeleton.get("train_losses", []))
    n_sub = len(skeleton["critical_states"])
    if verbose:
        print(f"  Found {n_sub} critical state(s).")

    plot_skeleton_topology(skeleton, rb,
                           os.path.join(save_dir, "topology_initial.png"))

    if n_sub == 0:
        print("No subgoals found; aborting.")
        plot_training_curves(metrics, save_dir)
        return None, None, skeleton, metrics

    meta_policy    = None
    meta_value_net = None
    sub_policies   = {}
    hitting_nets   = {}
    sp_converged   = True   # run Phase 2 on first iteration

    for iteration in range(num_iterations):
        if verbose:
            print(f"\n{'─'*60}")
            print(f"Iteration {iteration + 1}/{num_iterations}")
            print(f"{'─'*60}")

        # Phase 2 — only re-train V_H when sub-policies have converged
        if sp_converged or not hitting_nets:
            if verbose:
                reason = "first iteration" if not hitting_nets else "sub-policies converged"
                print(f"[Phase 2] Training hitting-time value nets ({reason})...")
            hitting_nets, p2_losses = phase2_train_value_net(
                skeleton, rb, gamma=gamma, device=device, verbose=verbose,
            )
            for net in hitting_nets.values():
                net.requires_grad_(False)
            metrics["phase2_losses"].append(p2_losses)
            sp_converged = False
        else:
            if verbose:
                print("[Phase 2] Skipping — V_H frozen until sub-policies converge.")
            metrics["phase2_losses"].append([])

        # Phase 3 — discrete actions
        # Threshold 1.0: covers proximity in the mixed continuous/binary state space
        sub_policies, p3_pi, p3_v = phase3_train_sub_policies(
            skeleton, rb, hitting_nets,
            task_distribution=task_distribution,
            state_dim=STATE_DIM, action_dim=ACTION_DIM,
            discrete=True,
            gamma=gamma, epochs=sub_epochs,
            reach_threshold=1.0,
            device=device, verbose=verbose,
            existing_sub_policies=sub_policies,
            carry_over_policies=True,
        )
        metrics["phase3_pi_losses"].append(p3_pi)
        metrics["phase3_v_losses"].append(p3_v)

        sp_converged = check_sub_policy_convergence(
            sub_policies, task_distribution, device=device,
        )
        if verbose:
            print(f"  Sub-policy convergence: "
                  f"{'converged' if sp_converged else 'not yet — V_H stays frozen'}")

        # Phase 4
        meta_policy, meta_value_net, p4_returns = phase4_train_meta_policy(
            sub_policies, skeleton, task_distribution,
            state_dim=STATE_DIM, gamma=gamma,
            meta_epochs=meta_epochs,
            total_iterations=num_iterations,
            current_iteration=iteration,
            device=device, verbose=verbose,
        )
        metrics["phase4_returns"].append(p4_returns)

        if meta_policy is None:
            continue

        # Evaluate
        eval_result = evaluate_policy(
            meta_policy, sub_policies, skeleton, task_distribution,
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
            sub_policies=sub_policies, hitting_nets=hitting_nets,
            skeleton_data=skeleton, replay_buffer=rb,
            metrics={
                "eval": eval_result,
                "p4_avg_return": float(np.mean(p4_returns)) if p4_returns else 0.0,
            },
        )
        improved = tracker.update(eval_result["success_rate"], ckpt_dir)
        if verbose and improved:
            print(f"  ★ New best (success_rate={eval_result['success_rate']:.1%})")

        save_iteration_visuals(skeleton, rb, metrics, save_dir, iteration)

        # Collect more data
        c_list = list(skeleton["critical_states"].keys())
        for _ in range(collect_episodes):
            task   = task_distribution.sample()
            env    = task.create_env()
            obs, _ = env.reset()
            obs_t  = torch.tensor(obs, dtype=torch.float32, device=device)
            done   = False
            while not done:
                with torch.no_grad():
                    c_idx = meta_policy(obs_t).sample().item()
                c_id = c_list[c_idx]
                sp   = sub_policies.get(c_id)
                T_c  = 0
                while sp is not None and not sp.is_terminated(obs_t, done, T_c):
                    with torch.no_grad():
                        a = sp.get_action(obs_t)
                    a_np = int(a.cpu().numpy()) if a.numel() == 1 else a.cpu().numpy()
                    obs_next, r, terminated, truncated, _ = env.step(a_np)
                    done = terminated or truncated
                    rb.push(obs_t.cpu().numpy(), np.array([int(a_np)]),
                            r, obs_next, done, task.id)
                    obs_t = torch.tensor(obs_next, dtype=torch.float32, device=device)
                    T_c += 1
                    if done:
                        break
            env.close()

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
            demo_mp, _, demo_sp, _, demo_skel = restore_models(
                best_ckpt, STATE_DIM, ACTION_DIM, device=device,
            )
            run_demos(demo_mp, demo_sp, demo_skel, task_distribution,
                      save_dir=os.path.join(save_dir, "demos"),
                      n_demos=n_demos, gamma=gamma, device=device)
        except Exception as e:
            print(f"  [Demo] Error: {e}")

    if verbose:
        print("\nKey-Door pipeline complete.")

    return meta_policy, sub_policies, skeleton, metrics


# ── Entry point ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Meta-RL on Key-Door maze")
    parser.add_argument("--tasks",        nargs="+", default=list(TASK_CONFIGS.keys()),
                        choices=list(TASK_CONFIGS.keys()),
                        help="Task keys (default: A B C)")
    parser.add_argument("--iterations",   type=int, default=5)
    parser.add_argument("--landmarks",    type=int, default=16)
    parser.add_argument("--meta-epochs",  type=int, default=200)
    parser.add_argument("--sub-epochs",   type=int, default=50)
    parser.add_argument("--timesteps",    type=int, default=15_000,
                        help="PPO timesteps per task in Phase 0")
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--n-demos",      type=int, default=5)
    parser.add_argument("--save-dir",     default="results/key_door")
    parser.add_argument("--load",         default=None, metavar="CKPT_DIR")
    parser.add_argument("--demo-only",    action="store_true")
    parser.add_argument("--device",       default="cpu")
    args = parser.parse_args()

    dist = KeyDoorTaskDistribution(task_keys=args.tasks)

    if args.demo_only:
        if args.load is None:
            parser.error("--demo-only requires --load <checkpoint_dir>")
        ckpt     = load_checkpoint(args.load, device=args.device)
        mp, _, sp, _, skel = restore_models(
            ckpt, STATE_DIM, ACTION_DIM, device=args.device,
        )
        run_demos(mp, sp, skel, dist,
                  save_dir=os.path.join(args.save_dir, "demos"),
                  n_demos=args.n_demos, device=args.device)
    else:
        main(
            task_distribution=dist,
            num_landmarks=args.landmarks,
            num_iterations=args.iterations,
            timesteps_per_task=args.timesteps,
            gamma=0.99,
            sub_epochs=args.sub_epochs,
            meta_epochs=args.meta_epochs,
            eval_episodes=args.eval_episodes,
            n_demos=args.n_demos,
            save_dir=args.save_dir,
            device=args.device,
            verbose=True,
        )
