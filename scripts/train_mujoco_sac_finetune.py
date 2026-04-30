"""
Fine-tune the best meta-trained task policies with plain SAC — no shaped
rewards and no meta-policy influence.

Loads the task-policy .zip files from a meta-RL checkpoint (default:
results/mujoco_meta/best/task_policies), then continues SAC training on
each task's raw environment.  Uses the same Walker2d-v5 task distribution
as train_mujoco_meta.py by default.

Produces per-task learning curves and a summary via utils/viz, written to
--save-dir (default: results/mujoco_sac_finetune).

Usage:
    python scripts/train_mujoco_sac_finetune.py
    python scripts/train_mujoco_sac_finetune.py \\
        --checkpoint results/mujoco_meta/best \\
        --task-steps 100000 --device cpu \\
        --save-dir results/sac_finetune_run1
    python scripts/train_mujoco_sac_finetune.py --no-warmstart
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

from utils.viz import (
    Phase3TrainingCallback,
    plot_phase3_results,
    plot_training_curves,
)


# ── Task distribution (mirrored from train_mujoco_meta) ──────────────────────

class _GymTask:
    def __init__(self, task_id: int, env_id: str, label: str = None):
        self.id       = task_id
        self.env_name = label or env_id
        self._env_id  = env_id

    def create_env(self, **kwargs) -> gym.Env:
        return gym.make(self._env_id, **kwargs)


class _ParameterizedGymTask(_GymTask):
    def __init__(self, task_id, base_env_id, *, label=None,
                 body_mass_scale=None, geom_friction_scale=1.0, gravity=None):
        super().__init__(task_id, base_env_id, label=label or base_env_id)
        self._base_env_id         = base_env_id
        self._body_mass_scale     = body_mass_scale or {}
        self._geom_friction_scale = geom_friction_scale
        self._gravity             = gravity

    def create_env(self, **kwargs) -> gym.Env:
        import mujoco as _mj
        env   = gym.make(self._base_env_id, **kwargs)
        model = env.unwrapped.model
        for name, scale in self._body_mass_scale.items():
            bid = _mj.mj_name2id(model, _mj.mjtObj.mjOBJ_BODY, name)
            if bid >= 0:
                model.body_mass[bid] *= scale
        if self._geom_friction_scale != 1.0:
            model.geom_friction[:] *= self._geom_friction_scale
        if self._gravity is not None:
            model.opt.gravity[:] = self._gravity
        return env


def _make_walker2d_tasks():
    return [
        _ParameterizedGymTask(0, "Walker2d-v5", label="Walker2d_baseline"),
        _ParameterizedGymTask(1, "Walker2d-v5", label="Walker2d_heavy",
                              body_mass_scale={"torso": 3.0}),
        _ParameterizedGymTask(2, "Walker2d-v5", label="Walker2d_slippery",
                              geom_friction_scale=0.2),
    ]


def _tasks_from_env_ids(env_ids):
    return [_GymTask(i, eid) for i, eid in enumerate(env_ids)]


# ── Load saved task policies ──────────────────────────────────────────────────

def load_task_policies(checkpoint_dir: str, device: str = "cpu") -> dict:
    """Load SB3 SAC models from a checkpoint's task_policies/ sub-folder.

    Returns {task_id: SB3 model} (env not yet bound).
    """
    tp_dir = os.path.join(checkpoint_dir, "task_policies")
    manifest_path = os.path.join(tp_dir, "manifest.json")
    if not os.path.isfile(manifest_path):
        raise FileNotFoundError(f"No manifest.json in {tp_dir}")

    with open(manifest_path) as f:
        manifest = json.load(f)

    policies = {}
    for task_id_str, info in manifest.items():
        algo_name = info.get("algo", "SAC")
        if algo_name != "SAC":
            raise ValueError(
                f"Expected SAC policies, got {algo_name} for task {task_id_str}"
            )
        zip_path = os.path.join(tp_dir, f"task_{task_id_str}.zip")
        if not os.path.isfile(zip_path):
            raise FileNotFoundError(f"Policy zip not found: {zip_path}")
        policies[int(task_id_str)] = SAC.load(zip_path, device=device)
        print(f"  Loaded task {task_id_str} policy from {zip_path}")

    return policies


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate_task_policies(
    task_policies: dict,
    tasks: list,
    n_episodes: int = 10,
    max_steps: int = 1000,
    verbose: bool = True,
) -> dict:
    """Run n_episodes per task deterministically; return {task_id: mean_return}."""
    results = {}
    for task in tasks:
        policy = task_policies.get(task.id)
        if policy is None:
            continue
        returns = []
        for _ in range(n_episodes):
            env = task.create_env()
            obs, _ = env.reset()
            ep_ret = 0.0
            done = False
            t = 0
            while not done and t < max_steps:
                action, _ = policy.predict(obs, deterministic=True)
                obs, r, terminated, truncated, _ = env.step(action)
                ep_ret += float(r)
                done = terminated or truncated
                t += 1
            env.close()
            returns.append(ep_ret)
        mean_ret = float(np.mean(returns))
        results[task.id] = mean_ret
        if verbose:
            print(
                f"  [Eval] {task.env_name} (task {task.id}): "
                f"mean_return={mean_ret:.2f}  (n={n_episodes})"
            )
    return results


# ── Main training loop ────────────────────────────────────────────────────────

def train_plain_sac(
    tasks: list,
    checkpoint_dir: str,
    task_steps: int = 100_000,
    iterations: int = 1,
    eval_episodes: int = 10,
    warmstart: bool = True,
    device: str = "cpu",
    save_dir: str = "results/mujoco_sac_finetune",
    verbose: bool = True,
) -> None:
    os.makedirs(save_dir, exist_ok=True)

    if verbose:
        print("=" * 60)
        print("Plain SAC fine-tune  (no meta influence)")
        print(f"  tasks: {[t.env_name for t in tasks]}")
        print(f"  checkpoint: {checkpoint_dir}")
        print(f"  warmstart: {warmstart}")
        print(f"  task_steps/iter: {task_steps}  iterations: {iterations}")
        print(f"  save_dir: {save_dir}")
        print("=" * 60)

    # Load saved policies (used as warm starts or just to know which tasks exist)
    saved_policies = load_task_policies(checkpoint_dir, device=device)

    # Verify task IDs in checkpoint match the task distribution
    for task in tasks:
        if task.id not in saved_policies:
            raise ValueError(
                f"Task id {task.id} not found in checkpoint "
                f"(available: {sorted(saved_policies.keys())})"
            )

    metrics = {
        "eval_returns": [],       # list of {task_id: mean_return} per iteration
        "phase3_env_returns": [], # list of {env_name: mean_env_return} per iteration
    }

    # current policies (updated each iteration)
    current_policies: dict = {t.id: None for t in tasks}

    # Initial eval of the meta-trained starting point
    if verbose:
        print("\n[Eval] Evaluating meta-trained starting point...")
    for task in tasks:
        policy = saved_policies[task.id]
        policy.set_env(make_vec_env(task.create_env, n_envs=1))
    init_returns = evaluate_task_policies(saved_policies, tasks,
                                          n_episodes=eval_episodes, verbose=verbose)
    metrics["eval_returns"].append(init_returns)
    if verbose:
        print(f"  Mean return (init): {np.mean(list(init_returns.values())):.2f}")

    for iteration in range(iterations):
        if verbose:
            print(f"\n{'─' * 60}")
            print(f"Iteration {iteration + 1}/{iterations}")
            print(f"{'─' * 60}")

        phase3_stats: dict = {}

        for task in tasks:
            if verbose:
                print(
                    f"\n  [SAC] {task.env_name} (task {task.id})  "
                    f"steps={task_steps}"
                )

            def _make_env(t=task):
                return t.create_env()

            vec_env = make_vec_env(_make_env, n_envs=1)
            cb = Phase3TrainingCallback()

            # Pick starting weights
            if warmstart and iteration == 0:
                prior = saved_policies[task.id]
            elif iteration > 0 and current_policies[task.id] is not None:
                prior = current_policies[task.id]
            else:
                prior = None

            if prior is not None:
                with tempfile.TemporaryDirectory() as td:
                    prior.save(os.path.join(td, "prior"))
                    model = SAC.load(
                        os.path.join(td, "prior"), env=vec_env, device=device
                    )
                model.learn(
                    total_timesteps=task_steps,
                    reset_num_timesteps=(iteration == 0 and warmstart),
                    callback=cb,
                )
            else:
                model = SAC("MlpPolicy", vec_env, verbose=0, device=device)
                model.learn(total_timesteps=task_steps, callback=cb)

            vec_env.close()
            current_policies[task.id] = model

            # Save model
            model_path = os.path.join(save_dir, f"task_{task.id}_iter{iteration:03d}")
            model.save(model_path)

            phase3_stats[task.env_name] = {
                "ep_rewards":     cb.ep_rewards,
                "ep_env_rewards": cb.ep_rewards,  # no shaping; env reward == total reward
                "ep_shaping":     cb.ep_shaping,
                "ep_successes":   cb.ep_successes,
                "ep_lengths":     cb.ep_lengths,
                "ep_timesteps":   cb.ep_timesteps,
            }

            if verbose and cb.ep_env_rewards:
                last20 = cb.ep_env_rewards[-20:]
                print(
                    f"    last-20 avg_env_r={np.mean(last20):.3f}"
                    f"  ep_count={len(cb.ep_rewards)}"
                )

        # Plot per-iteration phase3 curves
        plot_phase3_results(phase3_stats, save_dir, iteration=iteration)

        # Track per-iteration env returns
        p3_env_returns = {
            name: float(np.mean(v["ep_env_rewards"][-20:]))
            if v["ep_env_rewards"] else 0.0
            for name, v in phase3_stats.items()
        }
        metrics["phase3_env_returns"].append(p3_env_returns)

        # Evaluate current policies
        if verbose:
            print(f"\n[Eval] Evaluating after iteration {iteration + 1}...")
        iter_returns = evaluate_task_policies(
            current_policies, tasks,
            n_episodes=eval_episodes, verbose=verbose,
        )
        metrics["eval_returns"].append(iter_returns)
        mean_ret = float(np.mean(list(iter_returns.values())))
        if verbose:
            print(f"  Mean return: {mean_ret:.2f}")

    # Save final policies
    final_dir = os.path.join(save_dir, "final_policies")
    os.makedirs(final_dir, exist_ok=True)
    manifest = {}
    for task in tasks:
        model = current_policies[task.id]
        if model is not None:
            mp = os.path.join(final_dir, f"task_{task.id}")
            model.save(mp)
            manifest[str(task.id)] = {"algo": "SAC", "env_name": task.env_name}
    with open(os.path.join(final_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    # Summary JSON
    summary = {
        "checkpoint": checkpoint_dir,
        "warmstart": warmstart,
        "task_steps_per_iter": task_steps,
        "iterations": iterations,
        "tasks": [{"id": t.id, "env_name": t.env_name} for t in tasks],
        "eval_returns": [
            {str(k): v for k, v in d.items()}
            for d in metrics["eval_returns"]
        ],
        "phase3_env_returns": metrics["phase3_env_returns"],
    }
    summary_path = os.path.join(save_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    # Training curves
    plot_training_curves(metrics, save_dir)

    if verbose:
        print(f"\nDone.  Results saved to {save_dir}")
        init = metrics["eval_returns"][0]
        final = metrics["eval_returns"][-1]
        print(f"  Init mean return:  {np.mean(list(init.values())):.2f}")
        print(f"  Final mean return: {np.mean(list(final.values())):.2f}")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune meta-trained task policies with plain SAC."
    )
    parser.add_argument(
        "--checkpoint",
        default="results/mujoco_meta/best",
        help="Path to the checkpoint dir containing task_policies/ "
             "(default: results/mujoco_meta/best)",
    )
    parser.add_argument(
        "--task-set",
        default="walker2d",
        choices=["walker2d"],
        help="Predefined task distribution (default: walker2d). "
             "Ignored when --envs is provided.",
    )
    parser.add_argument(
        "--envs", nargs="+", default=None, metavar="ENV",
        help="Custom Gymnasium env IDs.  Overrides --task-set. "
             "Task IDs are assigned in order (0, 1, …).",
    )
    parser.add_argument(
        "--task-steps", type=int, default=100_000,
        help="SAC training steps per task per iteration (default: 100000)",
    )
    parser.add_argument(
        "--iterations", type=int, default=1,
        help="Number of training iterations (default: 1)",
    )
    parser.add_argument(
        "--eval-episodes", type=int, default=10,
        help="Evaluation episodes per task (default: 10)",
    )
    parser.add_argument(
        "--no-warmstart", action="store_true",
        help="Train from random initialisation instead of the checkpoint weights",
    )
    parser.add_argument(
        "--save-dir", default="results/mujoco_sac_finetune",
        help="Output directory (default: results/mujoco_sac_finetune)",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--no-verbose", action="store_true")
    args = parser.parse_args()

    verbose = not args.no_verbose

    if args.envs:
        tasks = _tasks_from_env_ids(args.envs)
    else:
        tasks = _make_walker2d_tasks()

    # Resolve checkpoint relative to project root if not absolute
    checkpoint = args.checkpoint
    if not os.path.isabs(checkpoint):
        checkpoint = os.path.join(_ROOT, checkpoint)

    save_dir = args.save_dir
    if not os.path.isabs(save_dir):
        save_dir = os.path.join(_ROOT, save_dir)

    train_plain_sac(
        tasks=tasks,
        checkpoint_dir=checkpoint,
        task_steps=args.task_steps,
        iterations=args.iterations,
        eval_episodes=args.eval_episodes,
        warmstart=not args.no_warmstart,
        device=args.device,
        save_dir=save_dir,
        verbose=verbose,
    )


if __name__ == "__main__":
    main()
