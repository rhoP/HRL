"""
Render environment states at the critical (sub-goal) states identified by the
topology-guided skeleton for a MuJoCo single-task experiment.

Critical states are not persisted by train_mujoco_single.py, so this script
re-runs Phase 1 (skeleton build) from the saved replay buffer and then renders:

  1. A static frame for each critical state (nearest actual observation from
     the replay buffer is used to set the MuJoCo physics state).
  2. An eval rollout video of the final policy with per-step colouring that
     highlights the nearest critical state at every time-step.
  3. A summary strip (one row per critical state) saved as a single PNG.

Usage:
    python scripts/render_mujoco_critical.py \
        --env HalfCheetah-v5 \
        --save-dir results/mujoco_single_cheetah

    # Re-use previously computed skeleton (saved as skeleton.pkl in save-dir):
    python scripts/render_mujoco_critical.py \
        --env HalfCheetah-v5 \
        --save-dir results/mujoco_single_cheetah \
        --load-skeleton

Outputs (all written to <save-dir>/renders/):
    critical_state_<id>.png      — frame at each critical state
    critical_states_strip.png    — composite strip of all critical frames
    rollout_episode.mp4          — annotated rollout video (requires ffmpeg)
    rollout_episode.gif          — fallback animated GIF (imageio)
    rollout_critical_dist.png    — per-step distance to every critical state
    skeleton.pkl                 — re-computed skeleton (for --load-skeleton reuse)
    metrics.png                  — training eval returns curve
"""

import argparse
import os
import pickle
import sys

import gymnasium as gym
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import torch
torch.set_default_dtype(torch.float32)

from stable_baselines3 import SAC
from utils.checkpoint import load_checkpoint, load_replay_buffer, restore_models
from utils.skeleton import build_skeleton
from utils.viz import _P, _c, _save_fig, plot_skeleton_topology


# ── MuJoCo state helpers ───────────────────────────────────────────────────────

# Map state_dim → canonical Gymnasium env id (MuJoCo v5).
_STATE_DIM_TO_ENV = {
    11: "Hopper-v5",
    17: "HalfCheetah-v5",  # also Walker2d-v5; disambiguated by save_dir name
    27: "Ant-v5",
}


def resolve_env_id(env_arg: str | None, save_dir: str, state_dim: int,
                   verbose: bool = True) -> str:
    """Pick the env id from --env, save-dir name, or state_dim (in that order)."""
    if env_arg:
        return env_arg

    name = os.path.basename(os.path.normpath(save_dir)).lower()
    for key, env_id in (("halfcheetah", "HalfCheetah-v5"),
                        ("hopper",      "Hopper-v5"),
                        ("humanoid",      "Humanoid-v5"),
                        ("walker",      "Walker2d-v5"),
                        ("ant",         "Ant-v5")):
        if key in name:
            if verbose:
                print(f"[Auto] env inferred from save-dir → {env_id}")
            return env_id

    if state_dim in _STATE_DIM_TO_ENV:
        env_id = _STATE_DIM_TO_ENV[state_dim]
        if verbose:
            print(f"[Auto] env inferred from state_dim={state_dim} → {env_id}")
        return env_id

    raise ValueError(
        f"Could not auto-detect env (save_dir={save_dir!r}, state_dim={state_dim}). "
        f"Please pass --env explicitly."
    )


def obs_to_qpos_qvel(obs: np.ndarray, env_id: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Reconstruct (qpos, qvel) from a flat observation vector.

    HalfCheetah (obs=17):
        qpos (9) = [rootx=0, obs[0:8]]    (rootx excluded from observation)
        qvel (9) = obs[8:17]

    Hopper (obs=11):
        qpos (6) = [rootx=0, obs[0:5]]
        qvel (6) = obs[5:11]

    Ant (obs=27):
        qpos (15) = [rootx=0, rooty=0, obs[0:13]]
        qvel (14) = obs[13:27]

    Falls back to a generic split for unknown environments.
    """
    env_lower = env_id.lower()
    obs = np.asarray(obs, dtype=np.float64)

    if "halfcheetah" in env_lower:
        qpos = np.zeros(9,  dtype=np.float64)
        qvel = np.zeros(9,  dtype=np.float64)
        qpos[1:] = obs[:8]
        qvel[:]  = obs[8:17]
    elif "hopper" in env_lower:
        qpos = np.zeros(6, dtype=np.float64)
        qvel = np.zeros(6, dtype=np.float64)
        qpos[1:] = obs[:5]
        qvel[:]  = obs[5:11]
    elif "ant" in env_lower:
        qpos = np.zeros(15, dtype=np.float64)
        qvel = np.zeros(14, dtype=np.float64)
        qpos[2:] = obs[:13]
        qvel[:]  = obs[13:27]
    elif "walker" in env_lower:
        qpos = np.zeros(9, dtype=np.float64)
        qvel = np.zeros(9, dtype=np.float64)
        qpos[1:] = obs[:8]
        qvel[:]  = obs[8:17]
    else:
        # Generic: assume obs = qpos[1:] + qvel with equal split
        n = len(obs)
        half = n // 2
        qpos = np.zeros(half + 1, dtype=np.float64)
        qvel = np.zeros(n - half,  dtype=np.float64)
        qpos[1:] = obs[:half]
        qvel[:]  = obs[half:]

    return qpos, qvel


def render_state(env, obs: np.ndarray, env_id: str) -> np.ndarray:
    """Set MuJoCo physics to obs and return an rgb_array frame."""
    qpos, qvel = obs_to_qpos_qvel(obs, env_id)
    try:
        env.unwrapped.set_state(qpos, qvel)
    except Exception:
        env.reset()
    return env.render()


# ── Skeleton helpers ───────────────────────────────────────────────────────────

def build_or_load_skeleton(
    rb,
    state_dim: int,
    action_dim: int,
    save_dir: str,
    load: bool = False,
    landmarks: int = 200,
    gamma: float = 0.999,
    device: str = "cpu",
    verbose: bool = True,
) -> dict:
    skel_path = os.path.join(save_dir, "skeleton.pkl")
    if load and os.path.exists(skel_path):
        if verbose:
            print(f"  [Skeleton] Loading from {skel_path}")
        with open(skel_path, "rb") as f:
            skel = pickle.load(f)
        # Ensure landmarks tensor
        if not isinstance(skel.get("landmarks"), torch.Tensor):
            skel["landmarks"] = torch.tensor(
                np.asarray(skel["landmarks"], dtype=np.float32), dtype=torch.float32
            )
        return skel

    if verbose:
        print(f"  [Skeleton] Building Morse skeleton (landmarks={landmarks}, gamma={gamma})...")
    skel = build_skeleton(
        rb,
        state_dim=state_dim,
        action_dim=action_dim,
        num_landmarks=landmarks,
        gamma=gamma,
        survived_only=True,
        min_task_support=0.0,
        device=device,
        verbose=verbose,
    )
    with open(skel_path, "wb") as f:
        pickle.dump(skel, f)
    if verbose:
        print(f"  [Skeleton] Saved to {skel_path}")
    return skel


def get_critical_states(skeleton: dict) -> dict:
    """Return {id: np.ndarray} of critical state vectors."""
    raw = skeleton.get("meta_subgoals") or skeleton.get("critical_states", {})
    out = {}
    for k, v in raw.items():
        state = v["state"] if isinstance(v, dict) else v
        out[k] = np.asarray(state, dtype=np.float32)
    return out


def nearest_buffer_state(target: np.ndarray, all_states: np.ndarray) -> np.ndarray:
    """Return the actual replay-buffer state closest to target in L2."""
    dists = np.linalg.norm(all_states - target, axis=1)
    return all_states[int(np.argmin(dists))]


# ── Rendering ─────────────────────────────────────────────────────────────────

def render_critical_frames(
    critical_states: dict,
    all_states: np.ndarray,
    env_id: str,
    out_dir: str,
    verbose: bool = True,
) -> list[tuple]:
    """
    For every critical state render one static frame.

    Returns list of (key, frame_np, nearest_obs) for compositing.
    """
    env = gym.make(env_id, render_mode="rgb_array")
    env.reset(seed=0)

    frames_info = []
    for i, (cid, crit_obs) in enumerate(critical_states.items()):
        nearest = nearest_buffer_state(crit_obs, all_states)
        frame   = render_state(env, nearest, env_id)
        fname   = os.path.join(out_dir, f"critical_state_{i:02d}.png")
        plt.imsave(fname, frame)
        frames_info.append((cid, frame, nearest))
        if verbose:
            dist = float(np.linalg.norm(crit_obs - nearest))
            print(f"  [Render] critical {i:02d} (id={cid})  "
                  f"nearest-dist={dist:.4f}  → {fname}")

    env.close()
    return frames_info


def render_strip(
    frames_info: list[tuple],
    out_dir: str,
    env_id: str,
) -> None:
    """Save all critical-state frames as one horizontal strip."""
    if not frames_info:
        return
    n = len(frames_info)
    h, w = frames_info[0][1].shape[:2]
    fig, axes = plt.subplots(1, n, figsize=(n * 3, 3.2))
    if n == 1:
        axes = [axes]
    for ax, (cid, frame, _) in zip(axes, frames_info):
        ax.imshow(frame)
        ax.set_title(f"Subgoal {cid}", fontsize=8)
        ax.axis("off")
    fig.suptitle(f"{env_id} — Critical States", fontsize=10)
    fig.tight_layout()
    out = os.path.join(out_dir, "critical_states_strip.png")
    _save_fig(fig, out)
    print(f"  [Render] Strip → {out}")


def render_rollout(
    model: SAC,
    critical_states: dict,
    env_id: str,
    out_dir: str,
    n_episodes: int = 1,
    max_steps: int = 1000,
    verbose: bool = True,
) -> None:
    """
    Render eval rollout(s) with the final policy.
    Each frame is annotated with the nearest critical state and distance.
    Saves a GIF (and MP4 if ffmpeg is available).
    """
    try:
        import imageio
    except ImportError:
        print("  [Render] imageio not installed — skipping video output.")
        return

    crit_keys  = list(critical_states.keys())
    crit_array = np.stack([critical_states[k] for k in crit_keys])
    n_crit     = len(crit_keys)
    crit_colors = [_c(i) for i in range(n_crit)]

    for ep_idx in range(n_episodes):
        env = gym.make(env_id, render_mode="rgb_array")
        obs, _ = env.reset(seed=ep_idx)
        frames = []
        dist_traces = [[] for _ in range(n_crit)]
        ep_return = 0.0
        done = False
        step = 0

        while not done and step < max_steps:
            frame = env.render()

            # Annotate frame with nearest critical state label
            dists   = np.linalg.norm(crit_array - obs, axis=1)
            nearest = int(np.argmin(dists))
            for ci in range(n_crit):
                dist_traces[ci].append(float(dists[ci]))

            # Overlay text on frame
            import PIL.Image
            import PIL.ImageDraw
            import PIL.ImageFont
            pil_img = PIL.Image.fromarray(frame)
            draw    = PIL.ImageDraw.Draw(pil_img)
            label   = f"step={step}  nearest_subgoal={nearest}  d={dists[nearest]:.2f}"
            draw.text((6, 6), label, fill=(255, 255, 0))
            frames.append(np.array(pil_img))

            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            ep_return += float(reward)
            done  = terminated or truncated
            step += 1

        env.close()

        # Save GIF
        gif_path = os.path.join(out_dir, f"rollout_episode_{ep_idx:02d}.gif")
        imageio.mimsave(gif_path, frames, fps=30)
        if verbose:
            print(f"  [Render] Ep {ep_idx}  return={ep_return:.1f}  "
                  f"steps={step}  gif → {gif_path}")

        # Try MP4 via imageio[ffmpeg]
        try:
            mp4_path = os.path.join(out_dir, f"rollout_episode_{ep_idx:02d}.mp4")
            writer = imageio.get_writer(mp4_path, fps=30)
            for f in frames:
                writer.append_data(f)
            writer.close()
            if verbose:
                print(f"  [Render] mp4 → {mp4_path}")
        except Exception:
            pass

        # Distance-to-critical-states plot
        _plot_dist_traces(dist_traces, crit_keys, crit_colors, out_dir, ep_idx)


def _plot_dist_traces(
    dist_traces: list[list],
    crit_keys: list,
    crit_colors: list[str],
    out_dir: str,
    ep_idx: int,
) -> None:
    """Plot per-step distance from agent to each critical state."""
    if not dist_traces or not dist_traces[0]:
        return
    fig, ax = plt.subplots(figsize=(10, 3))
    for ci, (key, color, trace) in enumerate(zip(crit_keys, crit_colors, dist_traces)):
        ax.plot(trace, color=color, linewidth=1.2, alpha=0.85, label=f"subgoal {ci}")
    ax.set_xlabel("step")
    ax.set_ylabel("L2 distance to subgoal")
    ax.legend(fontsize=7, loc="upper right", ncol=min(4, len(crit_keys)))
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out = os.path.join(out_dir, f"rollout_ep{ep_idx:02d}_critical_dist.png")
    _save_fig(fig, out)
    print(f"  [Render] Distance trace → {out}")


# ── Metrics plot ───────────────────────────────────────────────────────────────

def plot_eval_returns(metrics: dict, out_dir: str) -> None:
    iters   = metrics.get("iterations", [])
    returns = metrics.get("eval_returns", [])
    if not returns:
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(iters, returns, marker="o", color=_P["blue"], linewidth=2)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Mean eval return (10 episodes)")
    ax.set_title("HalfCheetah-v5 topology-guided SAC")
    ax.grid(alpha=0.3)
    for i, (x, y) in enumerate(zip(iters, returns)):
        ax.annotate(f"{y:.0f}", (x, y), textcoords="offset points",
                    xytext=(4, 4), fontsize=8)
    fig.tight_layout()
    out = os.path.join(out_dir, "metrics.png")
    _save_fig(fig, out)
    print(f"  [Render] Metrics → {out}")


# ── Meta-mode helpers ─────────────────────────────────────────────────────────

def find_latest_checkpoint(save_dir: str) -> str | None:
    """Return path to best/ or the highest-numbered iter_XXX/ directory."""
    best = os.path.join(save_dir, "best")
    if os.path.isdir(best) and os.path.exists(os.path.join(best, "skeleton.pkl")):
        return best
    iters = sorted(
        d for d in os.listdir(save_dir)
        if d.startswith("iter_") and os.path.isdir(os.path.join(save_dir, d))
    )
    return os.path.join(save_dir, iters[-1]) if iters else None


def get_task_states(rb, task_id: int) -> np.ndarray:
    """Filter replay-buffer states to those collected for task_id."""
    tids   = np.array(rb._task_ids, dtype=np.int64)
    states = np.stack(rb._states)          # [N, state_dim]
    mask   = tids == task_id
    return states[mask] if mask.any() else states


def render_critical_frames_task(
    critical_states: dict,
    task_states: np.ndarray,
    task,
    out_dir: str,
    verbose: bool = True,
) -> list[tuple]:
    """Like render_critical_frames but creates the env via task.create_env().

    Uses task.env_name (the display label) for obs_to_qpos_qvel so that
    ParameterizedGymTask variants like 'Walker2d_heavy' still hit the
    'walker' branch.
    """
    env = task.create_env(render_mode="rgb_array")
    env.reset(seed=0)

    frames_info = []
    for i, (cid, crit_obs) in enumerate(critical_states.items()):
        nearest = nearest_buffer_state(crit_obs, task_states)
        frame   = render_state(env, nearest, task.env_name)
        fname   = os.path.join(out_dir, f"critical_state_{i:02d}.png")
        plt.imsave(fname, frame)
        frames_info.append((cid, frame, nearest))
        if verbose:
            dist = float(np.linalg.norm(crit_obs - nearest))
            print(f"  [Render] critical {i:02d} (id={cid})  "
                  f"nearest-dist={dist:.4f}  → {fname}")
    env.close()
    return frames_info


def render_meta_rollout(
    meta_policy,
    critical_states: dict,
    task,
    out_dir: str,
    n_episodes: int = 1,
    max_steps: int = 1000,
    device: str = "cpu",
    verbose: bool = True,
) -> None:
    """Render GRU meta-policy rollout(s) on one task with critical-state overlay.

    Produces per-episode GIF/MP4 (if ffmpeg available) and a distance-trace
    plot, mirroring render_rollout() but using the meta-policy's hidden state.
    """
    try:
        import imageio
    except ImportError:
        print("  [Render] imageio not installed — skipping video output.")
        return
    import PIL.Image
    import PIL.ImageDraw

    crit_keys   = list(critical_states.keys())
    n_crit      = len(crit_keys)
    crit_array  = np.stack([critical_states[k] for k in crit_keys]) if n_crit else None
    crit_colors = [_c(i) for i in range(n_crit)]

    for ep_idx in range(n_episodes):
        env    = task.create_env(render_mode="rgb_array")
        obs, _ = env.reset(seed=ep_idx)
        h      = meta_policy.init_hidden(device)

        frames      = []
        dist_traces = [[] for _ in range(n_crit)]
        ep_return   = 0.0
        done        = False
        step        = 0

        with torch.no_grad():
            while not done and step < max_steps:
                frame = env.render()

                s_arr = np.asarray(obs, dtype=np.float32)
                if n_crit > 0:
                    dists   = np.linalg.norm(crit_array - s_arr, axis=1)
                    nearest = int(np.argmin(dists))
                    for ci in range(n_crit):
                        dist_traces[ci].append(float(dists[ci]))
                    d_str = f"{dists[nearest]:.2f}"
                else:
                    nearest = -1
                    d_str   = "n/a"

                pil_img = PIL.Image.fromarray(frame)
                draw    = PIL.ImageDraw.Draw(pil_img)
                draw.text(
                    (6, 6),
                    f"step={step}  nearest_subgoal={nearest}  d={d_str}",
                    fill=(255, 255, 0),
                )
                frames.append(np.array(pil_img))

                a_dist = meta_policy.forward_with_hidden(s_arr, h)
                a      = a_dist.sample()
                a_np   = a.cpu().numpy().flatten()
                obs, reward, terminated, truncated, _ = env.step(a_np)
                ep_return += float(reward)
                done = terminated or truncated
                h    = meta_policy.update_hidden(s_arr, a_np, float(reward), h)
                step += 1

        env.close()

        prefix   = f"{task.env_name}_ep{ep_idx:02d}"
        gif_path = os.path.join(out_dir, f"{prefix}_rollout.gif")
        imageio.mimsave(gif_path, frames, fps=30)
        if verbose:
            print(f"  [Render] ep {ep_idx}  return={ep_return:.1f}  "
                  f"steps={step}  → {gif_path}")

        try:
            mp4_path = os.path.join(out_dir, f"{prefix}_rollout.mp4")
            writer   = imageio.get_writer(mp4_path, fps=30)
            for f in frames:
                writer.append_data(f)
            writer.close()
            if verbose:
                print(f"  [Render] mp4 → {mp4_path}")
        except Exception:
            pass

        _plot_dist_traces(dist_traces, crit_keys, crit_colors, out_dir, ep_idx)


def run_meta(args) -> None:
    """Meta-mode entry point.

    Loads a MetaPolicy checkpoint produced by train_mujoco_meta.py, then for
    each task (or --task-id) renders critical-state frames and GRU rollout
    videos.  Outputs are written to <save-dir>/renders/<task_name>/.
    """
    import json

    # Import task classes from the training script
    _scripts = os.path.join(_ROOT, "scripts")
    if _scripts not in sys.path:
        sys.path.insert(0, _scripts)
    from train_mujoco_meta import GymTaskDistribution, make_walker2d_task_dist

    save_dir = args.save_dir
    out_dir  = args.out_dir or os.path.join(save_dir, "renders")
    os.makedirs(out_dir, exist_ok=True)

    # ── Task distribution ──────────────────────────────────────────────────
    if args.envs:
        task_dist = GymTaskDistribution.from_env_ids(args.envs)
    else:
        task_dist = make_walker2d_task_dist()

    tasks = list(task_dist.tasks)
    if args.task_id is not None:
        tasks = [t for t in tasks if t.id == args.task_id]
        if not tasks:
            raise ValueError(f"--task-id {args.task_id} not found in task distribution "
                             f"(available: {[t.id for t in task_dist.tasks]})")

    # ── Replay buffer ──────────────────────────────────────────────────────
    rb_path = os.path.join(save_dir, "replay_buffer.npz")
    if not os.path.exists(rb_path):
        raise FileNotFoundError(f"Replay buffer not found: {rb_path}")
    print(f"[Load] Replay buffer: {rb_path}")
    rb         = load_replay_buffer(rb_path, device=args.device)
    state_dim  = rb.state_dim
    action_dim = rb.action_dim
    print(f"       {len(rb)} transitions  state_dim={state_dim}  action_dim={action_dim}")

    # ── Checkpoint ─────────────────────────────────────────────────────────
    if args.checkpoint_iter == "best":
        ckpt_dir = find_latest_checkpoint(save_dir)
    else:
        ckpt_dir = os.path.join(save_dir, f"iter_{int(args.checkpoint_iter):03d}")
    if not ckpt_dir or not os.path.isdir(ckpt_dir):
        raise FileNotFoundError(
            f"No checkpoint found in {save_dir}. "
            "Run at least one full iteration of train_mujoco_meta.py first."
        )
    print(f"[Load] Checkpoint: {ckpt_dir}")

    ckpt = load_checkpoint(ckpt_dir, device=args.device)
    meta_policy, _, _, skeleton_data = restore_models(
        ckpt, state_dim, action_dim, discrete=False, device=args.device
    )
    if meta_policy is None:
        raise RuntimeError("Checkpoint contains no meta_policy weights.")
    meta_policy.eval()

    # ── Critical states from checkpoint skeleton ───────────────────────────
    critical = get_critical_states(skeleton_data)
    if not critical:
        print("[Warning] No critical states in checkpoint skeleton — "
              "renders will show rollout only (no subgoal overlay).")
    else:
        print(f"[Critical states] {len(critical)} subgoal(s) identified")

    # ── Shared visualisations ──────────────────────────────────────────────
    plot_skeleton_topology(skeleton_data, rb, os.path.join(out_dir, "skeleton_topology.png"))

    metrics_path = os.path.join(save_dir, "metrics.json")
    if os.path.exists(metrics_path):
        with open(metrics_path) as f:
            plot_eval_returns(json.load(f), out_dir)

    # ── Per-task rendering ─────────────────────────────────────────────────
    for task in tasks:
        task_out = os.path.join(out_dir, task.env_name)
        os.makedirs(task_out, exist_ok=True)
        print(f"\n── Task: {task.env_name} (id={task.id}) ──")

        task_states = get_task_states(rb, task.id)
        print(f"  buffer states for this task: {len(task_states)}")

        if critical:
            print("  [1/2] Rendering critical state frames...")
            frames_info = render_critical_frames_task(
                critical, task_states, task, task_out, verbose=True,
            )
            render_strip(frames_info, task_out, task.env_name)

        if not args.no_video:
            print(f"  [2/2] Rendering {args.rollout_eps} meta-policy rollout(s)...")
            render_meta_rollout(
                meta_policy, critical, task, task_out,
                n_episodes=args.rollout_eps,
                max_steps=args.max_steps,
                device=args.device,
                verbose=True,
            )

    print(f"\n[Done] All renders saved to: {out_dir}")
    print("Outputs:")
    for root, _, files in os.walk(out_dir):
        for fname in sorted(files):
            print(f"  {os.path.join(root, fname)}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Render critical-state frames and rollout video for a trained MuJoCo agent."
    )
    # ── Mode ────────────────────────────────────────────────────────────────
    parser.add_argument("--meta", action="store_true",
                        help="Meta mode: load MetaPolicy from train_mujoco_meta checkpoint "
                             "and render per-task GRU rollouts.")
    # ── Shared ──────────────────────────────────────────────────────────────
    parser.add_argument("--save-dir",     default="results/mujoco_meta",
                        help="Checkpoint directory (default: results/mujoco_meta)")
    parser.add_argument("--out-dir",      default=None,
                        help="Output directory (default: <save-dir>/renders)")
    parser.add_argument("--rollout-eps",  type=int, default=3,
                        help="Number of rollout episodes to render per task (default: 1)")
    parser.add_argument("--max-steps",    type=int, default=1000,
                        help="Max steps per rollout episode (default: 1000)")
    parser.add_argument("--no-video",     action="store_true",
                        help="Skip rollout video rendering")
    parser.add_argument("--device",       default="cpu")
    parser.add_argument("--no-verbose",   action="store_true")
    # ── Meta-mode args ───────────────────────────────────────────────────────
    parser.add_argument("--task-set",     default="walker2d",
                        help="Predefined task set for meta mode (default: walker2d)")
    parser.add_argument("--envs",         nargs="+", default=None, metavar="ENV",
                        help="Custom env IDs for meta mode, overrides --task-set")
    parser.add_argument("--task-id",      type=int, default=None,
                        help="Render only this task id (default: all tasks)")
    parser.add_argument("--checkpoint-iter", default="best",
                        help="Which checkpoint to load: integer iter or 'best' (default: best)")
    # ── Single-task-mode args ────────────────────────────────────────────────
    parser.add_argument("--env",          default=None,
                        help="[single-task] Gymnasium env ID used during training "
                             "(auto-detected from save-dir / state_dim if omitted)")
    parser.add_argument("--model",        default=None,
                        help="[single-task] Path to SAC model zip "
                             "(default: <save-dir>/model_final.zip)")
    parser.add_argument("--landmarks",    type=int, default=200,
                        help="[single-task] FPS landmark count (default: 200)")
    parser.add_argument("--gamma",        type=float, default=0.999,
                        help="[single-task] Discount for skeleton rebuild (default: 0.999)")
    parser.add_argument("--load-skeleton", action="store_true",
                        help="[single-task] Load skeleton.pkl from save-dir if present")
    args = parser.parse_args()

    if args.meta:
        run_meta(args)
        return

    # ── Single-task mode (original behaviour) ────────────────────────────────
    verbose  = not args.no_verbose
    save_dir = args.save_dir
    out_dir  = args.out_dir or os.path.join(save_dir, "renders")
    os.makedirs(out_dir, exist_ok=True)

    # ── Load replay buffer ─────────────────────────────────────────────────
    rb_path = os.path.join(save_dir, "replay_buffer.npz")
    if not os.path.exists(rb_path):
        raise FileNotFoundError(f"Replay buffer not found: {rb_path}")
    if verbose:
        print(f"[Load] Replay buffer: {rb_path}")
    rb = load_replay_buffer(rb_path, device=args.device)
    state_dim  = rb.state_dim
    action_dim = rb.action_dim
    if verbose:
        print(f"       {len(rb)} transitions  state_dim={state_dim}  action_dim={action_dim}")

    # ── Resolve env id ─────────────────────────────────────────────────────
    args.env = resolve_env_id(args.env, save_dir, state_dim, verbose=verbose)

    # ── Load model ─────────────────────────────────────────────────────────
    model_path = args.model or os.path.join(save_dir, "model_final.zip")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    if verbose:
        print(f"[Load] Model: {model_path}")
    model = SAC.load(model_path, device=args.device)

    # ── Load metrics ───────────────────────────────────────────────────────
    import json
    metrics_path = os.path.join(save_dir, "metrics.json")
    metrics = {}
    if os.path.exists(metrics_path):
        with open(metrics_path) as f:
            metrics = json.load(f)
    plot_eval_returns(metrics, out_dir)

    # ── Build / load skeleton ──────────────────────────────────────────────
    skeleton = build_or_load_skeleton(
        rb, state_dim, action_dim,
        save_dir=save_dir,
        load=args.load_skeleton,
        landmarks=args.landmarks,
        gamma=args.gamma,
        device=args.device,
        verbose=verbose,
    )

    # Save topology plot into renders dir as well
    topo_out = os.path.join(out_dir, "skeleton_topology.png")
    plot_skeleton_topology(skeleton, rb, topo_out)

    # ── Extract critical states ────────────────────────────────────────────
    critical = get_critical_states(skeleton)
    if not critical:
        print("[Warning] No critical states found in skeleton — "
              "try adjusting --landmarks or running more training iterations.")
        return

    if verbose:
        print(f"[Critical states] {len(critical)} subgoal(s) identified")

    # All actual states from the replay buffer (for nearest-neighbour lookup)
    all_states_np = rb.get_all_states().cpu().numpy()   # [N, state_dim]

    # ── Render critical state frames ───────────────────────────────────────
    if verbose:
        print("\n[Phase 1/3] Rendering critical state frames...")
    frames_info = render_critical_frames(
        critical, all_states_np, args.env, out_dir, verbose=verbose
    )
    render_strip(frames_info, out_dir, args.env)

    # ── Render rollout video ───────────────────────────────────────────────
    if not args.no_video:
        if verbose:
            print(f"\n[Phase 2/3] Rendering {args.rollout_eps} rollout episode(s)...")
        render_rollout(
            model, critical, args.env, out_dir,
            n_episodes=args.rollout_eps,
            max_steps=args.max_steps,
            verbose=verbose,
        )

    # ── Summary ────────────────────────────────────────────────────────────
    if verbose:
        print(f"\n[Done] All renders saved to: {out_dir}")
        best_return = max(metrics.get("eval_returns", [0]))
        print(f"       Best eval return: {best_return:.1f}")
        print(f"       Critical states: {len(critical)}")
        print("\nOutputs:")
        for fname in sorted(os.listdir(out_dir)):
            print(f"  {os.path.join(out_dir, fname)}")


if __name__ == "__main__":
    main()
