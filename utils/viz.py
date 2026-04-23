"""
Visualization utilities for the meta-RL pipeline.

  plot_training_curves(metrics, save_dir)
      Multi-panel figure: phase losses, sub-policy losses, meta-policy returns,
      per-iteration evaluation success rate.

  plot_skeleton_topology(skeleton_data, replay_buffer, save_path)
      PCA-projected witness complex: landmarks, simplices, Morse values,
      critical states highlighted.

  evaluate_policy(meta_policy, sub_policies, skeleton_data, task_dist, ...)
      Roll out the policy on evaluation tasks; return success_rate and avg_return.

  run_demos(meta_policy, sub_policies, skeleton_data, task_dist, save_dir, ...)
      Record demo episodes: render frames → GIF, plot PCA trajectories,
      write per-episode summary JSON.
"""

import os
import json

import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.collections as mc
from matplotlib.tri import Triangulation

try:
    from stable_baselines3.common.callbacks import BaseCallback as _SB3BaseCallback

    _SB3_AVAILABLE = True
except ImportError:
    _SB3_AVAILABLE = False
    _SB3BaseCallback = object

from matplotlib.colors import LinearSegmentedColormap

# ── Palette ────────────────────────────────────────────────────────────────
_P = {
    "blue": "#1B3B6F",
    "red": "#B0433F",
    "gold": "#C9A46C",
    "plum": "#2F1B2E",
    "fill": "#D8C4D6",
    "light": "#E6E6E6",
    "mid": "#66606B",
}
_PALETTE_CYCLE = [_P["blue"], _P["red"], _P["gold"], _P["plum"]]


def _c(i: int) -> str:
    """Return palette colour by index, cycling over the 4 primary hues."""
    return _PALETTE_CYCLE[i % 4]


# Diverging: blue → lavender fill → red  (centred at 0; for Morse values)
CMAP_DIV = LinearSegmentedColormap.from_list(
    "pal_div", [_P["blue"], _P["fill"], _P["red"]]
)
# Sequential: light grey → deep plum  (for Φ potential, low→high)
CMAP_SEQ = LinearSegmentedColormap.from_list("pal_seq", [_P["light"], _P["plum"]])


# ── Figure save helper ─────────────────────────────────────────────────────


def _save_fig(fig, path: str) -> None:
    """Save figure as PNG; also write .tex via tikzplotlib or .pgf as fallback."""
    fig.savefig(path, dpi=120, bbox_inches="tight")
    stem = os.path.splitext(path)[0]
    try:
        import tikzplotlib

        tikzplotlib.save(
            stem + ".tex", figure=fig, extra_axis_parameters={"width=\\linewidth"}
        )
    except Exception:
        try:
            fig.savefig(stem + ".pgf", bbox_inches="tight")
        except Exception:
            pass
    plt.close(fig)


# ── Training curves ────────────────────────────────────────────────────────


def plot_training_curves(metrics: dict, save_dir: str) -> None:
    """
    Write one PNG (+.tex) per metric series into save_dir.

    Output files (only written when data is present):
        metrics_phase2_td_loss.png/.tex
        metrics_phase3_success.png/.tex
        metrics_meta_loss.png/.tex  (via plot_meta_loss)
        metrics_eval.png/.tex

    skeleton_train_losses is intentionally omitted.
    """
    os.makedirs(save_dir, exist_ok=True)

    if metrics.get("phase2_losses"):
        fig, ax = plt.subplots(figsize=(6, 4))
        for i, run in enumerate(metrics["phase2_losses"]):
            if run:
                ax.plot(run, color=_c(i), alpha=0.8, label=f"iter {i}")
        ax.set_xlabel("gradient step")
        ax.set_ylabel("MSE loss")
        ax.legend(fontsize=7)
        fig.tight_layout()
        out = os.path.join(save_dir, "metrics_phase2_td_loss.png")
        _save_fig(fig, out)
        print(f"  [Viz] {out}")

    if metrics.get("phase3_success_rates"):
        fig, ax = plt.subplots(figsize=(6, 4))
        p3_sr = metrics["phase3_success_rates"]
        ax.plot(range(len(p3_sr)), p3_sr, marker="o", color=_P["blue"], linewidth=1.8)
        ax.set_ylim(0, 1.05)
        ax.set_xlabel("iteration")
        ax.set_ylabel("avg success rate (all tasks)")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        out = os.path.join(save_dir, "metrics_phase3_success.png")
        _save_fig(fig, out)
        print(f"  [Viz] {out}")

    if metrics.get("phase4_losses"):
        plot_meta_loss(metrics["phase4_losses"], save_dir)

    if metrics.get("eval_success_rates") or metrics.get("eval_returns"):
        fig, ax = plt.subplots(figsize=(6, 4))
        ref = metrics.get("eval_success_rates") or metrics.get("eval_returns")
        iters = list(range(len(ref)))
        if metrics.get("eval_success_rates"):
            ax.plot(iters, metrics["eval_success_rates"], marker="o", color=_P["blue"])
            ax.set_ylabel("success rate", color=_P["blue"])
            ax.tick_params(axis="y", labelcolor=_P["blue"])
        if metrics.get("eval_returns"):
            ax2 = ax.twinx()
            ax2.plot(
                iters,
                metrics["eval_returns"],
                marker="s",
                color=_P["gold"],
                linestyle="--",
            )
            ax2.set_ylabel("avg return", color=_P["gold"])
            ax2.tick_params(axis="y", labelcolor=_P["gold"])
        ax.set_xlabel("iteration")
        fig.tight_layout()
        out = os.path.join(save_dir, "metrics_eval.png")
        _save_fig(fig, out)
        print(f"  [Viz] {out}")


# ── Meta-loss breakdown ────────────────────────────────────────────────────


def plot_meta_loss(phase4_losses: list, save_dir: str) -> None:
    """
    Plot meta-policy gradient loss components as one continuous series across
    all iterations.  Vertical dashed lines mark iteration boundaries;
    small labels at the top of each panel identify the iteration number.

    phase4_losses: list of iterations, each a list of per-epoch dicts with keys
                   total / policy / value / entropy.
    Writes metrics_meta_loss.png to save_dir.
    """
    if not phase4_losses:
        return

    def _extract(run, key):
        out = []
        for v in run:
            if isinstance(v, dict):
                out.append(v.get(key, float("nan")))
            elif key == "total":
                out.append(float(v))
            else:
                out.append(float("nan"))
        return out

    keys = ["total", "policy", "value", "entropy"]
    labels = {
        "total": "Total loss",
        "policy": "Policy loss",
        "value": "Value loss",
        "entropy": "Entropy",
    }
    colors = {
        "total": _P["plum"],
        "policy": _P["blue"],
        "value": _P["gold"],
        "entropy": _P["red"],
    }

    # Concatenate all iterations into one flat series; record boundary positions
    all_y: dict = {k: [] for k in keys}
    boundaries: list = []  # global epoch index where each iteration ends
    offset = 0
    for run in phase4_losses:
        if not run:
            continue
        for k in keys:
            all_y[k].extend(_extract(run, k))
        offset += len(run)
        boundaries.append(offset)

    if not all_y["total"]:
        return

    # The final boundary is just the end of data — don't draw a line there
    iter_boundaries = boundaries[:-1]
    n_epochs = len(all_y["total"])
    x = list(range(n_epochs))

    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    axes = axes.flatten()

    for ax, key in zip(axes, keys):
        y = np.array(all_y[key], dtype=float)

        # Raw trace (thin, transparent)
        ax.plot(x, y, color=colors[key], alpha=0.25, linewidth=0.7)

        # Smoothed trace — window = ~5% of total length, min 5
        w = max(5, n_epochs // 20)
        if w < n_epochs:
            kernel = np.ones(w) / w
            smooth = np.convolve(y, kernel, mode="valid")
            ax.plot(range(w - 1, n_epochs), smooth, color=colors[key], linewidth=2.0)

        # Iteration boundary lines
        for b in iter_boundaries:
            ax.axvline(x=b, color=_P["mid"], linestyle="--", linewidth=0.9, alpha=0.7)

        # Iteration labels just above the x-axis at each boundary
        # ylim = ax.get_ylim()
        # y_label = ylim[0] + (ylim[1] - ylim[0]) * 0.04
        # for i, b in enumerate(iter_boundaries):
        #    ax.text(b + n_epochs * 0.005, y_label,
        #            fontsize=6, color=_P["mid"], va="bottom")

        ax.set_title(labels[key], fontsize=10)
        ax.set_xlabel("epoch", fontsize=8)
        ax.grid(alpha=0.25)

    n_iters = len([r for r in phase4_losses if r])
    # fig.suptitle(f"Meta-policy gradient losses  ({n_iters} iteration(s), "
    #            f"{n_epochs} total epochs)", fontsize=11)
    fig.tight_layout()
    out = os.path.join(save_dir, "metrics_meta_loss.png")
    _save_fig(fig, out)
    print(f"  [Viz] {out}")


# ── Skeleton topology ──────────────────────────────────────────────────────


def plot_skeleton_topology(skeleton_data: dict, replay_buffer, save_path: str) -> None:
    """
    PCA-project all states to 2-D and draw:
      - background scatter of collected states (grey)
      - landmark vertices coloured by Morse value
      - 1-simplices as lines, 2-simplices as transparent triangles
      - critical 0-simplices with red rings
    """
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)

    from sklearn.decomposition import PCA

    landmarks = skeleton_data["landmarks"]  # [L, D] tensor
    simplices = skeleton_data["simplices"]  # dict dim→list[tuple]
    critical_states = skeleton_data["critical_states"]  # {c_id: np.array}
    morse_values = skeleton_data.get("morse_values", {})  # {v: scalar}

    L_np = landmarks.cpu().numpy() if hasattr(landmarks, "cpu") else np.array(landmarks)
    all_states_t = replay_buffer.get_all_states()
    all_np = all_states_t.cpu().numpy()

    D = L_np.shape[1]
    combined = np.vstack([L_np, all_np])

    pca = None
    if D > 2:
        pca = PCA(n_components=2)
        combined_2d = pca.fit_transform(combined)
    else:
        combined_2d = combined[:, :2]

    L_2d = combined_2d[: len(L_np)]
    all_2d = combined_2d[len(L_np) :]

    fig, ax = plt.subplots(figsize=(8, 7))

    # Background states
    ax.scatter(
        all_2d[:, 0],
        all_2d[:, 1],
        s=3,
        c=_P["light"],
        alpha=0.5,
        zorder=1,
        label="states",
    )

    # 2-simplices (triangles)
    for tri in simplices.get(2, []):
        pts = L_2d[list(tri)]
        poly = plt.Polygon(
            pts,
            alpha=0.12,
            facecolor=_P["blue"],
            edgecolor=_P["blue"],
            linewidth=0.5,
            zorder=2,
        )
        ax.add_patch(poly)

    # 1-simplices (edges)
    segs = []
    for edge in simplices.get(1, []):
        segs.append([L_2d[edge[0]], L_2d[edge[1]]])
    if segs:
        lc = mc.LineCollection(
            segs, colors=_P["blue"], linewidths=0.8, alpha=0.6, zorder=3
        )
        ax.add_collection(lc)

    # Landmark vertices coloured by Morse value
    mv_arr = np.array(
        [
            float(morse_values.get((i,), morse_values.get(i, 0.0)))
            for i in range(len(L_np))
        ]
    )
    sc = ax.scatter(
        L_2d[:, 0],
        L_2d[:, 1],
        c=mv_arr,
        cmap=CMAP_DIV,
        s=40,
        edgecolors="k",
        linewidths=0.4,
        zorder=4,
        label="landmarks",
    )
    # Critical states — project centroid arrays, not landmark indices
    c_ids = list(critical_states.keys())
    if c_ids:
        crit_arr = np.stack(
            [np.asarray(critical_states[k], dtype=np.float32) for k in c_ids]
        )  # [K, D_crit]
        crit_d = crit_arr.shape[-1]
        if pca is not None and crit_d == D:
            crit_xy = pca.transform(crit_arr)
        elif pca is not None and crit_d != D:
            # Critical centroids are in a projected subspace (e.g. 8D EE space).
            # Find the nearest landmark in the shared leading dimensions and use
            # that landmark's 2D PCA position as a proxy.
            min_d = min(D, crit_d)
            L_sub = L_np[:, :min_d]        # [L, min_d]
            c_sub = crit_arr[:, :min_d]    # [K, min_d]
            dists = np.linalg.norm(L_sub[:, None] - c_sub[None, :], axis=-1)  # [L, K]
            nearest = dists.argmin(axis=0)  # [K]
            crit_xy = L_2d[nearest]
        else:
            crit_xy = crit_arr[:, :2]
        ax.scatter(
            crit_xy[:, 0],
            crit_xy[:, 1],
            s=200,
            facecolors="none",
            edgecolors=_P["red"],
            linewidths=2.0,
            zorder=5,
        )
    # Unwanted annotations for publication
    # for i, c_id in enumerate(c_ids):
    #    ax.annotate(str(c_id),
    #                xy=crit_xy[i], xytext=(4, 4),
    #                textcoords="offset points", fontsize=8,
    #                color=_P["red"])

    ax.axis("off")
    fig.tight_layout()
    _save_fig(fig, save_path)
    print(f"  [Viz] Skeleton topology saved → {save_path}")

    # Standalone colorbar
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize

    stem = os.path.splitext(save_path)[0]
    cb_path = stem + "_colorbar.png"
    fig_cb, ax_cb = plt.subplots(figsize=(0.55, 3.5))
    fig_cb.colorbar(
        ScalarMappable(
            norm=Normalize(vmin=float(mv_arr.min()), vmax=float(mv_arr.max())),
            cmap=CMAP_DIV,
        ),
        cax=ax_cb,
        label="Marginal Potential",
    )
    fig_cb.tight_layout()
    _save_fig(fig_cb, cb_path)
    print(f"  [Viz] Colorbar → {cb_path}")


# ── Policy evaluation ──────────────────────────────────────────────────────


def _step_with_policy(policy, obs_arr, env):
    """
    Execute one step using either an SB3 model (task_policies) or a SubPolicy.
    Returns (action_np, obs_next_arr, r, done, info).
    """
    if policy is None:
        a_np = env.action_space.sample()
    elif hasattr(policy, "predict"):  # SB3 model
        a_np, _ = policy.predict(obs_arr, deterministic=False)
    else:  # legacy SubPolicy
        obs_t = torch.tensor(obs_arr)
        a_np = policy.get_action(obs_t).cpu().numpy()
    obs_next, r, terminated, truncated, info = env.step(a_np)
    return (
        a_np,
        np.asarray(obs_next, dtype=np.float32).flatten(),
        float(r),
        terminated or truncated,
        info,
    )


def evaluate_policy(
    meta_policy,
    task_policies: dict,
    skeleton_data: dict,
    task_distribution,
    n_episodes: int = 20,
    max_steps: int = 500,
    option_steps: int = 20,
    gamma: float = 0.99,
    device: str = "cpu",
) -> dict:
    """
    Evaluate the trained meta-policy + task policies on n_episodes tasks.
    task_policies: {task_id: SB3 model}  (or legacy {c_id: SubPolicy})
    Returns dict: success_rate, avg_return, per_env (dict).
    """
    c_list = list(skeleton_data["critical_states"].keys())
    c_states = [
        np.asarray(skeleton_data["critical_states"][c], dtype=np.float32)
        for c in c_list
    ]
    successes: list = []
    returns: list = []
    env_results: dict = {}

    for _ in range(n_episodes):
        task = task_distribution.sample()
        env = task.create_env()
        obs, _ = env.reset()
        obs_arr = np.asarray(obs, dtype=np.float32).flatten()
        obs_t = torch.tensor(obs_arr, device=device)
        done = False
        ep_return = 0.0
        t = 0
        success = False

        # Determine whether we're in task-policy mode or legacy sub-policy mode
        use_task_policies = task_policies and hasattr(
            next(iter(task_policies.values())), "predict"
        )

        while not done and t < max_steps:
            with torch.no_grad():
                c_idx = meta_policy(obs_t).sample().item()

            if use_task_policies:
                policy = task_policies.get(task.id)
                c_target = c_states[c_idx]
                for _ in range(option_steps):
                    if done or t >= max_steps:
                        break
                    a_np, obs_arr, r, done, info = _step_with_policy(
                        policy, obs_arr, env
                    )
                    success = success or bool(info.get("success", 0.0) > 0.5)
                    ep_return += (gamma**t) * r
                    obs_t = torch.tensor(obs_arr, device=device)
                    t += 1
                    if np.linalg.norm(obs_arr - c_target) < 0.5:
                        break
            else:
                # Legacy SubPolicy path
                c_id = c_list[c_idx]
                sp = task_policies.get(c_id)
                T_c = 0
                while sp is not None and not sp.is_terminated(obs_t, done, T_c):
                    a_np, obs_arr, r, done, info = _step_with_policy(sp, obs_arr, env)
                    success = success or bool(info.get("success", 0.0) > 0.5)
                    ep_return += (gamma**t) * r
                    obs_t = torch.tensor(obs_arr, device=device)
                    t += 1
                    T_c += 1
                    if done or t >= max_steps:
                        break

        env.close()
        successes.append(float(success))
        returns.append(ep_return)
        env_results.setdefault(task.env_name, []).append(float(success))

    per_env = {k: float(np.mean(v)) for k, v in env_results.items()}
    return {
        "success_rate": float(np.mean(successes)),
        "avg_return": float(np.mean(returns)),
        "per_env": per_env,
    }


# ── Demo runner ────────────────────────────────────────────────────────────


def run_demos(
    meta_policy,
    task_policies: dict,
    skeleton_data: dict,
    task_distribution,
    save_dir: str,
    n_demos: int = 5,
    max_steps: int = 500,
    option_steps: int = 20,
    render: bool = True,
    gamma: float = 0.99,
    device: str = "cpu",
) -> list:
    """
    Run n_demos episodes, saving for each:
      - An animated GIF of rendered frames (if render=True)
      - A 2D PCA trajectory plot showing subgoal selections
      - A JSON summary (success, return, subgoals chosen)
    task_policies: {task_id: SB3 model}  (or legacy {c_id: SubPolicy})
    Returns list of per-episode result dicts.
    """
    import imageio
    from sklearn.decomposition import PCA

    os.makedirs(save_dir, exist_ok=True)

    c_list = list(skeleton_data["critical_states"].keys())
    c_states = [
        np.asarray(skeleton_data["critical_states"][c], dtype=np.float32)
        for c in c_list
    ]
    landmarks_np = skeleton_data["landmarks"].cpu().numpy()

    D = landmarks_np.shape[1]
    pca = PCA(n_components=2).fit(landmarks_np) if D > 2 else None

    def _project(s_np):
        if pca is not None:
            return pca.transform(s_np.reshape(1, -1))[0]
        return s_np[:2]

    def _project_lm(lm_np):
        if pca is not None:
            return pca.transform(lm_np)
        return lm_np[:, :2]

    lm_2d = _project_lm(landmarks_np)
    crit_2d = lm_2d[: len(c_list)] if c_list else np.empty((0, 2))

    use_task_policies = task_policies and hasattr(
        next(iter(task_policies.values())), "predict"
    )

    results = []

    for demo_i in range(n_demos):
        task = task_distribution.sample()

        if render:
            env_cls = task._env_cls
            raw_env = env_cls(render_mode="rgb_array")
            raw_env.set_task(task._mw_task)
            raw_env.reset()
        else:
            raw_env = None

        env = task.create_env()
        obs, _ = env.reset()
        obs_arr = np.asarray(obs, dtype=np.float32).flatten()
        obs_t = torch.tensor(obs_arr, device=device)

        frames = []
        traj_pts = [_project(obs_arr)]
        sg_sequence = []
        ep_return = 0.0
        t = 0
        done = False
        success = False

        while not done and t < max_steps:
            with torch.no_grad():
                c_idx = meta_policy(obs_t).sample().item()
            sg_sequence.append(c_idx)

            if use_task_policies:
                policy = task_policies.get(task.id)
                c_target = c_states[c_idx]
                for _ in range(option_steps):
                    if done or t >= max_steps:
                        break
                    a_np, obs_arr, r, done, info = _step_with_policy(
                        policy, obs_arr, env
                    )
                    success = success or bool(info.get("success", 0.0) > 0.5)
                    ep_return += (gamma**t) * r
                    obs_t = torch.tensor(obs_arr, device=device)
                    traj_pts.append(_project(obs_arr))
                    if render and raw_env is not None:
                        raw_env.step(a_np)
                        frames.append(raw_env.render())
                    t += 1
                    if np.linalg.norm(obs_arr - c_target) < 0.5:
                        break
            else:
                c_id = c_list[c_idx]
                sp = task_policies.get(c_id)
                T_c = 0
                while sp is not None and not sp.is_terminated(obs_t, done, T_c):
                    a_np, obs_arr, r, done, info = _step_with_policy(sp, obs_arr, env)
                    success = success or bool(info.get("success", 0.0) > 0.5)
                    ep_return += (gamma**t) * r
                    obs_t = torch.tensor(obs_arr, device=device)
                    traj_pts.append(_project(obs_arr))
                    if render and raw_env is not None:
                        raw_env.step(a_np)
                        frames.append(raw_env.render())
                    t += 1
                    T_c += 1
                    if done or t >= max_steps:
                        break

        env.close()
        if render and raw_env is not None:
            raw_env.close()

        ep_result = {
            "demo": demo_i,
            "task": task.env_name,
            "task_id": task.id,
            "success": success,
            "return": ep_return,
            "steps": t,
            "subgoals": sg_sequence,
        }
        results.append(ep_result)

        # Save GIF
        if render and frames:
            gif_path = os.path.join(save_dir, f"demo_{demo_i:02d}_{task.env_name}.gif")
            imageio.mimsave(gif_path, frames, fps=15)

        # Trajectory plot
        traj = np.array(traj_pts)
        fig, ax = plt.subplots(figsize=(6, 5))

        # Draw 1-simplices as faint background
        for edge in skeleton_data["simplices"].get(1, []):
            xs = [lm_2d[edge[0], 0], lm_2d[edge[1], 0]]
            ys = [lm_2d[edge[0], 1], lm_2d[edge[1], 1]]
            ax.plot(xs, ys, color=_P["fill"], linewidth=0.5, alpha=0.5, zorder=1)

        ax.scatter(
            lm_2d[:, 0],
            lm_2d[:, 1],
            s=20,
            c=_P["mid"],
            alpha=0.5,
            zorder=2,
            label="landmarks",
        )
        if len(crit_2d):
            ax.scatter(
                crit_2d[:, 0],
                crit_2d[:, 1],
                s=150,
                facecolors="none",
                edgecolors=_P["red"],
                linewidths=1.5,
                zorder=3,
                label="subgoals",
            )

        # Colour trajectory by active subgoal
        step_sg = []
        for sg in sg_sequence:
            step_sg.extend([sg] * max(1, (t // max(len(sg_sequence), 1))))
        step_sg = step_sg[: len(traj) - 1]
        for k in range(len(traj) - 1):
            sg_idx = step_sg[k] if k < len(step_sg) else 0
            col = _c(c_list.index(sg_idx) if sg_idx in c_list else 0)
            ax.plot(
                traj[k : k + 2, 0],
                traj[k : k + 2, 1],
                color=col,
                linewidth=1.2,
                alpha=0.8,
                zorder=4,
            )

        ax.plot(*traj[0], marker="^", ms=10, color=_P["blue"], zorder=5, label="start")
        ax.plot(
            *traj[-1],
            marker="*",
            ms=12,
            color=_P["gold"] if success else _P["red"],
            zorder=5,
            label="end (✓)" if success else "end (✗)",
        )

        ax.legend(fontsize=7, loc="upper right")
        ax.axis("off")
        fig.tight_layout()
        status_tag = "success" if success else "fail"
        traj_path = os.path.join(
            save_dir,
            f"demo_{demo_i:02d}_{task.env_name}_{status_tag}_traj.png",
        )
        _save_fig(fig, traj_path)

        status = "SUCCESS" if success else "fail"
        print(
            f"  [Demo {demo_i}] {task.env_name}  {status}  "
            f"return={ep_return:.3f}  steps={t}"
        )

    # Summary JSON
    summary_path = os.path.join(save_dir, "demo_summary.json")
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)
    success_rate = np.mean([r["success"] for r in results])
    print(f"  [Demo] Success rate: {success_rate:.1%}  summary → {summary_path}")
    return results


# ── Callbacks ─────────────────────────────────────────────────────────────


class BootstrapCallback(_SB3BaseCallback):
    """
    SB3 callback for Phase 0 bootstrap training.

    Captures raw episode returns and lengths from SB3's Monitor wrapper
    (info["episode"]["r"] / info["episode"]["l"]) which are populated for
    every env in a VecEnv at episode end.
    """

    def __init__(self):
        super().__init__(verbose=0)
        self.ep_rewards: list = []
        self.ep_lengths: list = []

    def _on_step(self) -> bool:
        for done, info in zip(
            self.locals.get("dones", []), self.locals.get("infos", [])
        ):
            if done:
                ep = info.get("episode", {})
                if ep:
                    self.ep_rewards.append(float(ep.get("r", 0.0)))
                    self.ep_lengths.append(int(ep.get("l", 0)))
        return True


class Phase3TrainingCallback(_SB3BaseCallback):
    """
    SB3 callback that records per-episode shaped reward, env reward, shaping
    bonus, success flag, and episode length during Phase 3 model.learn().

    Stores lists: ep_rewards, ep_env_rewards, ep_shaping, ep_successes, ep_lengths.
    """

    def __init__(self):
        super().__init__(verbose=0)
        self.ep_rewards: list = []
        self.ep_env_rewards: list = []
        self.ep_shaping: list = []
        self.ep_successes: list = []
        self.ep_lengths: list = []
        self._step_env_r: float = 0.0
        self._step_shaping: float = 0.0

    def _on_step(self) -> bool:
        # Accumulate per-step env reward and shaping bonus (single env assumed)
        infos = self.locals.get("infos", [{}])
        self._step_env_r += float(infos[0].get("r_env", 0.0))
        self._step_shaping += float(infos[0].get("r_shaping", 0.0))

        for done, info in zip(
            self.locals.get("dones", []), self.locals.get("infos", [])
        ):
            if done:
                ep = info.get("episode", {})
                self.ep_rewards.append(float(ep.get("r", 0.0)))
                self.ep_lengths.append(int(ep.get("l", 0)))
                self.ep_successes.append(
                    float(info.get("success", info.get("is_success", 0.0)))
                )
                self.ep_env_rewards.append(self._step_env_r)
                self.ep_shaping.append(self._step_shaping)
                self._step_env_r = 0.0
                self._step_shaping = 0.0
        return True


def plot_phase3_results(
    phase3_stats: dict,
    save_dir: str,
    iteration: int = 0,
) -> None:
    """
    Write three separate PNG (+.tex) files per iteration:
        phase3_iter_{N:03d}_shaped_reward.png/.tex
        phase3_iter_{N:03d}_reward_split.png/.tex
        phase3_iter_{N:03d}_final_perf.png/.tex

    phase3_stats:
        {task_name: {"ep_rewards":     list[float],
                     "ep_env_rewards": list[float],
                     "ep_shaping":     list[float],
                     "ep_successes":   list[float],
                     "ep_lengths":     list[int]}}
    """
    if not phase3_stats:
        return

    os.makedirs(save_dir, exist_ok=True)
    tasks = list(phase3_stats.keys())
    n = len(tasks)
    prefix = os.path.join(save_dir, f"phase3_iter_{iteration:03d}")

    # ── File 1: shaped reward learning curves ─────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 4))
    for i, tname in enumerate(tasks):
        ep_r = phase3_stats[tname]["ep_rewards"]
        if not ep_r:
            continue
        col = _c(i)
        ax.plot(ep_r, color=col, alpha=0.2, linewidth=0.6)
        w = max(1, min(50, len(ep_r) // 5))
        if len(ep_r) >= w:
            sm = np.convolve(ep_r, np.ones(w) / w, mode="valid")
            ax.plot(
                np.arange(w - 1, len(ep_r)), sm, color=col, linewidth=2.0, label=tname
            )
    ax.set_xlabel("episode")
    ax.set_ylabel("shaped reward")
    ax.legend(fontsize=7, loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    _save_fig(fig, f"{prefix}_shaped_reward.png")

    # ── File 2: env reward vs shaping bonus split ──────────────────────────
    fig, ax = plt.subplots(figsize=(7, 4))
    has_shaping = any(
        any(v != 0 for v in phase3_stats[t].get("ep_shaping", [])) for t in tasks
    )
    if has_shaping:
        for i, tname in enumerate(tasks):
            ep_e = phase3_stats[tname].get("ep_env_rewards", [])
            ep_s = phase3_stats[tname].get("ep_shaping", [])
            if not ep_e:
                continue
            col = _c(i)
            w = max(1, min(50, len(ep_e) // 5))
            sm_e = np.convolve(ep_e, np.ones(w) / w, mode="valid")
            sm_s = np.convolve(ep_s, np.ones(w) / w, mode="valid")
            xs = np.arange(w - 1, len(ep_e))
            ax.plot(
                xs, sm_e, color=col, linewidth=1.8, linestyle="-", label=f"{tname} env"
            )
            ax.plot(
                xs,
                sm_s,
                color=col,
                linewidth=1.2,
                linestyle="--",
                alpha=0.7,
                label=f"{tname} shaping",
            )
        ax.set_xlabel("episode")
        ax.set_ylabel("cumulative reward / ep")
        ax.legend(fontsize=6, loc="lower right", ncol=2)
        ax.grid(alpha=0.3)
    else:
        ax.text(
            0.5,
            0.5,
            "No shaping active\n(potential=None)",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=10,
        )
    fig.tight_layout()
    _save_fig(fig, f"{prefix}_reward_split.png")

    # ── File 3: final success rate + avg return (last 20 ep) ──────────────
    succ_rates, avg_rets = [], []
    for tname in tasks:
        ep_s = phase3_stats[tname]["ep_successes"]
        ep_r = phase3_stats[tname]["ep_rewards"]
        succ_rates.append(float(np.mean(ep_s[-20:])) if ep_s else 0.0)
        avg_rets.append(float(np.mean(ep_r[-20:])) if ep_r else 0.0)

    fig, ax = plt.subplots(figsize=(max(4, n * 1.5), 4))
    x = np.arange(n)
    bw = 0.35
    cols = [_c(i) for i in range(n)]
    ax.bar(x - bw / 2, succ_rates, bw, color=cols, alpha=0.85)
    ax2 = ax.twinx()
    ax2.bar(x + bw / 2, avg_rets, bw, color=cols, alpha=0.45, hatch="//")
    ax.set_xticks(x)
    ax.set_xticklabels([t.replace("-", "\n") for t in tasks], fontsize=8)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("success rate (last 20 ep)", color=_P["blue"])
    ax.tick_params(axis="y", labelcolor=_P["blue"])
    ax2.set_ylabel("avg shaped return (last 20 ep)", color=_P["gold"])
    ax2.tick_params(axis="y", labelcolor=_P["gold"])
    ax.grid(axis="y", alpha=0.3)
    leg = [
        mpatches.Patch(facecolor=_P["mid"], alpha=0.85, label="success rate"),
        mpatches.Patch(
            facecolor=_P["mid"], alpha=0.45, hatch="//", label="avg shaped return"
        ),
    ]
    ax.legend(handles=leg, fontsize=7, loc="upper right")
    fig.tight_layout()
    _save_fig(fig, f"{prefix}_final_perf.png")
    print(f"  [Viz] Phase 3 results → {prefix}_*.png")


# ── Per-iteration topology snapshot ───────────────────────────────────────


def save_iteration_visuals(
    skeleton_data: dict, replay_buffer, metrics: dict, save_dir: str, iteration: int
) -> None:
    """Convenience wrapper: save topology + training curves after each iteration."""
    topo_path = os.path.join(save_dir, f"topology_iter_{iteration:03d}.png")
    plot_skeleton_topology(skeleton_data, replay_buffer, topo_path)
    plot_training_curves(metrics, save_dir)
