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


# ── Training curves ────────────────────────────────────────────────────────

def plot_training_curves(metrics: dict, save_dir: str) -> None:
    """
    Plot all collected training metrics and save to <save_dir>/training_curves.png.

    Expected keys in metrics (all optional):
        phase2_losses   list[list[float]]   one inner list per (iteration, subgoal)
        phase3_pi_losses  list[list[float]]
        phase3_v_losses   list[list[float]]
        phase4_returns    list[list[float]]   inner list = per-epoch avg option return
        eval_returns      list[float]         one value per outer iteration
        eval_success_rates list[float]
        skeleton_train_losses list[float]     BackwardValueNet training loss
    """
    os.makedirs(save_dir, exist_ok=True)

    panels = []
    # Determine which subplots to draw
    if metrics.get("skeleton_train_losses"):
        panels.append("skeleton")
    if metrics.get("phase2_losses"):
        panels.append("phase2")
    if metrics.get("phase3_pi_losses") or metrics.get("phase3_v_losses"):
        panels.append("phase3")
    if metrics.get("phase4_returns"):
        panels.append("phase4")
    if metrics.get("eval_success_rates") or metrics.get("eval_returns"):
        panels.append("eval")

    if not panels:
        return

    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
    if n == 1:
        axes = [axes]

    cmap = plt.get_cmap("tab10")

    for ax, panel in zip(axes, panels):
        if panel == "skeleton":
            losses = metrics["skeleton_train_losses"]
            for i, run_losses in enumerate(losses):
                ax.plot(run_losses, color=cmap(i), alpha=0.8,
                        label=f"iter {i}")
            ax.set_title("Skeleton: BackwardValueNet loss")
            ax.set_xlabel("epoch"); ax.set_ylabel("MSE loss")
            if len(losses) > 1:
                ax.legend(fontsize=7)

        elif panel == "phase2":
            for i, run in enumerate(metrics["phase2_losses"]):
                if run:
                    ax.plot(run, color=cmap(i), alpha=0.8, label=f"iter {i}")
            ax.set_title("Phase 2: V_H TD loss")
            ax.set_xlabel("gradient step"); ax.set_ylabel("MSE loss")

        elif panel == "phase3":
            pi_runs = metrics.get("phase3_pi_losses", [])
            v_runs  = metrics.get("phase3_v_losses",  [])
            for i, (pi, v) in enumerate(zip(pi_runs, v_runs)):
                c = cmap(i)
                if pi:
                    ax.plot(pi, color=c, linestyle="-",  alpha=0.8,
                            label=f"π iter {i}")
                if v:
                    ax.plot(v,  color=c, linestyle="--", alpha=0.6,
                            label=f"V iter {i}")
            ax.set_title("Phase 3: Sub-policy losses")
            ax.set_xlabel("gradient step"); ax.set_ylabel("loss")
            if pi_runs:
                ax.legend(fontsize=7)

        elif panel == "phase4":
            for i, run in enumerate(metrics["phase4_returns"]):
                if run:
                    ax.plot(run, color=cmap(i), alpha=0.8, label=f"iter {i}")
            ax.set_title("Phase 4: Meta-policy avg option return")
            ax.set_xlabel("epoch"); ax.set_ylabel("return")

        elif panel == "eval":
            iters = list(range(len(metrics.get("eval_success_rates", []))))
            if metrics.get("eval_success_rates"):
                ax.plot(iters, metrics["eval_success_rates"],
                        marker="o", color="steelblue", label="success rate")
                ax.set_ylabel("success rate", color="steelblue")
                ax.tick_params(axis="y", labelcolor="steelblue")
            if metrics.get("eval_returns"):
                ax2 = ax.twinx()
                ax2.plot(iters, metrics["eval_returns"],
                         marker="s", color="darkorange",
                         linestyle="--", label="avg return")
                ax2.set_ylabel("avg return", color="darkorange")
                ax2.tick_params(axis="y", labelcolor="darkorange")
            ax.set_title("Evaluation (per iteration)")
            ax.set_xlabel("iteration")

    fig.suptitle("Meta-RL Training Metrics", fontsize=12, y=1.01)
    fig.tight_layout()
    out = os.path.join(save_dir, "training_curves.png")
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  [Viz] Training curves saved → {out}")


# ── Skeleton topology ──────────────────────────────────────────────────────

def plot_skeleton_topology(skeleton_data: dict, replay_buffer,
                           save_path: str) -> None:
    """
    PCA-project all states to 2-D and draw:
      - background scatter of collected states (grey)
      - landmark vertices coloured by Morse value
      - 1-simplices as lines, 2-simplices as transparent triangles
      - critical 0-simplices with red rings
    """
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)

    from sklearn.decomposition import PCA

    landmarks       = skeleton_data["landmarks"]                 # [L, D] tensor
    simplices       = skeleton_data["simplices"]                 # dict dim→list[tuple]
    critical_states = skeleton_data["critical_states"]           # {c_id: np.array}
    morse_values    = skeleton_data.get("morse_values", {})      # {v: scalar}

    L_np = landmarks.cpu().numpy() if hasattr(landmarks, "cpu") else np.array(landmarks)
    all_states_t = replay_buffer.get_all_states()
    all_np = all_states_t.cpu().numpy()

    D = L_np.shape[1]
    combined = np.vstack([L_np, all_np])

    if D > 2:
        pca = PCA(n_components=2)
        combined_2d = pca.fit_transform(combined)
    else:
        combined_2d = combined[:, :2]

    L_2d    = combined_2d[:len(L_np)]
    all_2d  = combined_2d[len(L_np):]

    fig, ax = plt.subplots(figsize=(8, 7))

    # Background states
    ax.scatter(all_2d[:, 0], all_2d[:, 1],
               s=3, c="lightgrey", alpha=0.3, zorder=1, label="states")

    # 2-simplices (triangles)
    for tri in simplices.get(2, []):
        pts = L_2d[list(tri)]
        poly = plt.Polygon(pts, alpha=0.12, facecolor="steelblue",
                           edgecolor="steelblue", linewidth=0.5, zorder=2)
        ax.add_patch(poly)

    # 1-simplices (edges)
    segs = []
    for edge in simplices.get(1, []):
        segs.append([L_2d[edge[0]], L_2d[edge[1]]])
    if segs:
        lc = mc.LineCollection(segs, colors="royalblue", linewidths=0.8,
                               alpha=0.6, zorder=3)
        ax.add_collection(lc)

    # Landmark vertices coloured by Morse value
    mv_arr = np.array([
        float(morse_values.get(i, 0.0)) for i in range(len(L_np))
    ])
    sc = ax.scatter(L_2d[:, 0], L_2d[:, 1],
                    c=mv_arr, cmap="RdBu_r", s=40,
                    edgecolors="k", linewidths=0.4, zorder=4,
                    label="landmarks")
    plt.colorbar(sc, ax=ax, label="Morse value", fraction=0.03)

    # Critical states
    c_ids = list(critical_states.keys())
    if c_ids:
        crit_xy = L_2d[c_ids]
        ax.scatter(crit_xy[:, 0], crit_xy[:, 1],
                   s=200, facecolors="none", edgecolors="crimson",
                   linewidths=2.0, zorder=5)
        for c_id in c_ids:
            ax.annotate(f"c{c_id}",
                        xy=L_2d[c_id], xytext=(4, 4),
                        textcoords="offset points", fontsize=8,
                        color="crimson")

    ax.set_title("Witness Complex & Discrete Morse Function (PCA projection)")
    ax.set_xlabel("PC 1"); ax.set_ylabel("PC 2")

    legend_handles = [
        mpatches.Patch(color="lightgrey", label="collected states"),
        mpatches.Patch(color="steelblue", alpha=0.5, label="2-simplices"),
        mpatches.Patch(color="royalblue", label="1-simplices"),
        plt.Line2D([0], [0], marker="o", color="w",
                   markerfacecolor="grey", markersize=7, label="landmarks (Morse)"),
        plt.Line2D([0], [0], marker="o", color="w",
                   markerfacecolor="none", markeredgecolor="crimson",
                   markersize=10, markeredgewidth=2, label="critical states"),
    ]
    ax.legend(handles=legend_handles, fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  [Viz] Skeleton topology saved → {save_path}")


# ── Policy evaluation ──────────────────────────────────────────────────────

def evaluate_policy(
    meta_policy,
    sub_policies: dict,
    skeleton_data: dict,
    task_distribution,
    n_episodes: int = 20,
    max_steps: int  = 500,
    gamma: float    = 0.99,
    device: str     = "cpu",
) -> dict:
    """
    Evaluate the trained meta-policy on n_episodes tasks sampled from
    task_distribution.

    Returns dict: success_rate, avg_return, per_env_success (dict).
    """
    c_list   = list(skeleton_data["critical_states"].keys())
    successes = []
    returns   = []
    env_results: dict = {}

    for _ in range(n_episodes):
        task = task_distribution.sample()
        env  = task.create_env()
        obs, _ = env.reset()
        obs  = torch.tensor(
            np.asarray(obs, dtype=np.float32).flatten(), device=device
        )
        done      = False
        ep_return = 0.0
        t         = 0
        success   = False

        while not done and t < max_steps:
            with torch.no_grad():
                dist  = meta_policy(obs)
                c_idx = dist.sample().item()
            c_id = c_list[c_idx]
            sp   = sub_policies.get(c_id)
            T_c  = 0

            while sp is not None and not sp.is_terminated(obs, done, T_c):
                with torch.no_grad():
                    a = sp.get_action(obs)
                a_np = a.cpu().numpy()
                obs_next, r, terminated, truncated, info = env.step(a_np)
                done    = terminated or truncated
                success = success or bool(info.get("success", 0.0) > 0.5)
                ep_return += (gamma ** t) * r
                obs = torch.tensor(
                    np.asarray(obs_next, dtype=np.float32).flatten(), device=device
                )
                t   += 1
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
        "avg_return":   float(np.mean(returns)),
        "per_env":      per_env,
    }


# ── Demo runner ────────────────────────────────────────────────────────────

def run_demos(
    meta_policy,
    sub_policies: dict,
    skeleton_data: dict,
    task_distribution,
    save_dir: str,
    n_demos: int    = 5,
    max_steps: int  = 500,
    render: bool    = True,
    gamma: float    = 0.99,
    device: str     = "cpu",
) -> list:
    """
    Run n_demos episodes, saving for each:
      - An animated GIF of rendered frames (if render=True)
      - A 2D PCA trajectory plot showing subgoal selections
      - A JSON summary (success, return, subgoals chosen)

    Returns list of per-episode result dicts.
    """
    import imageio
    from sklearn.decomposition import PCA

    os.makedirs(save_dir, exist_ok=True)

    c_list       = list(skeleton_data["critical_states"].keys())
    landmarks_np = skeleton_data["landmarks"].cpu().numpy()

    # Fit PCA on landmarks for consistent projection across episodes
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

    lm_2d     = _project_lm(landmarks_np)
    crit_ids  = list(skeleton_data["critical_states"].keys())
    crit_2d   = lm_2d[crit_ids] if crit_ids else np.empty((0, 2))

    results = []
    cmap    = plt.get_cmap("tab10")

    for demo_i in range(n_demos):
        task = task_distribution.sample()

        if render:
            env_cls = task._env_cls
            raw_env = env_cls(render_mode="rgb_array")
            raw_env.set_task(task._mw_task)
            obs_raw, _ = raw_env.reset()
        else:
            raw_env = None

        env = task.create_env()
        obs, _ = env.reset()
        obs = torch.tensor(np.asarray(obs, dtype=np.float32).flatten(), device=device)

        frames      = []
        traj_pts    = [_project(obs.cpu().numpy())]
        sg_sequence = []
        ep_return   = 0.0
        t           = 0
        done        = False
        success     = False

        while not done and t < max_steps:
            with torch.no_grad():
                dist  = meta_policy(obs)
                c_idx = dist.sample().item()
            c_id = c_list[c_idx]
            sp   = sub_policies.get(c_id)
            sg_sequence.append(int(c_id))
            T_c  = 0

            while sp is not None and not sp.is_terminated(obs, done, T_c):
                with torch.no_grad():
                    a = sp.get_action(obs)
                a_np = a.cpu().numpy()

                obs_next, r, terminated, truncated, info = env.step(a_np)
                done    = terminated or truncated
                success = success or bool(info.get("success", 0.0) > 0.5)
                ep_return += (gamma ** t) * r
                obs_next_t = torch.tensor(
                    np.asarray(obs_next, dtype=np.float32).flatten(), device=device
                )
                traj_pts.append(_project(obs_next_t.cpu().numpy()))

                if render and raw_env is not None:
                    raw_env.step(a_np)
                    frames.append(raw_env.render())

                obs = obs_next_t
                t  += 1; T_c += 1
                if done or t >= max_steps:
                    break

        env.close()
        if render and raw_env is not None:
            raw_env.close()

        ep_result = {
            "demo":     demo_i,
            "task":     task.env_name,
            "task_id":  task.id,
            "success":  success,
            "return":   ep_return,
            "steps":    t,
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
            ax.plot(xs, ys, color="lightblue", linewidth=0.5, alpha=0.5, zorder=1)

        ax.scatter(lm_2d[:, 0], lm_2d[:, 1],
                   s=20, c="grey", alpha=0.5, zorder=2, label="landmarks")
        if len(crit_2d):
            ax.scatter(crit_2d[:, 0], crit_2d[:, 1],
                       s=150, facecolors="none", edgecolors="crimson",
                       linewidths=1.5, zorder=3, label="subgoals")

        # Colour trajectory by active subgoal
        step_sg = []
        for sg in sg_sequence:
            step_sg.extend([sg] * max(1, (t // max(len(sg_sequence), 1))))
        step_sg = step_sg[:len(traj) - 1]
        for k in range(len(traj) - 1):
            sg_idx = step_sg[k] if k < len(step_sg) else 0
            col = cmap(c_list.index(sg_idx) % 10 if sg_idx in c_list else 0)
            ax.plot(traj[k:k+2, 0], traj[k:k+2, 1],
                    color=col, linewidth=1.2, alpha=0.8, zorder=4)

        ax.plot(*traj[0],  marker="^", ms=10, color="green",
                zorder=5, label="start")
        ax.plot(*traj[-1], marker="*", ms=12,
                color="gold" if success else "red",
                zorder=5, label="end (✓)" if success else "end (✗)")

        title_suf = "✓ SUCCESS" if success else "✗ FAIL"
        ax.set_title(f"Demo {demo_i} | {task.env_name} | {title_suf}\n"
                     f"return={ep_return:.3f}  steps={t}")
        ax.set_xlabel("PC 1"); ax.set_ylabel("PC 2")
        ax.legend(fontsize=7, loc="upper right")
        fig.tight_layout()
        traj_path = os.path.join(save_dir, f"demo_{demo_i:02d}_{task.env_name}_traj.png")
        fig.savefig(traj_path, dpi=100, bbox_inches="tight")
        plt.close(fig)

        status = "SUCCESS" if success else "fail"
        print(f"  [Demo {demo_i}] {task.env_name}  {status}  "
              f"return={ep_return:.3f}  steps={t}")

    # Summary JSON
    summary_path = os.path.join(save_dir, "demo_summary.json")
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)
    success_rate = np.mean([r["success"] for r in results])
    print(f"  [Demo] Success rate: {success_rate:.1%}  "
          f"summary → {summary_path}")
    return results


# ── Per-iteration topology snapshot ───────────────────────────────────────

def save_iteration_visuals(skeleton_data: dict, replay_buffer,
                           metrics: dict, save_dir: str,
                           iteration: int) -> None:
    """Convenience wrapper: save topology + training curves after each iteration."""
    topo_path = os.path.join(save_dir, f"topology_iter_{iteration:03d}.png")
    plot_skeleton_topology(skeleton_data, replay_buffer, topo_path)
    plot_training_curves(metrics, save_dir)
