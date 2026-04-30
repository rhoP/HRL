"""
Render skeleton topology + Morse-value colorbar as a single combined figure
for every experiment folder under results/.

For each folder containing both a `skeleton.pkl` and a `replay_buffer.npz`,
write `<folder>/renders/skeleton_topology_with_cbar.png`.

Usage:
    python scripts/render_topology_all.py
    python scripts/render_topology_all.py --results-dir results --out-name topology.png
    python scripts/render_topology_all.py --folders mujoco_single_hopper ant
"""

import argparse
import os
import pickle
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.collections as mc
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import torch
torch.set_default_dtype(torch.float32)

from utils.checkpoint import load_replay_buffer
from utils.viz import _P, CMAP_DIV, _save_fig


def _draw_topology(ax, skeleton_data, replay_buffer):
    """Draw the PCA topology onto `ax`. Returns the Morse-value array used."""
    from sklearn.decomposition import PCA

    landmarks = skeleton_data["landmarks"]
    simplices = skeleton_data["simplices"]
    critical_states = skeleton_data["critical_states"]
    morse_values = skeleton_data.get("morse_values", {})

    L_np = landmarks.cpu().numpy() if hasattr(landmarks, "cpu") else np.array(landmarks)
    all_np = replay_buffer.get_all_states().cpu().numpy()

    D = L_np.shape[1]
    combined = np.vstack([L_np, all_np])
    pca = None
    if D > 2:
        pca = PCA(n_components=2)
        combined_2d = pca.fit_transform(combined)
    else:
        combined_2d = combined[:, :2]
    L_2d = combined_2d[: len(L_np)]
    all_2d = combined_2d[len(L_np):]

    ax.scatter(all_2d[:, 0], all_2d[:, 1], s=3, c=_P["light"],
               alpha=0.5, zorder=1)

    for tri in simplices.get(2, []):
        pts = L_2d[list(tri)]
        ax.add_patch(plt.Polygon(
            pts, alpha=0.12, facecolor=_P["blue"], edgecolor=_P["blue"],
            linewidth=0.5, zorder=2,
        ))

    segs = [[L_2d[e[0]], L_2d[e[1]]] for e in simplices.get(1, [])]
    if segs:
        ax.add_collection(mc.LineCollection(
            segs, colors=_P["blue"], linewidths=0.8, alpha=0.6, zorder=3,
        ))

    mv_arr = np.array([
        float(morse_values.get((i,), morse_values.get(i, 0.0)))
        for i in range(len(L_np))
    ])
    ax.scatter(L_2d[:, 0], L_2d[:, 1], c=mv_arr, cmap=CMAP_DIV,
               s=40, edgecolors="k", linewidths=0.4, zorder=4)

    c_ids = list(critical_states.keys())
    if c_ids:
        crit_arr = np.stack([
            np.asarray(critical_states[k], dtype=np.float32) for k in c_ids
        ])
        crit_d = crit_arr.shape[-1]
        if pca is not None and crit_d == D:
            crit_xy = pca.transform(crit_arr)
        elif pca is not None and crit_d != D:
            min_d = min(D, crit_d)
            dists = np.linalg.norm(
                L_np[:, None, :min_d] - crit_arr[None, :, :min_d], axis=-1,
            )
            crit_xy = L_2d[dists.argmin(axis=0)]
        else:
            crit_xy = crit_arr[:, :2]
        ax.scatter(crit_xy[:, 0], crit_xy[:, 1], s=200,
                   facecolors="none", edgecolors=_P["red"],
                   linewidths=2.0, zorder=5)

    ax.axis("off")
    return mv_arr


def render_combined(skel_path, rb_path, out_path, device="cpu"):
    with open(skel_path, "rb") as f:
        skel = pickle.load(f)
    if not isinstance(skel.get("landmarks"), torch.Tensor):
        skel["landmarks"] = torch.tensor(
            np.asarray(skel["landmarks"], dtype=np.float32), dtype=torch.float32,
        )
    rb = load_replay_buffer(rb_path, device=device)

    fig = plt.figure(figsize=(8.6, 7))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 0.04], wspace=0.05)
    ax = fig.add_subplot(gs[0, 0])
    cax = fig.add_subplot(gs[0, 1])

    mv_arr = _draw_topology(ax, skel, rb)

    vmin, vmax = float(mv_arr.min()), float(mv_arr.max())
    if vmin == vmax:
        vmax = vmin + 1e-6
    fig.colorbar(
        ScalarMappable(norm=Normalize(vmin=vmin, vmax=vmax), cmap=CMAP_DIV),
        cax=cax, label="Marginal Potential",
    )
    _save_fig(fig, out_path)


def iter_run_dirs(results_dir, folders=None):
    if folders:
        for f in folders:
            yield os.path.join(results_dir, f)
        return
    for name in sorted(os.listdir(results_dir)):
        path = os.path.join(results_dir, name)
        if os.path.isdir(path):
            yield path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--folders", nargs="+", default=None,
                        help="Specific subfolders under results/ to render")
    parser.add_argument("--out-name", default="skeleton_topology_with_cbar.png",
                        help="Filename written into <run>/renders/")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if not os.path.isdir(args.results_dir):
        raise FileNotFoundError(args.results_dir)

    n_done, n_skip, n_fail = 0, 0, 0
    for run_dir in iter_run_dirs(args.results_dir, args.folders):
        skel_path = os.path.join(run_dir, "skeleton.pkl")
        rb_path = os.path.join(run_dir, "replay_buffer.npz")
        if not (os.path.exists(skel_path) and os.path.exists(rb_path)):
            n_skip += 1
            continue

        out_dir = os.path.join(run_dir, "renders")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, args.out_name)
        if os.path.exists(out_path) and not args.overwrite:
            print(f"[skip-existing] {out_path}")
            n_skip += 1
            continue

        print(f"[render] {run_dir}")
        try:
            render_combined(skel_path, rb_path, out_path, device=args.device)
            n_done += 1
        except Exception as e:
            print(f"  [fail] {e!r}")
            n_fail += 1

    print(f"\nDone: {n_done} rendered, {n_skip} skipped, {n_fail} failed.")


if __name__ == "__main__":
    main()
