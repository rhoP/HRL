"""

"""

import random
from collections import defaultdict
from itertools import combinations

import numpy as np
import torch
import argparse
import sys
import os
import gymnasium as gym

# Allow running from project root or scripts/ directory
_ROOT = os.path.join(os.path.dirname(__file__), "..")
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from utils.collect import Collector
from algos.potential import KNNBackwardEstimator, WitnessBasedSimplexPotential   # noqa: E402
from models.state_action_encoder import (          # noqa: E402
    StateActionEncoder,
    train_state_action_encoder,
)
from utils.viz import _P, CMAP_DIV, CMAP_SEQ      # noqa: E402



class CollectorAdapter:
    """
    Wraps utils/collect.py::Collector to expose the interface expected by
    select_landmarks() (get_all_states) and KNNBackwardEstimator (iter_episodes).
    """

    def __init__(self, collector, device: str = "cpu"):
        self.collector = collector
        self.device = device

    def get_all_states(self) -> torch.Tensor:
        """Returns all observations as a float32 tensor [N, state_dim]."""
        arr = np.array(self.collector.states, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        return torch.tensor(arr, dtype=torch.float32, device=self.device)

    def get_all_actions(self) -> torch.Tensor:
        """Returns all actions as a float32 tensor [N, action_dim]."""
        arr = np.array(self.collector.actions, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        return torch.tensor(arr, dtype=torch.float32, device=self.device)

    def iter_episodes(self):
        """
        Yields one dict per episode:
            {"states": np.ndarray [T+1, state_dim],
             "rewards": np.ndarray [T],
             "dones":   np.ndarray [T, bool]}

        Uses the same index arithmetic as Collector.sample().
        """
        states_arr  = np.array(self.collector.states,  dtype=np.float32)
        rewards_arr = np.array(self.collector.rewards, dtype=np.float32)
        dones_arr   = np.array(self.collector.dones,   dtype=bool)
        ptrs_arr    = np.array(self.collector.ptrs,    dtype=np.int64)

        if states_arr.ndim == 1:
            states_arr = states_arr.reshape(-1, 1)

        for idx in range(len(ptrs_arr)):
            act_start = int(ptrs_arr[idx - 1]) if idx > 0 else 0
            act_end   = int(ptrs_arr[idx])
            obs_start = act_start + idx
            obs_end   = act_end   + idx + 1
            yield {
                "states":  states_arr [obs_start:obs_end],
                "rewards": rewards_arr[act_start:act_end],
                "dones":   dones_arr  [act_start:act_end],
            }


def _fps_indices(points: torch.Tensor, n: int, start_idx: int = None) -> list:
    """
    Farthest-point sampling on `points` [N, D].  Returns a list of n
    selected indices into points.  Shared by select_landmarks and
    select_landmarks_task_aware.

    start_idx: if given, FPS begins at this index instead of a random point.
        Useful when the caller wants to anchor the selection to a specific
        point (e.g. a terminal state) and expand outward from there.
    """
    N = len(points)
    n = min(n, N)
    if n == 0:
        return []
    start   = start_idx if start_idx is not None else random.randint(0, N - 1)
    indices = [start]
    dists   = torch.norm(points - points[start], dim=1)
    for _ in range(n - 1):
        idx = int(dists.argmax().item())
        indices.append(idx)
        new_d = torch.norm(points - points[idx], dim=1)
        dists = torch.minimum(dists, new_d)
    return indices


def _normalize_rewards_array(rewards, reward_normalizer) -> np.ndarray:
    """Normalise a reward array, or return a copy if no normalizer."""
    arr = np.asarray(rewards, dtype=np.float32)
    if reward_normalizer is not None:
        return np.array([reward_normalizer.normalize(float(r)) for r in arr],
                        dtype=np.float32)
    return arr.copy()


def select_landmarks(replay_buffer, num_landmarks=500, distance_metric='euclidean'):
    """Selects landmark states using farthest-point sampling."""
    all_states = replay_buffer.get_all_states()   # [N, state_dim]
    idxs = _fps_indices(all_states, num_landmarks)
    return all_states[idxs]                       # [L, state_dim]


def select_landmarks_task_aware(
    replay_buffer,
    num_landmarks: int = 500,
    reward_normalizer=None,
    gamma: float = 0.99,
    action_window: int = 1,
) -> tuple:
    """
    Two-step task-aware landmark selection (see module docstring).

    Step 1 — Terminal-state return pooling with cluster smoothing.
        For each terminal episode compute the full normalised discounted
        return.  R_0 = mean of those returns.  For every episode, states
        at positions where the per-step backward return >= R_0 are pooled.
        The pool is then clustered via FPS-seeded nearest-neighbour
        assignment; each state's phi_norm is replaced by the mean Φ of its
        cluster, smoothing the landscape without losing spatial diversity.

    Step 2 — Per-task backward state + action-context FPS.
        For each task, episodes are iterated backwards (terminal → start)
        so the FPS is anchored to goal-proximate states first and expands
        toward the episode start.  Context vectors are (state ∥
        mean_action_window) so two fragments with similar states but
        different required actions produce distinct landmarks.  phi_norm is
        set to the normalised backward return at the selected position.

    A final FPS pass over the merged pool produces exactly `num_landmarks`
    diverse landmarks.

    Returns:
        landmark_states : [L, state_dim] float32 tensor
        landmark_meta   : list of L dicts {"task_ids": set[int], "phi_norm": float}
    """
    # ── Step 1: terminal-state return pooling ─────────────────────────────
    terminal_returns = []
    for ep in replay_buffer.iter_episodes():
        if not ep["dones"][-1]:
            continue
        rewards = ep["rewards"]
        T = len(rewards)
        if T == 0:
            continue
        norm_r = _normalize_rewards_array(rewards, reward_normalizer)
        G = float(np.sum(norm_r * (gamma ** np.arange(T - 1, -1, -1, dtype=np.float32))))
        terminal_returns.append(G)

    if not terminal_returns:
        plain = select_landmarks(replay_buffer, num_landmarks)
        meta  = [{"task_ids": set(), "phi_norm": 0.0}] * len(plain)
        return plain, meta

    R0 = float(np.mean(terminal_returns))

    pool_states: list = []
    pool_meta:   list = []
    for ep in replay_buffer.iter_episodes():
        states  = np.asarray(ep["states"],  dtype=np.float32)   # [T+1, D]
        rewards = ep["rewards"]
        task_id = int(ep.get("task_id", 0))
        T       = len(rewards)
        if T == 0:
            continue
        norm_r = _normalize_rewards_array(rewards, reward_normalizer)

        backward_returns = np.zeros(T, dtype=np.float32)
        G = 0.0
        for t in range(T - 1, -1, -1):
            G = float(norm_r[t]) + gamma * G
            backward_returns[t] = G

        hits = np.where(backward_returns >= R0)[0]
        if len(hits) == 0:
            continue
        j_start = int(hits[0])
        for t in range(j_start, T):
            pool_states.append(states[t])
            pool_meta.append({"task_ids": {task_id},
                               "phi_norm": float(backward_returns[t])})

    # ── Step 1 k-NN clustering: smooth phi_norm within spatial clusters ────
    # FPS selects k cluster seeds from the pool; each pool state is assigned
    # to its nearest seed; phi_norm is replaced by the cluster mean, making
    # the Φ landscape smoother without reducing the pool's spatial coverage.
    if len(pool_states) > 1:
        pool_t     = torch.tensor(np.array(pool_states, dtype=np.float32))
        n_clusters = min(num_landmarks, len(pool_states))
        seed_idx   = _fps_indices(pool_t, n_clusters)
        centers    = pool_t[seed_idx]                                   # [K, D]
        assignments = torch.cdist(pool_t, centers).argmin(dim=1).tolist()

        phi_sums = [0.0] * n_clusters
        phi_cnts = [0]   * n_clusters
        for i, cid in enumerate(assignments):
            phi_sums[cid] += pool_meta[i]["phi_norm"]
            phi_cnts[cid] += 1
        phi_means = [phi_sums[c] / max(phi_cnts[c], 1) for c in range(n_clusters)]

        for i, cid in enumerate(assignments):
            pool_meta[i] = {"task_ids": pool_meta[i]["task_ids"],
                            "phi_norm": phi_means[cid]}

    # ── Step 2: per-task backward state + action-context FPS ─────────────
    unique_tasks = sorted({next(iter(m["task_ids"])) for m in pool_meta
                           if m["task_ids"]})
    step2_states: list = []
    step2_meta:   list = []

    for tid in unique_tasks:
        ctx_vecs     = []
        orig_states  = []
        phi_norms_s2 = []

        for ep in replay_buffer.iter_episodes():
            if int(ep.get("task_id", 0)) != tid:
                continue
            states  = np.asarray(ep["states"],  dtype=np.float32)   # [T+1, D]
            actions = np.asarray(
                ep.get("actions", np.zeros((len(ep["rewards"]), 1), np.float32)),
                dtype=np.float32,
            )                                                        # [T, A]
            rewards = ep["rewards"]
            T = len(actions)
            if T == 0:
                continue

            norm_r = _normalize_rewards_array(rewards, reward_normalizer)
            backward_returns = np.zeros(T, dtype=np.float32)
            G = 0.0
            for t in range(T - 1, -1, -1):
                G = float(norm_r[t]) + gamma * G
                backward_returns[t] = G

            # Iterate backwards: terminal states enter ctx_vecs first so the
            # FPS seed (index 0) is anchored to a goal-proximate state and
            # expands toward the episode start.
            for t in range(T - 1, -1, -1):
                lo = max(0, t - action_window)
                hi = min(T, t + action_window + 1)
                mean_a = actions[lo:hi].mean(axis=0)
                ctx_vecs.append(np.concatenate([states[t], mean_a]))
                orig_states.append(states[t])
                phi_norms_s2.append(float(backward_returns[t]))

        if not ctx_vecs:
            continue

        ctx_t  = torch.tensor(np.array(ctx_vecs, dtype=np.float32))
        n_task = max(1, min(num_landmarks // max(1, len(unique_tasks)), len(ctx_t)))
        lm_idx = _fps_indices(ctx_t, n_task, start_idx=0)
        for idx in lm_idx:
            step2_states.append(orig_states[idx])
            step2_meta.append({"task_ids": {tid}, "phi_norm": phi_norms_s2[idx]})

    # ── Final FPS over merged pool ────────────────────────────────────────
    all_states_list = pool_states + step2_states
    all_meta_list   = pool_meta   + step2_meta
    if not all_states_list:
        plain = select_landmarks(replay_buffer, num_landmarks)
        meta  = [{"task_ids": set(), "phi_norm": 0.0}] * len(plain)
        return plain, meta

    all_t   = torch.tensor(np.array(all_states_list, dtype=np.float32))
    n_sel   = min(num_landmarks, len(all_t))
    sel_idx = _fps_indices(all_t, n_sel)
    return all_t[sel_idx], [all_meta_list[i] for i in sel_idx]


def build_witness_complex(
    landmarks: torch.Tensor,
    all_states: torch.Tensor,
    nu: int = 2,
    max_dim: int = 2,
    chunk_size: int = 1024,
    state_task_ids=None,
    all_actions=None,
) -> tuple:
    """
    Builds a lazy witness complex on landmarks witnessed by all_states.

    A k-simplex sigma is included iff at least `nu` witness states have
    all vertices of sigma among their (max_dim + nu) nearest landmarks.

    Args:
        state_task_ids: optional int tensor [N] giving the task_id of each
            witness state.  When provided, each accepted simplex is annotated
            with the union of task_ids of its witnesses.
        all_actions: optional float tensor [N, action_dim] giving the action
            taken at each witness state.  When provided, each accepted
            1-simplex (edge) is annotated with the mean action vector of all
            its witnesses, producing meta-action labels for the skeleton.

    Returns:
        simplices          : {dim: [simplex_tuple]}
        simplex_task_ids   : {simplex_tuple: set[int]} — empty sets when
            state_task_ids is None.
        edge_action_labels : {edge_simplex: np.ndarray[action_dim]} — only
            populated for dim-1 simplices when all_actions is provided.
    """
    n_landmarks = len(landmarks)
    k = min(max_dim + nu, n_landmarks)

    landmarks_f  = landmarks.float()
    all_states_f = all_states.float()
    track_tasks   = state_task_ids is not None
    track_actions = all_actions is not None

    actions_np = None
    if track_actions:
        actions_np = (all_actions.cpu().numpy()
                      if isinstance(all_actions, torch.Tensor)
                      else np.asarray(all_actions, dtype=np.float32))

    # Phase 1: top-k landmark neighbours per witness
    neighbor_lists:     list = []
    task_per_witness:   list = []
    action_per_witness: list = []

    for start in range(0, len(all_states_f), chunk_size):
        chunk = all_states_f[start : start + chunk_size]
        dists = torch.cdist(chunk, landmarks_f)
        _, top_k = dists.topk(k, dim=1, largest=False)
        for i, row in enumerate(top_k.tolist()):
            w_abs = start + i
            neighbor_lists.append(tuple(sorted(row)))
            if track_tasks:
                task_per_witness.append(int(state_task_ids[w_abs].item()))
            if track_actions:
                # actions_np may be shorter than all_states when the caller
                # stores T+1 states but only T actions per episode (e.g.
                # CollectorAdapter).  Terminal states beyond the actions array
                # are recorded as None so they still count as witnesses but
                # don't contribute to edge action labels.
                act = actions_np[w_abs] if w_abs < len(actions_np) else None
                action_per_witness.append(act)

    # Phase 2: count witnesses; accumulate task sets and edge action sums
    witness_count:     dict = defaultdict(int)
    simplex_task_sets: dict = defaultdict(set)
    edge_action_sums:  dict = {}
    edge_action_cnts:  dict = defaultdict(int)

    for w_idx, neighbors in enumerate(neighbor_lists):
        tid = task_per_witness[w_idx]   if track_tasks   else None
        act = action_per_witness[w_idx] if track_actions else None
        for d in range(max_dim + 1):
            for sigma in combinations(neighbors, d + 1):
                witness_count[sigma] += 1
                if track_tasks:
                    simplex_task_sets[sigma].add(tid)
                if track_actions and d == 1 and act is not None:
                    if sigma not in edge_action_sums:
                        edge_action_sums[sigma] = np.zeros(len(act), dtype=np.float32)
                    edge_action_sums[sigma] += act
                    edge_action_cnts[sigma] += 1

    # Phase 3: threshold
    simplices:            dict = {d: [] for d in range(max_dim + 1)}
    simplex_task_ids_out: dict = {}
    edge_action_labels:   dict = {}

    for sigma, count in witness_count.items():
        if count >= nu:
            dim = len(sigma) - 1
            if dim <= max_dim:
                simplices[dim].append(sigma)
                simplex_task_ids_out[sigma] = simplex_task_sets.get(sigma, set())
                if track_actions and dim == 1 and edge_action_cnts[sigma] > 0:
                    edge_action_labels[sigma] = (
                        edge_action_sums[sigma] / edge_action_cnts[sigma]
                    )

    return simplices, simplex_task_ids_out, edge_action_labels


# Define $\Phi on simplices$
# For a simplex $\sigma = [v_0, \dots, v_k]$ (a set of landmark states), we define the set $S_\sigma$
# as the union of $\epsilon$-balls around each vertex, or simply as the set of landmark states themselves.
# We approximate $\Phi(S_\sigma)$ by evaluating the backward value function at a reference start state
# $s_0$ (or the minimum over a set of start states):

def compute_simplex_potential(simplex_vertices, landmark_states_np, knn_estimator,
                              task_ids):
    """
    Φ(σ) = mean over vertices of mean over tasks of k-NN backward return.

    landmark_states_np : np.ndarray [L, D]
    knn_estimator      : KNNBackwardEstimator
    task_ids           : list[int]
    """
    if not task_ids:
        return 0.0
    vals = []
    for v in simplex_vertices:
        s = landmark_states_np[v]
        for tid in task_ids:
            vals.append(knn_estimator.phi(s, tid))
    return float(np.mean(vals)) if vals else 0.0


def compute_morse_function(simplices, landmark_states, knn_estimator, task_ids,
                           witness_potential=None):
    """
    Compute the discrete Morse function on all simplices using k-NN potentials.

    phi_values  — potential estimates for every simplex.
      dim 0     : k-NN backward-return at the landmark vertex.
      dim ≥ 1   : witness-based mean Φ when witness_potential is provided,
                  otherwise vertex-mean k-NN (legacy path).

    morse_values — Discrete Morse function used for critical-simplex detection:

        dim 0  f(v) = Φ(v)
        dim 1  f(u→v) = |Φ(v) − Φ(u)|   (gradient magnitude)
        dim ≥2 f(σ) = Φ(σ) − mean{ f(τ) : τ face of σ }

    Parameters
    ----------
    simplices          : dict {dim: [simplex_tuple, ...]}
    landmark_states    : torch.Tensor or np.ndarray [L, D]
    knn_estimator      : KNNBackwardEstimator
    task_ids           : list[int]
    witness_potential  : WitnessBasedSimplexPotential or None
        When provided, phi for all dim ≥ 1 simplices is the mean potential
        of their witness states rather than the vertex-mean k-NN estimate.
    """
    lm_np = (landmark_states.cpu().numpy()
             if hasattr(landmark_states, "cpu") else np.asarray(landmark_states))

    # ── Phase 1: potential for every simplex ──────────────────────────────
    # When witness_potential is provided it handles all dimensions:
    #   dim 0  — knn_estimator.phi at the landmark location
    #   dim ≥1 — mean knn_estimator.phi over witness states covering the simplex
    # Both paths use the same KNNBackwardEstimator, keeping phi values on a
    # consistent scale across dimensions.
    phi_values: dict = {}
    for dim, simplex_list in simplices.items():
        for simplex in simplex_list:
            if witness_potential is not None:
                phi_values[simplex] = witness_potential.get_potential(simplex)
            else:
                phi_values[simplex] = compute_simplex_potential(
                    simplex, lm_np, knn_estimator, task_ids
                )

    # Ensure orphan vertices (present in edges but not as 0-simplices) have
    # phi entries so edge gradients are computed correctly.
    all_verts = {v for sl in simplices.values() for s in sl for v in s}
    for v in all_verts:
        if (v,) not in phi_values:
            if witness_potential is not None:
                phi_values[(v,)] = witness_potential.get_potential((v,))
            else:
                phi_values[(v,)] = compute_simplex_potential(
                    (v,), lm_np, knn_estimator, task_ids
                )

    # ── Phase 2: Discrete Morse function ─────────────────────────────────
    morse_values: dict = {}
    for dim, simplex_list in simplices.items():
        for simplex in simplex_list:
            if dim == 0:
                morse_values[simplex] = phi_values[simplex]
            elif dim == 1:
                u, v = simplex
                morse_values[simplex] = abs(phi_values[(v,)] - phi_values[(u,)])
            else:
                faces      = get_faces(simplex)
                face_morse = [morse_values.get(f, phi_values.get(f, 0.0))
                              for f in faces]
                morse_values[simplex] = (phi_values[simplex]
                                         - sum(face_morse) / len(face_morse))

    return morse_values, phi_values


def get_faces(simplex):
    """Returns all codimension-1 faces of a simplex."""
    faces = []
    for i in range(len(simplex)):
        face = simplex[:i] + simplex[i+1:]
        faces.append(face)
    return faces


def identify_critical_simplices(simplices, morse_values, threshold_percentile=90):
    """
    Identify critical simplices using the Forman discrete Morse theory conditions.

    A simplex c is critical iff BOTH hold:
      (a) No coface β ⊃ c of dimension dim(c)+1 satisfies Φ(β) < Φ(c).
          Equivalently: all cofaces have Φ(β) ≥ Φ(c), or c has no cofaces.
      (b) No face  γ ⊂ c of dimension dim(c)-1 satisfies Φ(γ) > Φ(c).
          Equivalently: all faces have Φ(γ) ≤ Φ(c), or c has no faces.

    Interpretation by dimension (with the edge gradient formula in place):
      dim 0 — vertex is critical when all adjacent edges have higher or equal
               Morse value (i.e. it is a potential minimum — no edge pulls away
               from it downhill).
      dim 1 — edge is critical when both endpoint values ≤ edge gradient AND
               all incident triangles have gradient ≥ edge value (saddle).
      dim 2 — triangle is critical when all edge values ≤ triangle value and
               there are no cofaces (local maximum of the Morse function).

    threshold_percentile: optional magnitude gate — keeps only critical
        simplices whose |Φ| exceeds this percentile of all |Φ| values.
        Useful for pruning near-zero critical simplices that arise from
        numerical flatness.  Set to 0 to return every Forman-critical simplex.
    """
    # ── Build face / coface adjacency ─────────────────────────────────────
    face_dict:   dict = {}
    coface_dict: dict = {}
    for dim, simplex_list in simplices.items():
        for simplex in simplex_list:
            faces = get_faces(simplex)
            face_dict[simplex] = faces
            for face in faces:
                coface_dict.setdefault(face, []).append(simplex)

    # ── Optional magnitude threshold ──────────────────────────────────────
    all_vals = [v for v in morse_values.values() if not (v != v)]  # drop NaN
    mag_threshold = 0.0
    if threshold_percentile > 0 and all_vals:
        mag_threshold = float(np.percentile(np.abs(all_vals), threshold_percentile))

    max_dim  = max(simplices.keys(), default=0)
    critical = {d: [] for d in range(max_dim + 1)}

    for dim, simplex_list in simplices.items():
        for simplex in simplex_list:
            if simplex not in morse_values:
                continue
            val = morse_values[simplex]

            # Condition (a): no coface β with Φ(β) < Φ(c)
            cofaces     = coface_dict.get(simplex, [])
            coface_vals = [morse_values[β] for β in cofaces if β in morse_values]
            cond_a      = (not coface_vals) or (min(coface_vals) >= val)

            # Condition (b): no face γ with Φ(γ) > Φ(c)
            faces     = face_dict.get(simplex, [])
            face_vals = [morse_values[γ] for γ in faces if γ in morse_values]
            cond_b    = (not face_vals) or (max(face_vals) <= val)

            if cond_a and cond_b and abs(val) >= mag_threshold:
                critical[dim].append(simplex)

    return critical


def visualize_morse_complex(
    results: dict,
    save_path: str = None,
    show: bool = True,
) -> None:
    """
    Three-panel figure visualizing the witness complex and Morse function.

    Panel 1 — Φ landscape: landmarks colored by potential Φ, edges and filled
               triangles overlaid, critical simplices ringed in red/blue.
    Panel 2 — Morse values: same layout colored by discrete Morse value
               (diverging RdBu), critical nodes annotated.
    Panel 3 — Φ distribution: histogram of k-NN backward returns across landmark vertices.

    Args:
        results:   dict returned by run_morse_pipeline()
        save_path: file path to save figure (PNG/PDF); None = don't save
        show:      call plt.show() at the end
    """
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    import matplotlib.colors as mcolors
    from matplotlib.patches import Polygon as MplPolygon
    from matplotlib.collections import PatchCollection
    from sklearn.decomposition import PCA

    landmarks    = results["landmarks"].cpu().numpy()      # [L, state_dim]
    simplices    = results["simplices"]
    phi_values   = results["phi_values"]
    morse_values = results["morse_values"]
    critical     = results["critical_simplices"]
    train_losses = results["train_losses"]

    L, state_dim = landmarks.shape

    # ── Project landmarks to 2-D ──────────────────────────────────────────
    if state_dim == 1:
        # Sort by state value; spread y using phi so nodes don't all overlap
        phi_arr = np.array([phi_values.get((i,), 0.0) for i in range(L)])
        pos2d = np.column_stack([landmarks[:, 0], phi_arr])
    elif state_dim == 2:
        pos2d = landmarks
    else:
        pca = PCA(n_components=2)
        pos2d = pca.fit_transform(landmarks)

    # ── Collect critical sets ─────────────────────────────────────────────
    critical_verts = set(v for (v,) in critical.get(0, []))
    critical_edges = set(critical.get(1, []))
    critical_tris  = set(critical.get(2, []))

    # ── Helper: draw one complex panel ───────────────────────────────────
    def draw_complex(ax, values_0, values_1, values_2, cmap, title, vmin=None, vmax=None):
        """
        Draw landmarks, edges, triangles on ax colored by the provided value dicts.
        values_0: {simplex_tuple: float} for dim-0 simplices (vertices)
        values_1: {simplex_tuple: float} for dim-1 simplices (edges)
        values_2: {simplex_tuple: float} for dim-2 simplices (triangles)
        """
        all_vals = (
            list(values_0.values())
            + list(values_1.values())
            + list(values_2.values())
        )
        vmin = vmin if vmin is not None else float(np.min(all_vals)) if all_vals else 0
        vmax = vmax if vmax is not None else float(np.max(all_vals)) if all_vals else 1
        if vmin == vmax:
            vmax = vmin + 1e-6
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
        mapper = cm.ScalarMappable(norm=norm, cmap=cmap)

        # Draw 2-simplices (filled triangles)
        tri_patches = []
        tri_colors  = []
        for sigma in simplices.get(2, []):
            i, j, k = sigma
            tri = MplPolygon(pos2d[[i, j, k]], closed=True)
            tri_patches.append(tri)
            tri_colors.append(norm(values_2.get(sigma, 0.0)))
            if sigma in critical_tris:
                ax.fill(pos2d[[i, j, k, i], 0], pos2d[[i, j, k, i], 1],
                        color='none', edgecolor=_P["gold"], linewidth=2, zorder=4)
        if tri_patches:
            pc = PatchCollection(tri_patches, cmap=cmap, norm=norm,
                                 alpha=0.25, zorder=1)
            pc.set_array(np.array([v for v in
                [values_2.get(s, 0.0) for s in simplices.get(2, [])]]))
            ax.add_collection(pc)

        # Draw 1-simplices (edges)
        for sigma in simplices.get(1, []):
            i, j = sigma
            x = [pos2d[i, 0], pos2d[j, 0]]
            y = [pos2d[i, 1], pos2d[j, 1]]
            val = values_1.get(sigma, 0.0)
            color = mapper.to_rgba(val)
            lw = 2.5 if sigma in critical_edges else 1.0
            ax.plot(x, y, color=color, linewidth=lw, zorder=2, alpha=0.7)
            if sigma in critical_edges:
                ax.plot(x, y, color=_P["gold"], linewidth=3.5, zorder=2, alpha=0.4)

        # Draw 0-simplices (landmark nodes)
        node_vals = np.array([values_0.get((i,), 0.0) for i in range(L)])
        sc = ax.scatter(
            pos2d[:, 0], pos2d[:, 1],
            c=node_vals, cmap=cmap, norm=norm,
            s=120, zorder=5, edgecolors='white', linewidths=0.8,
        )

        # Highlight critical vertices
        for v in critical_verts:
            ax.scatter(pos2d[v, 0], pos2d[v, 1],
                       s=260, facecolors='none', edgecolors=_P["red"],
                       linewidths=2.0, zorder=6)
            ax.annotate(f"c{v}", xy=pos2d[v], xytext=(4, 4),
                        textcoords='offset points', fontsize=7,
                        color=_P["red"], zorder=7)

        plt.colorbar(mapper, ax=ax, shrink=0.7, pad=0.02)
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.set_xlabel("PC 1" if state_dim > 2 else ("State" if state_dim == 1 else "s₀"))
        ax.set_ylabel("Φ (potential)" if state_dim == 1 else
                      ("PC 2" if state_dim > 2 else "s₁"))
        ax.tick_params(labelsize=8)

    # ── Build per-dim value sub-dicts ─────────────────────────────────────
    phi_0 = {s: v for s, v in phi_values.items() if len(s) == 1}
    phi_1 = {s: v for s, v in phi_values.items() if len(s) == 2}
    phi_2 = {s: v for s, v in phi_values.items() if len(s) == 3}

    morse_0 = {s: v for s, v in morse_values.items() if len(s) == 1}
    morse_1 = {s: v for s, v in morse_values.items() if len(s) == 2}
    morse_2 = {s: v for s, v in morse_values.items() if len(s) == 3}

    # ── Layout ────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    draw_complex(axes[0], phi_0, phi_1, phi_2,
                 cmap=CMAP_SEQ,
                 title="Potential Landscape Φ on Witness Complex")

    # Diverging colormap centred at 0 for Morse values
    mv_all = list(morse_values.values())
    mv_abs = max(abs(v) for v in mv_all) if mv_all else 1.0
    draw_complex(axes[1], morse_0, morse_1, morse_2,
                 cmap=CMAP_DIV,
                 title="Discrete Morse Function\n(crimson rings = critical)",
                 vmin=-mv_abs, vmax=mv_abs)

    # Panel 3 — Φ value distribution
    ax3 = axes[2]
    phi_vertex_vals = [phi_values.get((i,), 0.0) for i in range(L)]
    ax3.hist(phi_vertex_vals, bins=min(20, max(5, L // 3)),
             color=_P["blue"], edgecolor='white', linewidth=0.5)
    ax3.set_xlabel("Φ (k-NN backward return)", fontsize=10)
    ax3.set_ylabel("Landmark count", fontsize=10)
    ax3.set_title("Φ Value Distribution\n(landmark vertices)", fontsize=11, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.tick_params(labelsize=8)

    # ── Legend for critical markers ───────────────────────────────────────
    from matplotlib.lines import Line2D
    legend_handles = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='w',
               markeredgecolor=_P["red"], markeredgewidth=2, markersize=10,
               label='Critical simplex'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='w',
               markeredgecolor=_P["gold"], markeredgewidth=2, markersize=10,
               label='Critical edge/triangle'),
    ]
    axes[1].legend(handles=legend_handles, loc='lower right', fontsize=8)

    n_crit = sum(len(v) for v in critical.values())
    plt.suptitle(
        f"Discrete Morse Theory on Witness Complex  "
        f"({L} landmarks, {n_crit} critical simplices)",
        fontsize=13, fontweight='bold',
    )
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved figure to {save_path}")

    if show:
        plt.show()

    plt.close(fig)


# ── Meta-MorseComplex architecture ────────────────────────────────────────

def stratified_terminal_sampling(
    replay_buffer,
    gamma: float = 0.99,
    strata: int  = 5,
) -> list:
    """
    Sample states from multiple return strata to cover the full state space,
    including bottleneck regions.  Returns list of np.ndarray states from
    the midpoint to the end of each trajectory in each stratum.
    """
    episodes     = list(replay_buffer.iter_episodes())
    traj_returns = []
    for ep in episodes:
        if not ep["dones"][-1]:
            traj_returns.append(None); continue
        rw = np.asarray(ep["rewards"], dtype=np.float32)
        T  = len(rw)
        G  = float(np.sum(rw * (gamma ** np.arange(T - 1, -1, -1, dtype=np.float32))))
        traj_returns.append(G)

    valid = [(i, r) for i, r in enumerate(traj_returns) if r is not None]
    if not valid:
        return []

    vals = np.array([r for _, r in valid])
    qs   = np.percentile(vals, np.linspace(0, 100, strata + 1))

    pool = []
    for si in range(strata):
        lo = qs[si]
        hi = qs[si + 1] + (1e-6 if si == strata - 1 else 0.0)
        for ep_idx, ret in valid:
            if lo <= ret < hi:
                ep      = episodes[ep_idx]
                states  = np.asarray(ep["states"], dtype=np.float32)
                T       = len(ep["rewards"])
                mid_idx = T // 2
                for t in range(mid_idx, T):
                    pool.append(states[t])
    return pool


def _check_topology_connectivity(simplices: dict, task_id: int, verbose: bool) -> int:
    """
    Return the number of connected components in the 1-skeleton.
    Prints a warning when > 1 (disconnected complex → likely zero persistent
    critical states; fix with state_projection_fn).
    """
    vertices: set = set()
    for sigma in simplices.get(0, []):
        vertices.update(sigma)
    for sigma in simplices.get(1, []):
        vertices.update(sigma)

    if not vertices:
        if verbose:
            print(f"      WARNING: task {task_id} complex is empty (no vertices).")
        return 0

    adj: dict = {v: [] for v in vertices}
    for u, v in simplices.get(1, []):
        adj[u].append(v)
        adj[v].append(u)

    unvisited = set(vertices)
    n_components = 0
    while unvisited:
        n_components += 1
        stack = [next(iter(unvisited))]
        while stack:
            node = stack.pop()
            if node in unvisited:
                unvisited.discard(node)
                stack.extend(adj[node])

    if n_components > 1 and verbose:
        n_edges = len(simplices.get(1, []))
        print(f"      WARNING: task {task_id} complex has {n_components} disconnected "
              f"components ({len(vertices)} vertices, {n_edges} edges). "
              f"Persistent critical states will likely be 0 — check state_projection_fn.")

    return n_components


def _build_task_complex(
    task_id: int,
    landmark_states: torch.Tensor,
    replay_buffer,
    sa_encoder,
    nu: int                  = 1,
    max_dim: int             = 1,
    chunk_size: int          = 1024,
    device: str              = "cpu",
    state_projection_fn      = None,
    task_episodes: list      = None,
) -> tuple:
    """
    Build witness complex for one task using SA-embedding-based neighbor distances.

    When sa_encoder is not None, both witness and landmark positions are
    projected into the joint (state, action) embedding space so that edges
    are only accepted when the action context is also consistent.

    When state_projection_fn is provided and sa_encoder is None, witness and
    landmark states are projected before distance computation.  Landmarks are
    still stored in full-dimensional space; projection only affects neighbor
    assignment.

    task_episodes: pre-filtered list of episode dicts for this task_id.
        When provided, skips scanning the full replay buffer.
    """
    n_landmarks = len(landmark_states)
    k  = min(max_dim + nu, n_landmarks)
    lm = landmark_states.float()

    # Collect this task's (state, action) pairs
    task_states, task_actions = [], []
    episodes_iter = (
        task_episodes if task_episodes is not None
        else (ep for ep in replay_buffer.iter_episodes()
              if int(ep.get("task_id", 0)) == task_id)
    )
    for ep in episodes_iter:
        states  = np.asarray(ep["states"],  dtype=np.float32)
        actions = np.asarray(
            ep.get("actions", np.zeros((len(ep["rewards"]), 1), np.float32)),
            dtype=np.float32,
        )
        T = len(ep["rewards"])
        for t in range(T):
            task_states.append(states[t])
            task_actions.append(actions[t])

    if not task_states:
        empty_dim = {d: [] for d in range(max_dim + 1)}
        return empty_dim, {}, {}, np.empty((0, landmark_states.shape[1]), dtype=np.float32), []

    ts_t  = torch.tensor(np.array(task_states,  dtype=np.float32), device=device)
    ta_np = np.array(task_actions, dtype=np.float32)

    # Choose query and key embeddings (SA or plain state)
    if sa_encoder is not None:
        ta_t      = torch.tensor(ta_np, device=device)
        act_dim   = ta_np.shape[-1]
        zero_acts = torch.zeros(n_landmarks, act_dim, device=device)
        with torch.no_grad():
            query_embs = sa_encoder(ts_t, ta_t).cpu()
            key_embs   = sa_encoder(lm.to(device), zero_acts).cpu()
    else:
        if state_projection_fn is not None:
            proj_w     = np.stack([state_projection_fn(s) for s in task_states])
            proj_lm    = np.stack([state_projection_fn(r) for r in lm.cpu().numpy()])
            query_embs = torch.tensor(proj_w,  dtype=torch.float32)
            key_embs   = torch.tensor(proj_lm, dtype=torch.float32)
        else:
            query_embs = ts_t.cpu()
            key_embs   = lm.cpu()

    witness_count      = defaultdict(int)
    edge_action_sums   = {}
    edge_action_cnts   = defaultdict(int)
    collected_neighbors: list = []   # per-witness sorted k-nearest landmark indices

    for start in range(0, len(query_embs), chunk_size):
        chunk_q = query_embs[start : start + chunk_size]
        dists   = torch.cdist(chunk_q, key_embs)
        _, top_k = dists.topk(k, dim=1, largest=False)

        for i, row in enumerate(top_k.tolist()):
            w_abs     = start + i
            neighbors = tuple(sorted(row))
            act       = ta_np[w_abs]
            collected_neighbors.append(neighbors)
            for d in range(max_dim + 1):
                for sigma in combinations(neighbors, d + 1):
                    witness_count[sigma] += 1
                    if d == 1:
                        if sigma not in edge_action_sums:
                            edge_action_sums[sigma] = np.zeros(len(act), dtype=np.float32)
                        edge_action_sums[sigma] += act
                        edge_action_cnts[sigma] += 1

    simplices          = {d: [] for d in range(max_dim + 1)}
    simplex_task_ids   = {}
    edge_action_labels = {}

    for sigma, count in witness_count.items():
        if count >= nu:
            dim = len(sigma) - 1
            if dim <= max_dim:
                simplices[dim].append(sigma)
                simplex_task_ids[sigma] = {task_id}
                if dim == 1 and edge_action_cnts[sigma] > 0:
                    edge_action_labels[sigma] = edge_action_sums[sigma] / edge_action_cnts[sigma]

    task_states_arr = np.array(task_states, dtype=np.float32)
    return simplices, simplex_task_ids, edge_action_labels, task_states_arr, collected_neighbors


def _persistent_critical_simplices(
    simplices: dict,
    phi_values: dict,
    persistence_threshold: float = 0.05,
) -> dict:
    """
    Forman-critical simplices filtered by a persistence gap.

    A simplex c survives iff:
      min coface phi ≥ phi(c) + persistence_threshold   (no near-cancelling coface)
      max face  phi ≤ phi(c) - persistence_threshold    (no near-cancelling face)
    """
    coface_dict: dict = {}
    face_dict:   dict = {}
    for dim, simplex_list in simplices.items():
        for sigma in simplex_list:
            faces = get_faces(sigma)
            face_dict[sigma] = faces
            for f in faces:
                coface_dict.setdefault(f, []).append(sigma)

    max_dim  = max(simplices.keys(), default=0)
    critical = {d: [] for d in range(max_dim + 1)}

    for dim, simplex_list in simplices.items():
        for sigma in simplex_list:
            if sigma not in phi_values:
                continue
            val = phi_values[sigma]

            cofaces     = coface_dict.get(sigma, [])
            coface_vals = [phi_values[b] for b in cofaces if b in phi_values]
            cond_a      = (not coface_vals) or (min(coface_vals) >= val + persistence_threshold)

            faces     = face_dict.get(sigma, [])
            face_vals = [phi_values[g] for g in faces if g in phi_values]
            cond_b    = (not face_vals) or (max(face_vals) <= val - persistence_threshold)

            if cond_a and cond_b:
                critical[dim].append(sigma)

    return critical


def meta_critical_states(
    task_critical: dict,
    landmark_states_np: np.ndarray,
    min_task_support: float = 0.6,
    eps: float              = 0.5,
) -> dict:
    """
    Soft intersection of per-task critical states via DBSCAN.

    task_critical: {task_id: {v_id: np.ndarray state}}
    Returns {sg_id: {'state': centroid, 'task_support': float, 'num_tasks': int}}
    """
    try:
        from sklearn.cluster import DBSCAN
    except ImportError:
        raise ImportError("pip install scikit-learn to use meta_critical_states")

    num_tasks = len(task_critical)
    if num_tasks == 0:
        return {}
    threshold = max(1, int(min_task_support * num_tasks))

    all_crit = []
    for tid, crit in task_critical.items():
        for v_id, state in crit.items():
            all_crit.append({"state": state, "task_id": tid})

    if not all_crit:
        return {}

    positions  = np.array([c["state"] for c in all_crit], dtype=np.float32)
    clustering = DBSCAN(eps=eps, min_samples=threshold).fit(positions)

    meta_sgs = {}
    for label in set(clustering.labels_):
        if label == -1:
            continue
        mask         = clustering.labels_ == label
        cluster_pos  = positions[mask]
        unique_tasks = len({all_crit[i]["task_id"] for i, m in enumerate(mask) if m})
        if unique_tasks >= threshold:
            centroid = cluster_pos.mean(axis=0)
            meta_sgs[f"meta_sg_{label}"] = {
                "state":        centroid,
                "task_support": unique_tasks / num_tasks,
                "num_tasks":    unique_tasks,
            }
    return meta_sgs


def compute_meta_centrality(
    simplices: dict,
    candidate_subgoals: dict,
    landmark_states_np: np.ndarray,
    centrality_threshold: float = 0.0,
) -> dict:
    """
    Filter candidate_subgoals by betweenness centrality in the meta-complex graph.
    Requires networkx.  If unavailable, returns candidate_subgoals unchanged.
    """
    try:
        import networkx as nx
    except ImportError:
        return candidate_subgoals

    G = nx.Graph()
    G.add_nodes_from(range(len(landmark_states_np)))
    for sigma in simplices.get(1, []):
        G.add_edge(sigma[0], sigma[1])

    if len(G.edges) == 0:
        return candidate_subgoals

    centrality = nx.betweenness_centrality(G)
    filtered   = {}
    for sg_id, sg_data in candidate_subgoals.items():
        dists     = np.linalg.norm(landmark_states_np - sg_data["state"], axis=1)
        closest_v = int(np.argmin(dists))
        c_val     = float(centrality.get(closest_v, 0.0))
        if c_val >= centrality_threshold:
            sg_data = dict(sg_data)
            sg_data["centrality"] = c_val
            filtered[sg_id] = sg_data
    return filtered


def compute_meta_potential(task_potentials: dict) -> dict:
    """
    Z-score–normalise per-task phi dicts and return the mean over shared simplices.
    Only simplices present in >= 2 tasks are included.
    """
    normalized = {}
    for tid, phi_dict in task_potentials.items():
        if not phi_dict:
            normalized[tid] = {}
            continue
        values = np.array(list(phi_dict.values()), dtype=np.float32)
        mean   = float(values.mean())
        std    = float(values.std())
        normalized[tid] = {k: float((v - mean) / (std + 1e-8)) for k, v in phi_dict.items()}

    all_simplices = set().union(*[set(p.keys()) for p in normalized.values()])
    meta_phi      = {}
    for sigma in all_simplices:
        vals = [normalized[t][sigma] for t in task_potentials if sigma in normalized.get(t, {})]
        if len(vals) >= 2:
            meta_phi[sigma] = float(np.mean(vals))
    return meta_phi


def build_meta_morse_complex(
    replay_buffer,
    state_dim: int,
    action_dim: int,
    num_landmarks: int           = 32,
    nu: int                      = 1,
    max_dim: int                 = 1,
    gamma: float                 = 0.99,
    sa_embedding_dim: int        = 32,
    batch_size: int              = 32,
    threshold_percentile: float  = 75.0,
    min_task_support: float      = 0.6,
    persistence_threshold: float = 0.05,
    centrality_threshold: float  = 0.0,
    sa_lr: float                 = 1e-3,
    sa_epochs: int               = 5,
    knn_k: int                   = 10,
    state_projection_fn          = None,
    device: str                  = "cpu",
    verbose: bool                = True,
) -> dict:
    """
    Full Meta-MorseComplex pipeline (Meta-MorseComplex.md).

    Step 1  Stratified terminal sampling → FPS landmark selection.
    Step 2  Per-task SA encoder + witness complex + k-NN backward returns +
            persistent critical simplices.
    Step 3  DBSCAN soft intersection → meta-subgoals → centrality filter
            → z-score meta-potential.

    Returns a dict with both new keys (meta_subgoals, meta_phi,
    knn_estimator, sa_encoder) and backward-compat keys used by
    skeleton.py / MetaPolicy.py (critical_states, phi_critical, …).
    """
    if len(replay_buffer) == 0:
        raise RuntimeError("Replay buffer is empty.")

    # ── Step 1: landmark selection ────────────────────────────────────────
    if verbose:
        print("  [MetaMorse] Step 1: stratified terminal sampling...")
    pool_states = stratified_terminal_sampling(replay_buffer, gamma=gamma, strata=5)

    if not pool_states:
        if verbose:
            print("             No terminal episodes found; using all states for FPS.")
        pool_states = replay_buffer.get_all_states().cpu().numpy().tolist()

    pool_t = torch.tensor(np.array(pool_states, dtype=np.float32))
    n_sel  = min(num_landmarks, len(pool_t))
    # FPS on projected states so landmark coverage reflects the positional
    # subspace rather than goal/task coordinates that inflate inter-task distances.
    if state_projection_fn is not None:
        pool_proj = np.stack([state_projection_fn(s) for s in pool_states])
        lm_idx    = _fps_indices(torch.tensor(pool_proj, dtype=torch.float32), n_sel)
    else:
        lm_idx = _fps_indices(pool_t, n_sel)
    landmarks = pool_t[lm_idx]                   # [L, full_D] — always full-dim
    landmark_meta = [{"task_ids": set(), "phi_norm": 0.0}] * len(landmarks)
    if verbose:
        print(f"             {len(landmarks)} landmarks selected.")

    # ── Train shared SA encoder ───────────────────────────────────────────
    # Use the actual stored action dimension (may be 1 for discrete envs that
    # push integer scalars) rather than the logical action vocab size.
    stored_action_dim = getattr(replay_buffer, "action_dim", None) or action_dim
    sa_encoder = None
    if stored_action_dim and stored_action_dim > 0:
        if verbose:
            print("  [MetaMorse] Training StateActionEncoder...")
        sa_encoder = StateActionEncoder(state_dim, stored_action_dim, sa_embedding_dim).to(device)
        sa_losses  = train_state_action_encoder(
            sa_encoder, replay_buffer,
            lr=sa_lr, n_epochs=sa_epochs, batch_size=batch_size, device=device,
        )
        if verbose and sa_losses:
            print(f"             SA encoder final loss: {sa_losses[-1]:.4f}")

    # ── Pre-index episodes by task_id (single O(N) pass, avoids O(N×T) scans) ──
    _task_episode_cache: dict = {}
    for _ep in replay_buffer.iter_episodes():
        _tid_ep = int(_ep.get("task_id", 0))
        _task_episode_cache.setdefault(_tid_ep, []).append(_ep)

    # ── Step 2: per-task complexes and Morse functions ────────────────────
    unique_tasks = sorted(_task_episode_cache.keys())
    if verbose:
        print(f"  [MetaMorse] Step 2: per-task analysis ({len(unique_tasks)} task(s))...")

    landmark_states = landmarks.to(device)
    lm_np           = landmark_states.cpu().numpy()

    # ── Shared k-NN backward estimator (all tasks) ────────────────────────
    knn_estimator = KNNBackwardEstimator(replay_buffer, gamma=gamma, k=knn_k)
    if verbose:
        n_tasks_with_data = len(knn_estimator.all_task_ids())
        print(f"  [MetaMorse] k-NN estimator built ({n_tasks_with_data} task(s) with terminal episodes).")
        stats = knn_estimator.back_ret_stats()
        for tid, s in stats.items():
            flag = "  ← FLAT phi, check terminal coverage" if s["std"] < 0.01 else ""
            print(f"    task {tid}: n={s['n']:5d} entries  "
                  f"back_ret mean={s['mean']:+.4f}  std={s['std']:.4f}{flag}")

    task_phi_values:  dict = {}
    task_critical:    dict = {}

    # Per-task complex cache: {tid: (t_simplices, t_stids, t_eal)}
    # Populated in the analysis loop below and reused by the meta-union loop,
    # eliminating the second full _build_task_complex pass per task.
    _task_complex_cache: dict = {}

    # Kept for first-task representative (used in return dict for visualisation)
    _repr_simplices   = None
    _repr_stids       = None
    _repr_eal         = None
    _repr_morse_vals  = None
    _repr_phi_vals    = None

    for tid in unique_tasks:
        if verbose:
            print(f"    Task {tid}:")

        # ── Task-specific witness complex ─────────────────────────────────
        t_simplices, t_stids, t_eal, t_states, t_neighbors = _build_task_complex(
            tid, landmark_states, replay_buffer,
            sa_encoder=sa_encoder, nu=nu, max_dim=max_dim, device=device,
            state_projection_fn=state_projection_fn,
            task_episodes=_task_episode_cache.get(tid, []),
        )

        # ── Topology connectivity check ───────────────────────────────────
        _check_topology_connectivity(t_simplices, tid, verbose)

        _task_complex_cache[tid] = (t_simplices, t_stids, t_eal)

        # ── Witness-based simplex potential ───────────────────────────────
        wit_assignments = {i: list(t_neighbors[i]) for i in range(len(t_neighbors))}
        wit_pot = WitnessBasedSimplexPotential(
            landmarks=lm_np,
            witness_states=t_states,
            witness_assignments=wit_assignments,
            knn_estimator=knn_estimator,
            task_id=tid,
        )

        # ── Morse function + persistent critical simplices ────────────────
        morse_vals, phi_vals = compute_morse_function(
            t_simplices, landmark_states, knn_estimator, [tid],
            witness_potential=wit_pot,
        )

        # Z-score normalise morse_vals before persistence filtering so that
        # the gap threshold is scale-invariant (fixes the apples-vs-oranges
        # comparison between dim-0 raw phi values and dim-1 gradients).
        mv_arr = np.array(list(morse_vals.values()), dtype=np.float32)
        mv_mean = float(mv_arr.mean()) if len(mv_arr) else 0.0
        mv_std  = float(mv_arr.std())  if len(mv_arr) else 1.0
        mv_std  = mv_std if mv_std > 1e-8 else 1.0
        morse_vals_norm = {s: (v - mv_mean) / mv_std for s, v in morse_vals.items()}

        persistent = _persistent_critical_simplices(
            t_simplices, morse_vals_norm, persistence_threshold=persistence_threshold,
        )

        t_crit_states = {sigma[0]: landmark_states[sigma[0]].cpu().numpy()
                         for sigma in persistent.get(0, [])}
        if verbose:
            print(f"      {len(t_crit_states)} persistent critical state(s).")

        task_phi_values[tid] = phi_vals
        task_critical[tid]   = t_crit_states

        if _repr_simplices is None:
            _repr_simplices  = t_simplices
            _repr_stids      = t_stids
            _repr_eal        = t_eal
            _repr_morse_vals = morse_vals
            _repr_phi_vals   = phi_vals

    # ── Step 3: meta-complex ──────────────────────────────────────────────
    if verbose:
        print("  [MetaMorse] Step 3: computing meta-complex...")

    meta_subgoals = meta_critical_states(
        task_critical, lm_np,
        min_task_support=min_task_support,
    )
    if verbose:
        print(f"             {len(meta_subgoals)} meta-subgoal(s) before centrality filter.")

    # Build union of all task simplices for the meta-complex
    meta_simplices: dict = {d: [] for d in range(max_dim + 1)}
    meta_stids:     dict = {}
    meta_eal:       dict = {}
    seen:           set  = set()

    for tid in unique_tasks:
        t_simplices, t_stids, t_eal = _task_complex_cache[tid]
        for d, sl in t_simplices.items():
            for sigma in sl:
                if sigma not in seen:
                    seen.add(sigma)
                    meta_simplices[d].append(sigma)
                    meta_stids[sigma] = t_stids.get(sigma, set())
                    if d == 1 and sigma in t_eal:
                        meta_eal[sigma] = t_eal[sigma]
                else:
                    meta_stids[sigma] = meta_stids.get(sigma, set()) | t_stids.get(sigma, set())

    meta_subgoals = compute_meta_centrality(
        meta_simplices, meta_subgoals, lm_np,
        centrality_threshold=centrality_threshold,
    )
    if verbose:
        print(f"             {len(meta_subgoals)} meta-subgoal(s) after centrality filter.")

    meta_phi = compute_meta_potential(task_phi_values)

    # ── Backward-compat: critical_states and phi_critical ─────────────────
    # Map meta-subgoal centroids to {sg_id: state_array}
    critical_states: dict = {}
    phi_critical:    dict = {}

    if meta_subgoals:
        for sg_id, sg_data in meta_subgoals.items():
            critical_states[sg_id] = sg_data["state"]
            # Nearest landmark's meta_phi as this subgoal's phi
            dists    = np.linalg.norm(lm_np - sg_data["state"], axis=1)
            closest  = int(np.argmin(dists))
            phi_critical[sg_id] = float(meta_phi.get((closest,), 0.0))
    else:
        # Fall back: union of per-task critical states
        for tid, crit in task_critical.items():
            phi_dict = task_phi_values.get(tid, {})
            for v_id, state in crit.items():
                key = (tid, v_id)
                critical_states[key] = state
                phi_critical[key]    = float(phi_dict.get((v_id,), 0.0))

    if verbose:
        print(f"  [MetaMorse] {len(critical_states)} final subgoal(s).")

    return {
        # Core topology
        "landmarks":            landmarks,
        "landmark_meta":        landmark_meta,
        "simplices":            meta_simplices,
        "simplex_task_ids":     meta_stids,
        "edge_action_labels":   meta_eal,
        # Backward-compat (used by MetaPolicy.py + skeleton.py)
        "critical_states":      critical_states,
        "phi_critical":         phi_critical,
        # New meta-complex data
        "meta_subgoals":        meta_subgoals,
        "meta_phi":             meta_phi,
        "task_critical_states": task_critical,
        "knn_estimator":        knn_estimator,
        "sa_encoder":           sa_encoder,
        # Morse / phi on first task's complex for visualisation
        "morse_values":         _repr_morse_vals or {},
        "phi_values":           _repr_phi_vals   or {},
        "phi_values_denorm":    _repr_phi_vals   or {},
        "train_losses":         [],
    }


def run_morse_pipeline(
    collector,
    state_dim: int,
    num_landmarks: int = 500,
    nu: int = 2,
    max_dim: int = 2,
    gamma: float = 0.99,
    knn_k: int = 10,
    threshold_percentile: float = 90.0,
    device: str = "cpu",
    verbose: bool = True,
) -> dict:
    """
    Full Discrete Morse pipeline from a Collector to critical simplices.

    Returns a dict with keys:
        landmarks, simplices, knn_estimator, morse_values, phi_values,
        critical_simplices, train_losses
    """
    adapter = CollectorAdapter(collector, device=device)

    if verbose:
        print("Step 1: Selecting landmarks via FPS...")
    landmarks = select_landmarks(adapter, num_landmarks=num_landmarks)

    if verbose:
        print(f"  {len(landmarks)} landmarks selected.")
        print("Step 2: Building witness complex...")

    all_states  = adapter.get_all_states()
    all_actions = adapter.get_all_actions()
    simplices, simplex_task_ids, edge_action_labels = build_witness_complex(
        landmarks, all_states, nu=nu, max_dim=max_dim,
        all_actions=all_actions,
    )

    if verbose:
        for d, slist in simplices.items():
            print(f"  dim {d}: {len(slist)} simplices")
        print("Step 3: Building k-NN backward estimator...")

    knn_estimator = KNNBackwardEstimator(adapter, gamma=gamma, k=knn_k)

    if verbose:
        stats = knn_estimator.back_ret_stats()
        s0    = stats.get(0, {"n": 0, "mean": 0.0, "std": 0.0})
        flag  = "  ← FLAT phi, check terminal coverage" if s0["std"] < 0.01 else ""
        print(f"  {s0['n']} state entries indexed  "
              f"back_ret mean={s0['mean']:+.4f}  std={s0['std']:.4f}{flag}")
        print("Step 4: Computing discrete Morse function...")

    landmark_states = landmarks.to(device)
    morse_values, phi_values = compute_morse_function(
        simplices, landmark_states, knn_estimator, [0]
    )

    if verbose:
        print("Step 5: Identifying critical simplices...")

    critical = identify_critical_simplices(
        simplices, morse_values, threshold_percentile=threshold_percentile
    )

    if verbose:
        for d, clist in critical.items():
            print(f"  dim {d}: {len(clist)} critical simplices")

    return {
        "landmarks":           landmarks,
        "landmark_meta":       [{"task_ids": set(), "phi_norm": 0.0}] * len(landmarks),
        "simplices":           simplices,
        "simplex_task_ids":    simplex_task_ids,
        "edge_action_labels":  edge_action_labels,
        "knn_estimator":       knn_estimator,
        "morse_values":        morse_values,
        "phi_values":          phi_values,
        "phi_values_denorm":   phi_values,
        "critical_simplices":  critical,
        "train_losses":        [],
    }


if __name__ == "__main__":
    # ── Per-environment defaults ──────────────────────────────────────────
    # make_kwargs   : passed to gym.make()
    # state_dim     : dimensionality of the observation after flattening
    # n_episodes    : random-policy episodes to collect
    # max_steps     : episode step cap
    # num_landmarks : FPS landmark count
    # nu            : witness threshold
    # max_dim       : max simplex dimension (1 = graph, 2 = triangulated)
    # knn_k         : number of nearest neighbours for k-NN backward estimator
    # threshold_pct : Morse criticality percentile
    ENV_CONFIGS = {
        "FrozenLake-v1": dict(
            make_kwargs={"is_slippery": True},
            state_dim=1, n_episodes=200, max_steps=100,
            num_landmarks=16, nu=1, max_dim=1,
            knn_k=5, threshold_pct=75.0,
        ),
        "LunarLander-v3": dict(
            make_kwargs={},
            state_dim=8, n_episodes=400, max_steps=500,
            num_landmarks=64, nu=2, max_dim=2,
            knn_k=10, threshold_pct=80.0,
        ),
        "CartPole-v1": dict(
            make_kwargs={},
            state_dim=4, n_episodes=300, max_steps=500,
            num_landmarks=32, nu=2, max_dim=2,
            knn_k=10, threshold_pct=80.0,
        ),
        "MountainCar-v0": dict(
            make_kwargs={},
            state_dim=2, n_episodes=500, max_steps=200,
            num_landmarks=32, nu=2, max_dim=2,
            knn_k=10, threshold_pct=80.0,
        ),
        "Acrobot-v1": dict(
            make_kwargs={},
            state_dim=6, n_episodes=300, max_steps=500,
            num_landmarks=48, nu=2, max_dim=1,
            knn_k=10, threshold_pct=80.0,
        ),
    }

    parser = argparse.ArgumentParser(
        description="Discrete Morse theory on a witness complex from random-policy rollouts."
    )
    parser.add_argument(
        "--env", default="FrozenLake-v1",
        choices=list(ENV_CONFIGS),
        help="Gymnasium environment ID (default: FrozenLake-v1)",
    )
    parser.add_argument("--episodes",     type=int,   default=None, help="Override n_episodes")
    parser.add_argument("--landmarks",    type=int,   default=None, help="Override num_landmarks")
    parser.add_argument("--epochs",       type=int,   default=200, help="Override n_epochs")
    parser.add_argument("--threshold",    type=float, default=None, help="Override threshold_pct")
    parser.add_argument("--no-show",      action="store_true",      help="Skip plt.show()")
    args = parser.parse_args()

    cfg = dict(ENV_CONFIGS[args.env])  # copy so we can mutate
    if args.episodes  is not None: cfg["n_episodes"]  = args.episodes
    if args.landmarks is not None: cfg["num_landmarks"] = args.landmarks
    if args.epochs    is not None: cfg["n_epochs"]    = args.epochs
    if args.threshold is not None: cfg["threshold_pct"] = args.threshold

    print("=" * 60)
    print(f"DISCRETE MORSE THEORY — {args.env}")
    print("=" * 60)

    # ── Generic random-policy collection ─────────────────────────────────
    env = gym.make(args.env, **cfg["make_kwargs"])

    demo_collector = Collector.__new__(Collector)
    demo_collector.states  = []
    demo_collector.actions = []
    demo_collector.rewards = []
    demo_collector.dones   = []
    demo_collector.ptrs    = []
    demo_collector.ptr     = 0

    print(f"Collecting {cfg['n_episodes']} random-policy episodes...")
    for _ in range(cfg["n_episodes"]):
        obs, _ = env.reset()
        obs = np.array(obs, dtype=np.float32).flatten()
        ep_states  = [obs]
        ep_actions, ep_rewards, ep_dones = [], [], []
        for _ in range(cfg["max_steps"]):
            a = env.action_space.sample()
            obs, r, terminated, truncated, _ = env.step(a)
            obs = np.array(obs, dtype=np.float32).flatten()
            ep_actions.append(np.array(a).flatten())
            ep_rewards.append(float(r))
            ep_dones.append(bool(terminated or truncated))
            ep_states.append(obs)
            if terminated or truncated:
                break
        demo_collector.add({
            "states":  np.array(ep_states,  dtype=np.float32),
            "actions": np.array(ep_actions, dtype=np.float32),
            "rewards": np.array(ep_rewards, dtype=np.float32),
            "dones":   np.array(ep_dones,   dtype=bool),
        })

    env.close()
    print(f"Collected {len(demo_collector.ptrs)} episodes.\n")

    # ── Run pipeline ──────────────────────────────────────────────────────
    results = run_morse_pipeline(
        demo_collector,
        state_dim=cfg["state_dim"],
        num_landmarks=cfg["num_landmarks"],
        nu=cfg["nu"],
        max_dim=cfg["max_dim"],
        knn_k=cfg["knn_k"],
        threshold_percentile=cfg["threshold_pct"],
        verbose=True,
    )

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Landmarks shape : {results['landmarks'].shape}")
    for d in range(cfg["max_dim"] + 1):
        print(f"{d}-simplices     : {len(results['simplices'].get(d, []))}")
    for d in range(cfg["max_dim"] + 1):
        print(f"Critical dim-{d}  : {len(results['critical_simplices'].get(d, []))}")
    n_indexed = len(results["knn_estimator"]._data.get(0, []))
    print(f"k-NN episodes    : {n_indexed} terminal episode(s) indexed")

    critical_0 = results["critical_simplices"].get(0, [])
    if critical_0:
        print(f"\nCritical landmark states (dim-0):")
        for sigma in critical_0:
            lm = results["landmarks"][sigma[0]]
            print(f"  landmark {sigma[0]}: state={lm.tolist()}")

    # ── Visualize ─────────────────────────────────────────────────────────
    out_dir = os.path.join(os.path.dirname(__file__), "..", "results", "from_scripts")
    os.makedirs(out_dir, exist_ok=True)
    env_tag = args.env.lower().replace("-", "_").replace("/", "_")
    save_path = os.path.join(out_dir, f"morse_complex_{env_tag}.png")
    print(f"\nGenerating visualization → {save_path}")
    visualize_morse_complex(results, save_path=save_path, show=not args.no_show)
