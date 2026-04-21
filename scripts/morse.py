"""
Goal: Given a replay buffer $\mathcal{D}$ containing trajectories from multiple tasks, construct:

    A graph-induced complex (specifically, a Vietoris-Rips complex or witness complex) on a subset of landmark states.

    A discrete Morse function $f$ on the vertices of this complex, derived from $\Phi(S)$.

    A Morse decomposition that identifies critical states (bottlenecks, decision points) in the meta-task skeleton.

Output: A set of critical states $\mathcal{C} = {c_1, c_2, \dots}$ that form the skeleton of the meta-MDP.

---
Potential function based on replay buffer (described in utils.md)
    - The reverse episodic return for a trajectory
    - $\tau = \{s_0, a_0, r_1, s_1, ..., s_n\}$, a discounting factor $0<\gamma<1$ and a given subset S of the state space with $s_n \in S$ and $s_i \notin S , \text{ if } s<n $
    $\Phi_{\tau}(S) = \sum_{t=1}^n \gamma^{n-t}r_t$.
    - The potential of the set $S$ is then for a given policy $\pi$ is
    $\Phi(S) = \min_{\pi} \mathbb{E}_{\pi} \Phi_{\tau}(S)$.
---


step1: Landmark selection
Starting with terminal state clusters:
    - Filter trajectories with dones: $Traj^0 = \{traj_i | traj_i["dones"][-1] == True\}$ and add their states to the pool.
    - Compute the minimum of the returns $R = min[\Phi_{traj_i}(traj_i["states"][-1]) : traj_i \in Traj^0]$
    - For all trajectories, find indices such that the upper bound criteria is satisfied:  $\Phi_{\traj}(...[-1]) >= R_0$.
    - If this condition is satisfied, all states from traj_i with index >=j are added in the pool and their last_accessed is updated to j.
    - cluster these states using k-nn and set the mean of each cluster's $\Phi_{traj}$ as the potential of that cluster.

 - For the rest of the states in the replay buffer, perform clustering based on FPS starting from the terminal sets and going towards the starting states
"""

import random
from collections import defaultdict
from itertools import combinations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class CollectorAdapter:
    """
    Wraps utils/collect.py::Collector to expose the interface expected by
    select_landmarks() (get_all_states) and train_backward_value_net (iter_episodes).
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


def select_landmarks(replay_buffer, num_landmarks=500, distance_metric='euclidean'):
    """
    Selects landmark states using farthest point sampling.
    """
    # Collect all unique states from buffer
    all_states = replay_buffer.get_all_states()  # [N, state_dim]

    num_landmarks = min(num_landmarks, len(all_states))

    # Initialize with random state
    landmarks = [all_states[random.randint(0, len(all_states)-1)]]

    # Compute distances to nearest landmark for all states
    dist_to_landmarks = torch.cdist(all_states, torch.stack(landmarks)).min(dim=1)[0]

    for _ in range(num_landmarks - 1):
        # Select state farthest from existing landmarks
        next_idx = dist_to_landmarks.argmax()
        landmarks.append(all_states[next_idx])

        # Update distances
        new_dists = torch.norm(all_states - landmarks[-1], dim=1)
        dist_to_landmarks = torch.minimum(dist_to_landmarks, new_dists)

    return torch.stack(landmarks)  # [num_landmarks, state_dim]


def build_witness_complex(
    landmarks: torch.Tensor,
    all_states: torch.Tensor,
    nu: int = 2,
    max_dim: int = 2,
    chunk_size: int = 1024,
) -> dict:
    """
    Builds a lazy witness complex on landmarks witnessed by all_states.

    A k-simplex sigma is included iff at least `nu` witness states have
    all vertices of sigma among their (max_dim + nu) nearest landmarks.

    nu: number of witnesses required to validate a simplex.
    """
    n_landmarks = len(landmarks)
    # Each witness certifies simplices whose vertices are all in its top-k list.
    # k = max_dim + nu ensures a witness can certify any dim<=max_dim simplex
    # while still requiring nu witnesses total.
    k = min(max_dim + nu, n_landmarks)

    landmarks_f = landmarks.float()
    all_states_f = all_states.float()

    # Phase 1: collect top-k landmark neighbors for each witness (chunked for memory)
    neighbor_lists = []
    for start in range(0, len(all_states_f), chunk_size):
        chunk = all_states_f[start : start + chunk_size]
        dists = torch.cdist(chunk, landmarks_f)           # [chunk, L]
        _, top_k = dists.topk(k, dim=1, largest=False)   # [chunk, k]
        for row in top_k.tolist():
            neighbor_lists.append(tuple(sorted(row)))

    # Phase 2: count witnesses per candidate simplex (all dims at once)
    witness_count: dict = defaultdict(int)
    for neighbors in neighbor_lists:
        for d in range(max_dim + 1):
            for sigma in combinations(neighbors, d + 1):
                witness_count[sigma] += 1

    # Phase 3: threshold — include simplex if enough witnesses
    simplices: dict = {d: [] for d in range(max_dim + 1)}
    for sigma, count in witness_count.items():
        if count >= nu:
            dim = len(sigma) - 1
            if dim <= max_dim:
                simplices[dim].append(sigma)

    return simplices


class RewardNormalizer:
    """
    Online Welford mean/variance tracker.
    Normalises individual rewards to approximately zero mean and unit std
    before they are used to build BackwardValueNet training targets.
    Call update() on all episode rewards before training, then pass the
    fitted normalizer to train_backward_value_net().
    """

    def __init__(self):
        self.mean  = 0.0
        self.M2    = 0.0
        self.count = 0

    def update(self, rewards) -> None:
        for r in rewards:
            self.count += 1
            delta      = float(r) - self.mean
            self.mean += delta / self.count
            delta2     = float(r) - self.mean
            self.M2   += delta * delta2

    @property
    def std(self) -> float:
        return float(np.sqrt(self.M2 / max(self.count - 1, 1)))

    def normalize(self, reward: float) -> float:
        return (float(reward) - self.mean) / (self.std + 1e-8)

    def denormalize(self, value: float) -> float:
        return float(value) * (self.std + 1e-8) + self.mean


class BackwardValueNet(nn.Module):
    """
    Backward-discounted value function V(s; S_proto).
    Uses BatchNorm1d after each linear layer for gradient stability.
    Batch size must be > 1 during training (single-sample batches are skipped).
    """

    def __init__(self, state_dim, set_embedding_dim=64, hidden_dim=256):
        super().__init__()
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
        )
        self.set_encoder = nn.Sequential(
            nn.Linear(state_dim, set_embedding_dim),
            nn.BatchNorm1d(set_embedding_dim),
            nn.ReLU(),
            nn.Linear(set_embedding_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
        )
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, s, S_proto):
        s_feat   = self.state_encoder(s)
        set_feat = self.set_encoder(S_proto)
        combined = torch.cat([s_feat, set_feat], dim=-1)
        return self.value_head(combined)


def train_backward_value_net(
    adapter,
    value_net: BackwardValueNet,
    optimizer,
    gamma: float              = 0.99,
    n_epochs: int             = 10,
    batch_size: int           = 256,
    device: str               = "cpu",
    clip_grad: float          = 1.0,
    reward_normalizer         = None,
    target_clip: float        = None,
    hit_threshold: float      = 1.0,
) -> list:
    """
    Trains BackwardValueNet using reverse-trajectory Monte Carlo targets.

    Target for state s_t: Φ_t = Σ_{k=t}^{hit-1} γ^{hit-1-k} · r̃_k
      where r̃_k is reward_normalizer.normalize(r_k) if a normalizer is
      supplied, and hit is the first step at which s_{k+1} enters the
      ε-ball around the terminal state (hit_threshold radius).

    Only episodes that actually reach their terminal state are used
    (dones[-1] == True for strictly terminal; truncated episodes are
    also accepted if the terminal state is visited earlier in the
    trajectory).

    Improvements over the plain MC version:
      1. reward_normalizer  — normalise each r_k to ~N(0,1) before
                               accumulating, preventing target explosion.
      2. Correct hit_idx    — only use states before S is first entered,
                               discarding post-goal steps that would
                               otherwise carry incorrect backward targets.
      3. Huber loss         — less sensitive to outlier targets that MSE,
                               making target_clip unnecessary in most cases.
      4. target_clip        — optional hard-clip of normalised MC targets.
                               Defaults to None (no clipping): Huber loss
                               already provides robustness to outliers.
                               Set to 3/(1-γ) ≈ 300 for γ=0.99 if you
                               want a finite ceiling on normalised returns.
      5. BatchNorm-safe     — batches of size ≤1 are skipped so BN layers
                               never receive a single-sample batch.

    Returns list of per-epoch mean losses.
    """
    all_s, all_s_hit, all_targets = [], [], []

    for ep in adapter.iter_episodes():
        states  = np.asarray(ep["states"],  dtype=np.float32)   # [T+1, D]
        rewards = np.asarray(ep["rewards"], dtype=np.float32)   # [T]
        dones   = np.asarray(ep["dones"],   dtype=bool)         # [T]
        T = len(rewards)
        if T == 0:
            continue

        # ── Find first hit into S ──────────────────────────────────────────
        # S_proto is the terminal state s_T (states[-1]).
        # We look for the earliest t where states[t+1] enters S.
        s_terminal = states[-1]
        hit_idx    = None

        for i in range(T):
            dist = float(np.linalg.norm(states[i + 1] - s_terminal))
            if dist < hit_threshold:
                hit_idx = i
                break

        # Fall back to T-1 for strictly terminal episodes
        if hit_idx is None:
            if not dones[-1]:
                continue          # truncated and never visited S — skip
            hit_idx = T - 1

        # ── Normalise rewards then compute backward targets ───────────────
        T_use = hit_idx + 1       # number of steps before (and including) hit
        raw_r = rewards[:T_use]

        if reward_normalizer is not None:
            norm_r = np.array(
                [reward_normalizer.normalize(r) for r in raw_r], dtype=np.float32
            )
        else:
            norm_r = raw_r.copy()

        # Φ_t = Σ_{k=t}^{T_use-1} γ^{T_use-1-k} · norm_r[k]
        gam_pows   = gamma ** np.arange(T_use - 1, -1, -1, dtype=np.float32)
        weighted   = gam_pows * norm_r
        mc_targets = np.flip(np.cumsum(np.flip(weighted))).astype(np.float32)

        # Hard-clip only when caller explicitly requests it
        if target_clip is not None:
            mc_targets = np.clip(mc_targets, -target_clip, target_clip)

        s_hit = states[hit_idx + 1]       # state at which S was entered
        for t in range(T_use):
            all_s.append(states[t])
            all_s_hit.append(s_hit)
            all_targets.append(float(mc_targets[t]))

    if not all_s:
        return []

    # First pass: fit reward normalizer if one was supplied and empty
    if reward_normalizer is not None and reward_normalizer.count == 0:
        for ep in adapter.iter_episodes():
            reward_normalizer.update(ep["rewards"].tolist()
                                     if hasattr(ep["rewards"], "tolist")
                                     else list(ep["rewards"]))

    s_tensor   = torch.tensor(np.array(all_s,       dtype=np.float32), device=device)
    st_tensor  = torch.tensor(np.array(all_s_hit,   dtype=np.float32), device=device)
    tgt_tensor = torch.tensor(np.array(all_targets, dtype=np.float32),
                               device=device).unsqueeze(1)

    N = len(s_tensor)
    epoch_losses = []
    value_net.train()

    for _ in range(n_epochs):
        perm = torch.randperm(N, device=device)
        batch_losses = []
        for start in range(0, N, batch_size):
            idx = perm[start : start + batch_size]
            if len(idx) <= 1:
                continue          # skip single-sample batches — BatchNorm needs B > 1

            s_b   = s_tensor[idx]
            st_b  = st_tensor[idx]
            tgt_b = tgt_tensor[idx]

            pred = value_net(s_b, st_b)
            loss = F.huber_loss(pred, tgt_b, delta=1.0)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(value_net.parameters(), clip_grad)
            optimizer.step()

            batch_losses.append(loss.item())

        if batch_losses:
            epoch_losses.append(float(np.mean(batch_losses)))

    value_net.eval()
    return epoch_losses


# Define $\Phi on simplices$
# For a simplex $\sigma = [v_0, \dots, v_k]$ (a set of landmark states), we define the set $S_\sigma$
# as the union of $\epsilon$-balls around each vertex, or simply as the set of landmark states themselves.
# We approximate $\Phi(S_\sigma)$ by evaluating the backward value function at a reference start state
# $s_0$ (or the minimum over a set of start states):

def compute_simplex_potential(simplex_vertices, landmark_states, start_states, value_net):
    """
    simplex_vertices: tuple of landmark indices
    landmark_states: [num_landmarks, state_dim]
    start_states: [num_starts, state_dim]
    value_net: trained BackwardValueNet
    """
    # Compute prototype of simplex as mean of its vertices
    S_proto = landmark_states[list(simplex_vertices)].mean(dim=0)  # [state_dim]

    # Expand for batch of start states
    S_proto_batch = S_proto.unsqueeze(0).expand(len(start_states), -1)

    # Compute backward values from all start states
    with torch.no_grad():
        values = value_net(start_states, S_proto_batch)  # [num_starts, 1]

    # Φ(σ) = min over start states
    return values.min().item()


# Constructing a discrete morse function

def compute_morse_function(simplices, landmark_states, start_states, value_net):
    """
    Computes discrete Morse function values for all simplices.
    """
    morse_values = {}

    # First, compute base Φ for all simplices
    phi_values = {}
    for dim, simplex_list in simplices.items():
        for simplex in simplex_list:
            phi_values[simplex] = compute_simplex_potential(
                simplex, landmark_states, start_states, value_net
            )

    # Compute Morse values
    for dim, simplex_list in simplices.items():
        for simplex in simplex_list:
            if dim == 0:
                # Vertex: Morse value is just Φ
                morse_values[simplex] = phi_values[simplex]
            else:
                # Higher dimension: marginal value
                faces = get_faces(simplex)  # All (dim-1)-dimensional faces
                mean_face_phi = sum(phi_values[face] for face in faces) / len(faces)
                morse_values[simplex] = phi_values[simplex] - mean_face_phi

    return morse_values, phi_values


def get_faces(simplex):
    """Returns all codimension-1 faces of a simplex."""
    faces = []
    for i in range(len(simplex)):
        face = simplex[:i] + simplex[i+1:]
        faces.append(face)
    return faces


# Identify critical simplices
def identify_critical_simplices(simplices, morse_values, threshold_percentile=90):
    """
    Identifies critical simplices based on Morse function extrema.
    """
    critical = {0: [], 1: [], 2: []}

    # Build adjacency for efficient lookup
    face_dict = {}  # simplex -> list of faces
    coface_dict = {}  # simplex -> list of cofaces

    for dim, simplex_list in simplices.items():
        for simplex in simplex_list:
            faces = get_faces(simplex)
            face_dict[simplex] = faces
            for face in faces:
                if face not in coface_dict:
                    coface_dict[face] = []
                coface_dict[face].append(simplex)

    # Find critical simplices
    all_morse_vals = list(morse_values.values())
    threshold = np.percentile(np.abs(all_morse_vals), threshold_percentile)

    for dim, simplex_list in simplices.items():
        for simplex in simplex_list:
            val = morse_values[simplex]

            # Check faces
            face_vals = [morse_values.get(face, float('-inf')) for face in face_dict.get(simplex, [])]
            max_face_val = max(face_vals) if face_vals else float('-inf')

            # Check cofaces
            coface_vals = [morse_values.get(coface, float('-inf')) for coface in coface_dict.get(simplex, [])]
            max_coface_val = max(coface_vals) if coface_vals else float('-inf')

            # Critical if it's a local maximum (value higher than all faces and cofaces)
            # AND the magnitude exceeds threshold
            if val > max_face_val and val > max_coface_val and abs(val) > threshold:
                critical[dim].append(simplex)

            # Also capture minima (for sinks/goals)
            min_face_val = min(face_vals) if face_vals else float('inf')
            min_coface_val = min(coface_vals) if coface_vals else float('inf')
            if val < min_face_val and val < min_coface_val and abs(val) > threshold:
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
    Panel 3 — Training loss: epoch-loss curve from BackwardValueNet training.

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
                        color='none', edgecolor='gold', linewidth=2, zorder=4)
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
                ax.plot(x, y, color='gold', linewidth=3.5, zorder=2, alpha=0.4)

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
                       s=260, facecolors='none', edgecolors='crimson',
                       linewidths=2.0, zorder=6)
            ax.annotate(f"c{v}", xy=pos2d[v], xytext=(4, 4),
                        textcoords='offset points', fontsize=7,
                        color='crimson', zorder=7)

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
    has_loss = bool(train_losses)
    n_panels = 3 if has_loss else 2
    fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 6))

    draw_complex(axes[0], phi_0, phi_1, phi_2,
                 cmap='RdYlGn',
                 title="Potential Landscape Φ on Witness Complex")

    # Diverging colormap centred at 0 for Morse values
    mv_all = list(morse_values.values())
    mv_abs = max(abs(v) for v in mv_all) if mv_all else 1.0
    draw_complex(axes[1], morse_0, morse_1, morse_2,
                 cmap='RdBu_r',
                 title="Discrete Morse Function\n(crimson rings = critical)",
                 vmin=-mv_abs, vmax=mv_abs)

    if has_loss:
        ax3 = axes[2]
        ax3.plot(range(1, len(train_losses) + 1), train_losses,
                 color='steelblue', linewidth=2, marker='o', markersize=4)
        ax3.set_xlabel("Epoch", fontsize=10)
        ax3.set_ylabel("MSE Loss", fontsize=10)
        ax3.set_title("BackwardValueNet Training Loss", fontsize=11, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        ax3.tick_params(labelsize=8)

    # ── Legend for critical markers ───────────────────────────────────────
    from matplotlib.lines import Line2D
    legend_handles = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='w',
               markeredgecolor='crimson', markeredgewidth=2, markersize=10,
               label='Critical simplex'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='w',
               markeredgecolor='gold', markeredgewidth=2, markersize=10,
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


def run_morse_pipeline(
    collector,
    state_dim: int,
    num_landmarks: int = 500,
    nu: int = 2,
    max_dim: int = 2,
    gamma: float = 0.99,
    hidden_dim: int = 256,
    set_embedding_dim: int = 64,
    lr: float = 3e-4,
    n_train_epochs: int = 10,
    batch_size: int = 256,
    threshold_percentile: float = 90.0,
    device: str = "cpu",
    num_start_states: int = 64,
    verbose: bool = True,
) -> dict:
    """
    Full Discrete Morse pipeline from a Collector to critical simplices.

    Returns a dict with keys:
        landmarks, simplices, value_net, morse_values, phi_values,
        critical_simplices, train_losses
    """
    adapter = CollectorAdapter(collector, device=device)

    if verbose:
        print("Step 1: Selecting landmarks via FPS...")
    landmarks = select_landmarks(adapter, num_landmarks=num_landmarks)

    if verbose:
        print(f"  {len(landmarks)} landmarks selected.")
        print("Step 2: Building witness complex...")

    all_states = adapter.get_all_states()
    simplices = build_witness_complex(landmarks, all_states, nu=nu, max_dim=max_dim)

    if verbose:
        for d, slist in simplices.items():
            print(f"  dim {d}: {len(slist)} simplices")

    if verbose:
        print("Step 3: Training BackwardValueNet...")

    # Fit reward normalizer on all episode rewards before training
    reward_normalizer = RewardNormalizer()
    for ep in adapter.iter_episodes():
        rw = ep["rewards"]
        reward_normalizer.update(rw.tolist() if hasattr(rw, "tolist") else list(rw))
    if verbose:
        print(f"  Reward stats: mean={reward_normalizer.mean:.4f}  "
              f"std={reward_normalizer.std:.4f}")

    value_net = BackwardValueNet(
        state_dim, set_embedding_dim=set_embedding_dim, hidden_dim=hidden_dim
    ).to(device)
    optimizer = optim.Adam(value_net.parameters(), lr=lr)
    train_losses = train_backward_value_net(
        adapter, value_net, optimizer,
        gamma=gamma, n_epochs=n_train_epochs,
        batch_size=batch_size, device=device,
        reward_normalizer=reward_normalizer,
    )

    if verbose and train_losses:
        print(f"  Final epoch loss: {train_losses[-1]:.6f}")

    if verbose:
        print("Step 4: Computing discrete Morse function...")

    perm = torch.randperm(len(all_states))[:num_start_states]
    start_states = all_states[perm].to(device)
    landmark_states = landmarks.to(device)

    morse_values, phi_values = compute_morse_function(
        simplices, landmark_states, start_states, value_net
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
        "landmarks":          landmarks,
        "simplices":          simplices,
        "value_net":          value_net,
        "morse_values":       morse_values,
        "phi_values":         phi_values,
        "critical_simplices": critical,
        "train_losses":       train_losses,
    }


if __name__ == "__main__":
    import argparse
    import sys
    import os
    import gymnasium as gym

    # Allow running from project root or scripts/ directory
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from utils.collect import Collector

    # ── Per-environment defaults ──────────────────────────────────────────
    # make_kwargs   : passed to gym.make()
    # state_dim     : dimensionality of the observation after flattening
    # n_episodes    : random-policy episodes to collect
    # max_steps     : episode step cap
    # num_landmarks : FPS landmark count
    # nu            : witness threshold
    # max_dim       : max simplex dimension (1 = graph, 2 = triangulated)
    # hidden_dim    : BackwardValueNet hidden size
    # set_emb_dim   : set encoder embedding size
    # n_epochs      : training epochs
    # batch_size    : training batch size
    # threshold_pct : Morse criticality percentile
    ENV_CONFIGS = {
        "FrozenLake-v1": dict(
            make_kwargs={"is_slippery": False},
            state_dim=1, n_episodes=200, max_steps=100,
            num_landmarks=16, nu=1, max_dim=1,
            hidden_dim=64, set_emb_dim=16,
            n_epochs=5, batch_size=32, threshold_pct=75.0,
        ),
        "LunarLander-v3": dict(
            make_kwargs={},
            state_dim=8, n_episodes=400, max_steps=500,
            num_landmarks=64, nu=2, max_dim=2,
            hidden_dim=128, set_emb_dim=32,
            n_epochs=15, batch_size=128, threshold_pct=80.0,
        ),
        "CartPole-v1": dict(
            make_kwargs={},
            state_dim=4, n_episodes=300, max_steps=500,
            num_landmarks=32, nu=2, max_dim=2,
            hidden_dim=64, set_emb_dim=16,
            n_epochs=10, batch_size=64, threshold_pct=80.0,
        ),
        "MountainCar-v0": dict(
            make_kwargs={},
            state_dim=2, n_episodes=500, max_steps=200,
            num_landmarks=32, nu=2, max_dim=2,
            hidden_dim=64, set_emb_dim=16,
            n_epochs=10, batch_size=64, threshold_pct=80.0,
        ),
        "Acrobot-v1": dict(
            make_kwargs={},
            state_dim=6, n_episodes=300, max_steps=500,
            num_landmarks=48, nu=2, max_dim=2,
            hidden_dim=128, set_emb_dim=32,
            n_epochs=10, batch_size=64, threshold_pct=80.0,
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
    parser.add_argument("--epochs",       type=int,   default=None, help="Override n_epochs")
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
        hidden_dim=cfg["hidden_dim"],
        set_embedding_dim=cfg["set_emb_dim"],
        n_train_epochs=cfg["n_epochs"],
        batch_size=cfg["batch_size"],
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
    if results["train_losses"]:
        print(f"Final train loss: {results['train_losses'][-1]:.6f}")
    else:
        print("Final train loss: N/A (no terminal episodes found)")

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
