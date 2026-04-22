"""
Potential functions for meta-skeleton reward shaping.

SkeletonPotential            — graph-distance on 1-simplices (pure topology).
EmpiricalHittingTimePotential — k-NN backward returns from replay buffer.
compute_sparse_shaped_reward  — sparse shaping that fires only near subgoals.
ShapedRewardWrapper           — Gymnasium wrapper applying potential shaping.
"""

import numpy as np
import networkx as nx
import gymnasium as gym
from scipy.spatial import KDTree


# ── Device-agnostic array conversion ──────────────────────────────────────

def _to_numpy_f32(x) -> np.ndarray:
    """Convert any array-like or torch tensor (including MPS/CUDA) to float32 numpy.

    All public entry points that accept state arrays call this so callers on
    MPS or CUDA devices never need to manually move tensors to CPU first.
    """
    if hasattr(x, "detach"):          # torch.Tensor on any device
        return x.detach().cpu().numpy().astype(np.float32, copy=False)
    return np.asarray(x, dtype=np.float32)


# ── Nearest-subgoal lookup ─────────────────────────────────────────────────

def _nearest_subgoal(s, meta_subgoals: dict):
    """Return (sg_id, distance) of the meta-subgoal nearest to s."""
    s_arr = _to_numpy_f32(s).flatten()
    best_id, best_d = None, float("inf")
    for sg_id, sg_data in meta_subgoals.items():
        d = float(np.linalg.norm(s_arr - _to_numpy_f32(sg_data["state"]).flatten()))
        if d < best_d:
            best_d, best_id = d, sg_id
    return best_id, best_d


# ── Sparse shaping signal ──────────────────────────────────────────────────

def compute_sparse_shaped_reward(
    s,
    s_next,
    r_env: float,
    meta_subgoals: dict,
    skeleton_potential,
    shaping_scale: float   = 1.0,
    subgoal_threshold: float = float("inf"),
) -> tuple:
    """
    Compute a sparsely shaped reward.

    Shaping fires only when the nearest meta-subgoal is within
    `subgoal_threshold` of s_next (default: always fire).

    Returns:
        r_shaped    float
        shaping_info dict  {subgoal_id, distance, r_intrinsic, triggered}
    """
    s_arr      = _to_numpy_f32(s).flatten()
    s_next_arr = _to_numpy_f32(s_next).flatten()

    sg_id, dist = _nearest_subgoal(s_next_arr, meta_subgoals)
    triggered   = (sg_id is not None) and (dist <= subgoal_threshold)

    if triggered:
        r_int    = skeleton_potential.get_intrinsic_reward(s_arr, s_next_arr, sg_id)
        r_shaped = float(r_env) + shaping_scale * r_int
    else:
        r_int    = 0.0
        r_shaped = float(r_env)

    return r_shaped, {
        "subgoal_id":  sg_id,
        "distance":    dist,
        "r_intrinsic": r_int,
        "triggered":   triggered,
    }


# ── Potential implementations ──────────────────────────────────────────────

class SkeletonPotential:
    """
    Φ(s; c) = −dist_G(landmark(s), landmark(c))

    Graph built from the 1-simplices of the meta-skeleton.  Edge weights are
    unit (hop count), so the distance is purely topological: the number of
    edges on the shortest path through the simplicial complex.  Distances are
    cached after first lookup.
    """

    def __init__(self, landmarks, simplices: dict, meta_subgoals: dict):
        self.landmarks = _to_numpy_f32(landmarks)
        self.subgoals  = meta_subgoals

        self.G = nx.Graph()
        for i in range(len(self.landmarks)):
            self.G.add_node(i)
        for edge in simplices.get(1, []):
            self.G.add_edge(edge[0], edge[1])   # unit weight — hop count

        self._sg_landmarks: dict = {
            sg_id: int(np.argmin(np.linalg.norm(
                self.landmarks - _to_numpy_f32(sg_data["state"]).flatten(), axis=1
            )))
            for sg_id, sg_data in meta_subgoals.items()
        }
        self._cache: dict = {}

    def _closest_landmark(self, s: np.ndarray) -> int:
        return int(np.argmin(np.linalg.norm(self.landmarks - s, axis=1)))

    def get_potential(self, s, sg_id) -> float:
        s_arr = _to_numpy_f32(s).flatten()
        v_s   = self._closest_landmark(s_arr)
        v_c   = self._sg_landmarks[sg_id]
        key   = (v_s, v_c)
        if key not in self._cache:
            try:
                d = nx.shortest_path_length(self.G, v_s, v_c)
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                d = len(self.landmarks)   # finite stand-in for unreachable nodes
            self._cache[key] = d
        return -self._cache[key]

    def get_intrinsic_reward(self, s, s_next, sg_id) -> float:
        return self.get_potential(s_next, sg_id) - self.get_potential(s, sg_id)


class EmpiricalHittingTimePotential:
    """
    Non-parametric potential estimated from replay-buffer hitting times.

    Φ(s; c) = mean backward-discounted return of the k nearest hitting
    trajectories (by start state) that ever reached c.
    """

    def __init__(
        self,
        replay_buffer,
        meta_subgoals: dict,
        k: int             = 10,
        gamma: float       = 0.99,
        hit_threshold: float = 0.5,
    ):
        self.k        = k
        self.gamma    = gamma
        self.subgoals = meta_subgoals

        self._trajs: dict = {sg_id: [] for sg_id in meta_subgoals}

        for ep in replay_buffer.iter_episodes():
            states  = np.asarray(ep["states"],  dtype=np.float32)
            rewards = np.asarray(ep["rewards"], dtype=np.float32)
            T = len(rewards)
            for sg_id, sg_data in meta_subgoals.items():
                sg_state = np.asarray(sg_data["state"], dtype=np.float32)
                hit_idx  = None
                for t in range(T):
                    if np.linalg.norm(states[t + 1] - sg_state) < hit_threshold:
                        hit_idx = t
                        break
                if hit_idx is None:
                    continue
                n    = hit_idx + 1
                exps = gamma ** np.arange(n - 1, -1, -1, dtype=np.float32)
                self._trajs[sg_id].append({
                    "start_state":     states[0].copy(),
                    "backward_return": float(np.dot(rewards[:n], exps)),
                })

        # Build one KDTree per subgoal (O(N log N) once, O(log N) per query).
        self._trees:     dict = {}
        self._back_rets: dict = {}
        for sg_id, trajs in self._trajs.items():
            if trajs:
                starts = np.stack([t["start_state"] for t in trajs])
                self._trees[sg_id]     = KDTree(starts)
                self._back_rets[sg_id] = np.array(
                    [t["backward_return"] for t in trajs], dtype=np.float32
                )

    def get_potential(self, s, sg_id) -> float:
        tree = self._trees.get(sg_id)
        if tree is None:
            return 0.0
        s_arr = _to_numpy_f32(s).flatten()
        k     = min(self.k, len(self._back_rets[sg_id]))
        _, nn_idx = tree.query(s_arr, k=k)
        return float(np.mean(self._back_rets[sg_id][nn_idx]))

    def get_intrinsic_reward(self, s, s_next, sg_id) -> float:
        return self.get_potential(s_next, sg_id) - self.get_potential(s, sg_id)


# ── Combined potential ────────────────────────────────────────────────────

class CombinedPotential:
    """
    Φ(s; c) = α · Φ̃_skel(s; c)  +  (1−α) · Φ̃_emp(s; c)

    Each component is normalised to unit standard deviation over a sample of
    (landmark, subgoal) evaluations before mixing, making the convex combination
    scale-invariant regardless of the magnitude difference between hop-count
    distances and discounted returns.

    α = 1.0  →  pure graph topology (SkeletonPotential)
    α = 0.0  →  pure empirical hitting-time returns (EmpiricalHittingTimePotential)

    When the empirical component has no hitting trajectories yet (empty buffer),
    its std is 0 → scale falls back to 1.0 and all empirical values are 0, so
    the combined potential degrades gracefully to the pure skeleton signal.
    """

    def __init__(
        self,
        skeleton_potential,
        empirical_potential,
        landmarks:  np.ndarray,
        alpha: float = 0.5,
    ):
        if not (0.0 <= alpha <= 1.0):
            raise ValueError(f"alpha must be in [0, 1], got {alpha}")
        self.subgoals = skeleton_potential.subgoals
        self._skel    = skeleton_potential
        self._emp     = empirical_potential
        self.alpha    = alpha

        landmarks        = _to_numpy_f32(landmarks)
        self._skel_scale = self._estimate_scale(skeleton_potential, landmarks)
        self._emp_scale  = self._estimate_scale(empirical_potential, landmarks)

    @staticmethod
    def _estimate_scale(pot, landmarks: np.ndarray) -> float:
        """Std of get_potential over up to 50 landmarks × all subgoals."""
        if landmarks is None or len(landmarks) == 0 or not pot.subgoals:
            return 1.0
        lm_sample = landmarks[: min(len(landmarks), 50)]
        vals = [
            pot.get_potential(lm, sg_id)
            for sg_id in pot.subgoals
            for lm in lm_sample
        ]
        if len(vals) < 2:
            return 1.0
        std = float(np.std(vals))
        return std if std > 1e-8 else 1.0

    def get_potential(self, s, sg_id) -> float:
        skel = self._skel.get_potential(s, sg_id) / self._skel_scale
        emp  = self._emp.get_potential(s, sg_id)  / self._emp_scale
        return self.alpha * skel + (1.0 - self.alpha) * emp

    def get_intrinsic_reward(self, s, s_next, sg_id) -> float:
        return self.get_potential(s_next, sg_id) - self.get_potential(s, sg_id)


# ── k-NN backward estimator (for Morse function) ──────────────────────────

class KNNBackwardEstimator:
    """
    Non-parametric potential estimated directly from replay-buffer trajectories.

    Φ(s; task) = (importance-weighted) mean backward-discounted return of the k
                 *visited states* nearest to s.

    Episode selection
    -----------------
    survived_only=False (default)
        Use episodes that ended with a natural terminal (terminated=True).
        Suitable for environments where goal-reaching ends the episode.
    survived_only=True   ← MuJoCo locomotion mode
        Use episodes that survived to the time limit (no terminated=True step),
        because in MuJoCo terminated=True means failure (falling), not success.
        Importance-sampling fallback: when fewer than `min_survived` such
        episodes exist, all episodes are included but each step is weighted by
          w(τ) = max(0, R(τ)) / R_max
        so high-return (near-successful) episodes dominate.

    Parameters
    ----------
    replay_source   : object with iter_episodes()
    gamma           : discount factor
    k               : number of nearest neighbours
    survived_only   : use truncated (survived) episodes instead of terminated ones
    min_survived    : minimum survived episodes before IS fallback kicks in
    """

    def __init__(
        self,
        replay_source,
        gamma: float        = 0.99,
        k: int              = 10,
        survived_only: bool = False,
        min_survived: int   = 3,
    ):
        self.gamma = gamma
        self.k     = k
        self._data: dict = {}

        # ── First pass: collect all episodes with metadata ─────────────────
        all_eps = []
        for ep in replay_source.iter_episodes():
            states     = np.asarray(ep["states"],  dtype=np.float32)
            rewards    = np.asarray(ep["rewards"], dtype=np.float32)
            assert "terminated" in ep, (
                "Episode dict is missing 'terminated' key — check that all "
                "replay_buffer.push() calls pass terminated= explicitly."
            )
            term_flags = np.asarray(ep["terminated"], dtype=bool)
            tid        = int(ep.get("task_id", 0))
            T          = len(rewards)
            survived   = not any(term_flags)   # no early termination = survived
            total_R    = float(np.dot(rewards,
                               gamma ** np.arange(T - 1, -1, -1, dtype=np.float32)))
            all_eps.append({
                "states": states, "rewards": rewards, "term_flags": term_flags,
                "tid": tid, "T": T, "survived": survived, "total_R": total_R,
            })

        # ── Select which episodes to use and compute IS weights ────────────
        use_is_weights = False
        if survived_only:
            survived_eps = [e for e in all_eps if e["survived"]]
            if len(survived_eps) >= min_survived:
                selected = survived_eps
            else:
                # IS fallback: include all but weight by return
                selected       = all_eps
                use_is_weights = True
        else:
            # Original behaviour: use episodes with at least one terminated step
            selected = [e for e in all_eps if any(e["term_flags"])]

        if use_is_weights:
            max_R = max((e["total_R"] for e in selected), default=1e-8)
            max_R = max(max_R, 1e-8)
            for e in selected:
                e["_w"] = float(max(0.0, e["total_R"]) / max_R)
        else:
            for e in selected:
                e["_w"] = 1.0

        # ── Second pass: build backward-return entries ─────────────────────
        for ep_data in selected:
            w  = ep_data["_w"]
            if w == 0.0:
                continue
            states     = ep_data["states"]
            rewards    = ep_data["rewards"]
            term_flags = ep_data["term_flags"]
            tid        = ep_data["tid"]
            T          = ep_data["T"]

            if survived_only:
                # Use all T steps of the episode (survived or IS-weighted)
                n = T
            else:
                # Original: steps up to (and including) the first terminal
                hit = next((t for t in range(T) if term_flags[t]), None)
                if hit is None:
                    continue
                n = hit + 1

            # back_ret[t] = Σ_{i=t}^{n-1} γ^{i-t} r_i
            back_rets = np.empty(n, dtype=np.float32)
            G = 0.0
            for t in range(n - 1, -1, -1):
                G = float(rewards[t]) + gamma * G
                back_rets[t] = G

            task_list = self._data.setdefault(tid, [])
            for t in range(n):
                task_list.append({
                    "state":    states[t].copy(),
                    "back_ret": float(back_rets[t]),
                    "weight":   w,
                })

        # ── Build one KDTree per task_id ───────────────────────────────────
        self._trees:     dict = {}
        self._back_rets: dict = {}
        self._weights:   dict = {}
        for tid, entries in self._data.items():
            pts = np.stack([e["state"] for e in entries])
            self._trees[tid]     = KDTree(pts)
            self._back_rets[tid] = np.array([e["back_ret"] for e in entries], dtype=np.float32)
            self._weights[tid]   = np.array([e["weight"]   for e in entries], dtype=np.float32)

    def phi(self, s: np.ndarray, task_id: int = 0) -> float:
        """Φ(s; task) — importance-weighted k-NN mean backward-discounted return."""
        tree = self._trees.get(task_id)
        if tree is None:
            return 0.0
        s_flat    = _to_numpy_f32(s).flatten()
        k         = min(self.k, len(self._back_rets[task_id]))
        _, nn_idx = tree.query(s_flat, k=k)
        w         = self._weights[task_id][nn_idx]
        br        = self._back_rets[task_id][nn_idx]
        w_sum     = float(w.sum())
        if w_sum < 1e-8:
            return float(np.mean(br))
        return float(np.dot(w, br) / w_sum)

    def back_ret_stats(self) -> dict:
        """Return per-task diagnostics: n, mean, std of back_ret values.

        Call after construction to check whether phi will be flat:
            stats = knn_estimator.back_ret_stats()
            for tid, s in stats.items():
                if s["std"] < 0.01:
                    print(f"task {tid}: phi is flat — check terminal episode coverage")
        """
        stats = {}
        for tid, entries in self._data.items():
            vals = np.array([e["back_ret"] for e in entries], dtype=np.float32)
            wts  = np.array([e.get("weight", 1.0) for e in entries], dtype=np.float32)
            w_sum = float(wts.sum())
            w_mean = float(np.dot(wts, vals) / w_sum) if w_sum > 1e-8 else float(vals.mean())
            stats[tid] = {
                "n":    len(vals),
                "mean": w_mean,
                "std":  float(vals.std()),
            }
        return stats

    def all_task_ids(self) -> list:
        return list(self._data.keys())


# ── Witness-based simplex potential ───────────────────────────────────────

class WitnessBasedSimplexPotential:
    """
    Consistent simplex potential using KNNBackwardEstimator as the single
    potential evaluator, with witness-based aggregation for higher simplices.

    dim 0 (vertices)
        Φ(v) = knn_estimator.phi(landmarks[v], task_id)
        The landmark location itself is queried — no witnesses needed.

    dim ≥ 1 (edges, triangles, …)
        Φ(σ) = mean over witness states w that have every vertex of σ
                among their k-nearest landmarks of
                knn_estimator.phi(witness_states[w], task_id)
        Falls back to mean of vertex potentials when no witnesses exist.

    Using the same knn_estimator for both cases keeps all phi values on the
    same scale, making the Morse function comparison across dimensions valid.

    Parameters
    ----------
    landmarks           : np.ndarray [L, D]   landmark positions
    witness_states      : np.ndarray [N, D]   state visited by the agent (query states)
    witness_assignments : {state_idx: [landmark_idx, ...]}  k-nearest per state
    knn_estimator       : KNNBackwardEstimator
    task_id             : int
    """

    def __init__(
        self,
        landmarks: np.ndarray,
        witness_states: np.ndarray,
        witness_assignments: dict,
        knn_estimator,
        task_id: int,
    ):
        self.landmarks           = landmarks
        self.witness_states      = witness_states
        self.witness_assignments = witness_assignments
        self.knn_estimator       = knn_estimator
        self.task_id             = task_id
        self._witness_cache: dict = {}   # simplex → list[state_idx]

    def get_simplex_witnesses(self, simplex: tuple) -> list:
        """Return state indices whose k-nearest landmarks cover all vertices of simplex."""
        if simplex in self._witness_cache:
            return self._witness_cache[simplex]
        simplex_set = set(simplex)
        witnesses = [
            idx for idx, lm_list in self.witness_assignments.items()
            if simplex_set.issubset(set(lm_list))
        ]
        self._witness_cache[simplex] = witnesses
        return witnesses

    def get_potential(self, simplex: tuple) -> float:
        """
        Compute Φ(σ) using KNNBackwardEstimator.

        Vertices use the landmark location directly; higher simplices use the
        mean knn phi over all witness states that cover every vertex of σ.
        """
        if len(simplex) == 1:
            return self.knn_estimator.phi(self.landmarks[simplex[0]], self.task_id)

        witnesses = self.get_simplex_witnesses(simplex)
        if not witnesses:
            # Fall back to mean of vertex potentials
            return float(np.mean([self.get_potential((v,)) for v in simplex]))

        return float(np.mean([
            self.knn_estimator.phi(self.witness_states[w], self.task_id)
            for w in witnesses
        ]))


# ── Gym wrapper ────────────────────────────────────────────────────────────

class ShapedRewardWrapper(gym.Wrapper):
    """
    r_shaped = r_env + scale · [Φ(s'; c) − Φ(s; c)]

    c is the nearest meta-subgoal to s'.  Shaping is skipped when the
    nearest subgoal is farther than `threshold` (default: always shape).
    """

    def __init__(
        self,
        env,
        potential,
        shaping_scale: float = 1.0,
        threshold: float     = float("inf"),
    ):
        super().__init__(env)
        self._potential     = potential
        self._shaping_scale = shaping_scale
        self._threshold     = threshold
        self._last_obs      = None

    def reset(self, **kwargs):
        obs, info      = self.env.reset(**kwargs)
        self._last_obs = _to_numpy_f32(obs).flatten()
        return obs, info

    def step(self, action):
        obs, r, terminated, truncated, info = self.env.step(action)
        obs_arr = _to_numpy_f32(obs).flatten()
        r_env = float(r)
        r_int = 0.0
        if self._last_obs is not None:
            sg_id, dist = _nearest_subgoal(obs_arr, self._potential.subgoals)
            if sg_id is not None and dist <= self._threshold:
                r_int = self._potential.get_intrinsic_reward(
                    self._last_obs, obs_arr, sg_id,
                )
                if not np.isfinite(r_int):
                    r_int = 0.0
                r = r_env + self._shaping_scale * r_int
        info["r_env"]     = r_env
        info["r_shaping"] = r_int
        self._last_obs = obs_arr
        return obs, float(r), terminated, truncated, info
