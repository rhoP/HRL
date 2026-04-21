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


# ── Nearest-subgoal lookup ─────────────────────────────────────────────────

def _nearest_subgoal(s: np.ndarray, meta_subgoals: dict):
    """Return (sg_id, distance) of the meta-subgoal nearest to s."""
    best_id, best_d = None, float("inf")
    for sg_id, sg_data in meta_subgoals.items():
        d = float(np.linalg.norm(s - np.asarray(sg_data["state"], dtype=np.float32)))
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
    s_arr      = np.asarray(s,      dtype=np.float32).flatten()
    s_next_arr = np.asarray(s_next, dtype=np.float32).flatten()

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

    def __init__(self, landmarks: np.ndarray, simplices: dict, meta_subgoals: dict):
        self.landmarks = landmarks
        self.subgoals  = meta_subgoals

        self.G = nx.Graph()
        for i in range(len(landmarks)):
            self.G.add_node(i)
        for edge in simplices.get(1, []):
            self.G.add_edge(edge[0], edge[1])   # unit weight — hop count

        self._sg_landmarks: dict = {
            sg_id: int(np.argmin(np.linalg.norm(
                landmarks - np.asarray(sg_data["state"], dtype=np.float32), axis=1
            )))
            for sg_id, sg_data in meta_subgoals.items()
        }
        self._cache: dict = {}

    def _closest_landmark(self, s: np.ndarray) -> int:
        return int(np.argmin(np.linalg.norm(self.landmarks - s, axis=1)))

    def get_potential(self, s, sg_id) -> float:
        s_arr = np.asarray(s, dtype=np.float32).flatten()
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

    def get_potential(self, s, sg_id) -> float:
        trajs = self._trajs.get(sg_id, [])
        if not trajs:
            return 0.0
        s_arr   = np.asarray(s, dtype=np.float32).flatten()
        starts  = np.stack([t["start_state"] for t in trajs])
        dists   = np.linalg.norm(starts - s_arr, axis=1)
        nearest = np.argsort(dists)[: self.k]
        return float(np.mean([trajs[i]["backward_return"] for i in nearest]))

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

    Φ(s; task) = mean backward-discounted return of the k *visited states*
                 nearest to s, restricted to episodes that reached a terminal
                 state (done=True).

    Every step t of every terminal episode contributes one entry
    {"state": states[t], "back_ret": Σ_{i=t}^{hit} γ^{i-t} r_i}.
    This gives a phi landscape that varies across the state space even when
    all episodes start from the same initial state.

    Parameters
    ----------
    replay_source : object with iter_episodes()
        Each episode dict must contain keys:
          "states"  : np.ndarray [T+1, D]
          "rewards" : np.ndarray [T]
          "dones"   : np.ndarray [T, bool]
          "task_id" : int  (optional, defaults to 0)
    gamma   : discount factor
    k       : number of nearest neighbours
    """

    def __init__(self, replay_source, gamma: float = 0.99, k: int = 10):
        self.gamma = gamma
        self.k     = k
        # {task_id: [{"state": np.ndarray [D], "back_ret": float}]}
        self._data: dict = {}

        for ep in replay_source.iter_episodes():
            states     = np.asarray(ep["states"],  dtype=np.float32)
            rewards    = np.asarray(ep["rewards"], dtype=np.float32)
            # Use `terminated` if present; fall back to `dones` for old buffers.
            term_flags = np.asarray(ep.get("terminated", ep["dones"]), dtype=bool)
            tid        = int(ep.get("task_id", 0))
            T          = len(rewards)

            # Only use episodes that ended with a natural terminal (goal reached).
            # Timeout truncations have near-zero backward returns and pollute phi.
            hit = None
            for t in range(T):
                if term_flags[t]:
                    hit = t
                    break
            if hit is None:
                continue

            n = hit + 1   # steps that count toward the return

            # Backward returns for every visited step in O(n).
            # back_ret[t] = Σ_{i=t}^{hit} γ^{i-t} r_i
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
                })

    def phi(self, s: np.ndarray, task_id: int = 0) -> float:
        """Φ(s; task) — k-NN mean backward-discounted return from visited states."""
        entries = self._data.get(task_id, [])
        if not entries:
            return 0.0
        s_flat = np.asarray(s, dtype=np.float32).flatten()
        states = np.stack([e["state"] for e in entries])
        dists  = np.linalg.norm(states - s_flat, axis=1)
        nn_idx = np.argsort(dists)[: self.k]
        return float(np.mean([entries[i]["back_ret"] for i in nn_idx]))

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
            stats[tid] = {
                "n":    len(vals),
                "mean": float(vals.mean()),
                "std":  float(vals.std()),
            }
        return stats

    def all_task_ids(self) -> list:
        return list(self._data.keys())


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
        self._last_obs = np.asarray(obs, dtype=np.float32).flatten()
        return obs, info

    def step(self, action):
        obs, r, terminated, truncated, info = self.env.step(action)
        obs_arr = np.asarray(obs, dtype=np.float32).flatten()
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
