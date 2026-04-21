"""
Flat replay buffer supporting both transition-level and episode-level sampling.

Implements the interfaces expected by:
  - scripts/morse.py::select_landmarks()       → get_all_states()
  - scripts/morse.py::train_backward_value_net() → iter_episodes()
  - algos/HTTD.py::train_hitting_time_value_net() → sample(batch_size)
"""

import random
import numpy as np
import torch


class ReplayBuffer:
    """
    Replay buffer stored as parallel Python lists (capped at `capacity`).

    Episode boundaries are tracked so complete trajectories can be recovered
    for Monte-Carlo returns and topology-discovery pipelines.
    """

    def __init__(self, capacity: int = 200_000, device: str = "cpu"):
        self.capacity = capacity
        self.device   = device

        # Flat transition storage
        self._states:      list = []
        self._actions:     list = []
        self._rewards:     list = []
        self._next_states: list = []
        self._dones:       list = []
        self._task_ids:    list = []

        # _ep_ends[i] = exclusive end index (into _states) of episode i
        self._ep_ends: list = []

        self._state_dim:  int | None = None
        self._action_dim: int | None = None

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def state_dim(self) -> int | None:
        return self._state_dim

    @property
    def action_dim(self) -> int | None:
        return self._action_dim

    def __len__(self) -> int:
        return len(self._states)

    # ------------------------------------------------------------------
    # Adding data
    # ------------------------------------------------------------------

    def push(self, s, a, r: float, s_next, done: bool, task_id: int = 0):
        """Add a single transition. Silently drops when at capacity."""
        if len(self._states) >= self.capacity:
            return

        s      = np.asarray(s,      dtype=np.float32).flatten()
        a      = np.asarray(a,      dtype=np.float32).flatten()
        s_next = np.asarray(s_next, dtype=np.float32).flatten()

        if self._state_dim is None:
            self._state_dim  = int(s.shape[0])
            self._action_dim = int(a.shape[0])

        self._states.append(s)
        self._actions.append(a)
        self._rewards.append(float(r))
        self._next_states.append(s_next)
        self._dones.append(bool(done))
        self._task_ids.append(int(task_id))

        if bool(done):
            self._ep_ends.append(len(self._states))

    def push_trajectory(self, traj: dict, task_id: int = 0):
        """
        Add a full trajectory.

        Expected keys:
            states  [T+1, state_dim]
            actions [T, action_dim]  (or [T])
            rewards [T]
            dones   [T]   (last entry True → natural terminal)
        Optional: task_id (overrides argument)
        """
        states  = np.asarray(traj["states"],  dtype=np.float32)
        actions = np.asarray(traj["actions"], dtype=np.float32)
        rewards = np.asarray(traj["rewards"], dtype=np.float32)
        dones   = np.asarray(traj["dones"],   dtype=bool)
        tid     = int(traj.get("task_id", task_id))
        T = len(rewards)
        for t in range(T):
            self.push(states[t], actions[t], float(rewards[t]),
                      states[t + 1], bool(dones[t]), tid)

    def add_trajectories(self, trajectories: list, task_id: int = 0):
        """Convenience wrapper: push a list of trajectory dicts."""
        for traj in trajectories:
            self.push_trajectory(traj, task_id)

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def sample(self, batch_size: int) -> dict:
        """
        Sample `batch_size` random transitions.

        Returns dict of float32 tensors on self.device:
            state, action, reward [B,1], next_state, done [B,1], task_id [B] (long)
        """
        N = len(self._states)
        if N == 0:
            raise RuntimeError("ReplayBuffer is empty.")
        idxs = np.random.randint(0, N, size=batch_size)

        def _t(lst):
            return torch.tensor(np.stack([lst[i] for i in idxs]),
                                dtype=torch.float32, device=self.device)

        return {
            "state":      _t(self._states),
            "action":     _t(self._actions),
            "reward":     torch.tensor([self._rewards[i]   for i in idxs],
                                       dtype=torch.float32, device=self.device).unsqueeze(1),
            "next_state": _t(self._next_states),
            "done":       torch.tensor([float(self._dones[i]) for i in idxs],
                                       dtype=torch.float32, device=self.device).unsqueeze(1),
            "task_id":    torch.tensor([self._task_ids[i]  for i in idxs],
                                       dtype=torch.long,    device=self.device),
        }

    def sample_trajectories(self, n: int) -> list:
        """
        Sample up to `n` complete episodes.

        Returns list of dicts: states[T+1], actions[T], rewards[T], dones[T], task_id.
        """
        if not self._ep_ends:
            return []
        n = min(n, len(self._ep_ends))
        chosen = random.sample(range(len(self._ep_ends)), n)
        out = []
        for ep_idx in chosen:
            start = self._ep_ends[ep_idx - 1] if ep_idx > 0 else 0
            end   = self._ep_ends[ep_idx]          # exclusive
            T     = end - start
            if T == 0:
                continue
            ep_states  = np.stack(
                [self._states[start + t] for t in range(T)] + [self._next_states[end - 1]]
            )
            ep_actions = np.stack([self._actions[start + t] for t in range(T)])
            ep_rewards = np.array([self._rewards[start + t] for t in range(T)], dtype=np.float32)
            ep_dones   = np.array([self._dones[start + t]   for t in range(T)], dtype=bool)
            out.append({
                "states":  ep_states,
                "actions": ep_actions,
                "rewards": ep_rewards,
                "dones":   ep_dones,
                "task_id": self._task_ids[start],
            })
        return out

    def iter_episodes(self):
        """
        Yield every complete episode as a dict.

        Compatible with scripts/morse.py::train_backward_value_net().
        Each dict: states [T+1, state_dim], rewards [T], dones [T].
        """
        for ep_idx, ep_end in enumerate(self._ep_ends):
            start = self._ep_ends[ep_idx - 1] if ep_idx > 0 else 0
            T     = ep_end - start
            if T == 0:
                continue
            ep_states  = np.stack(
                [self._states[start + t] for t in range(T)] + [self._next_states[ep_end - 1]]
            )
            ep_rewards = np.array([self._rewards[start + t] for t in range(T)], dtype=np.float32)
            ep_dones   = np.array([self._dones[start + t]   for t in range(T)], dtype=bool)
            yield {"states": ep_states, "rewards": ep_rewards, "dones": ep_dones}

    def get_all_states(self) -> torch.Tensor:
        """
        All stored states as a float32 tensor [N, state_dim].

        Compatible with scripts/morse.py::select_landmarks().
        """
        if not self._states:
            d = self._state_dim or 1
            return torch.zeros(0, d, dtype=torch.float32, device=self.device)
        arr = np.stack(self._states, axis=0).astype(np.float32)
        return torch.tensor(arr, dtype=torch.float32, device=self.device)
