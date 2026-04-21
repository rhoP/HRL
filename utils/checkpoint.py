"""
Checkpoint and data-persistence utilities.

Saves / loads:
  - meta_policy + meta_value_net   (torch state dicts)
  - task_policies                  (SB3 models, one zip per task)
  - skeleton_data                  (pickle; includes KNNBackwardEstimator)
  - replay_buffer                  (numpy npz)
  - metrics                        (JSON)

The potential function (Phase 2) is not stored — it is deterministically
reconstructed from skeleton_data + replay_buffer at restore time.

Best-model tracking:
  Call update_best(metric_value, ...) after each iteration;
  the checkpoint is hard-linked / copied to <save_dir>/best/ when improved.
"""

import os
import json
import pickle
import shutil

import numpy as np
import torch


# ── Replay buffer ──────────────────────────────────────────────────────────

def save_replay_buffer(rb, path: str) -> None:
    """Serialize the replay buffer as a compressed numpy archive."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    np.savez_compressed(
        path,
        states      = np.stack(rb._states)      if rb._states      else np.empty((0,)),
        actions     = np.stack(rb._actions)     if rb._actions     else np.empty((0,)),
        rewards     = np.array(rb._rewards,     dtype=np.float32),
        next_states = np.stack(rb._next_states) if rb._next_states else np.empty((0,)),
        dones       = np.array(rb._dones,       dtype=bool),
        terminated  = np.array(rb._terminated,  dtype=bool),
        task_ids    = np.array(rb._task_ids,    dtype=np.int64),
        ep_ends     = np.array(rb._ep_ends,     dtype=np.int64),
    )


def load_replay_buffer(path: str, device: str = "cpu"):
    """Reconstruct a ReplayBuffer from a saved npz file."""
    from utils.replay_buffer import ReplayBuffer
    data = np.load(path, allow_pickle=False)
    rb   = ReplayBuffer(device=device)

    states      = data["states"]
    actions     = data["actions"]
    rewards     = data["rewards"]
    next_states = data["next_states"]
    dones       = data["dones"]
    # backward-compat: old buffers without terminated field → fall back to dones
    terminated  = data["terminated"] if "terminated" in data else dones
    task_ids    = data["task_ids"]
    ep_ends     = data["ep_ends"].tolist()

    if states.ndim == 1 and len(states) == 0:
        return rb

    rb._state_dim  = states.shape[1]
    rb._action_dim = actions.shape[1] if actions.ndim > 1 else 1

    rb._states      = list(states)
    rb._actions     = list(actions)
    rb._rewards     = list(rewards.astype(float))
    rb._next_states = list(next_states)
    rb._dones       = list(dones.astype(bool))
    rb._terminated  = list(terminated.astype(bool))
    rb._task_ids    = list(task_ids.astype(int))
    rb._ep_ends     = ep_ends
    return rb


# ── Checkpoint save / load ─────────────────────────────────────────────────

def save_checkpoint(
    save_dir: str,
    *,
    iteration: int,
    meta_policy,
    meta_value_net,
    task_policies: dict = None,
    skeleton_data: dict,
    replay_buffer,
    metrics: dict,
) -> str:
    """
    Write a full checkpoint to <save_dir>/iter_<N>/.
    Returns the directory path of the saved checkpoint.

    task_policies: {task_id: SB3 model} — each model is saved as a zip file.
    The potential function is not checkpointed; it is derived from skeleton_data
    and the replay buffer (both already persisted) at restore time.
    """
    ckpt_dir = os.path.join(save_dir, f"iter_{iteration:03d}")
    os.makedirs(ckpt_dir, exist_ok=True)

    # Meta-policy
    if meta_policy is not None:
        torch.save(meta_policy.state_dict(),
                   os.path.join(ckpt_dir, "meta_policy.pt"))
    if meta_value_net is not None:
        torch.save(meta_value_net.state_dict(),
                   os.path.join(ckpt_dir, "meta_value_net.pt"))

    # Task policies (SB3 models)
    tp_dir = os.path.join(ckpt_dir, "task_policies")
    os.makedirs(tp_dir, exist_ok=True)
    tp_manifest: dict = {}
    for task_id, model in (task_policies or {}).items():
        algo_name  = type(model).__name__          # "PPO" or "SAC"
        model_path = os.path.join(tp_dir, f"task_{task_id}")
        model.save(model_path)                     # writes task_<id>.zip
        tp_manifest[str(task_id)] = {"algo": algo_name}
    with open(os.path.join(tp_dir, "manifest.json"), "w") as f:
        json.dump(tp_manifest, f, indent=2)

    # Skeleton (topology arrays + k-NN estimator — pickle is fine here)
    skel_save = {
        "landmarks":       skeleton_data["landmarks"].cpu().numpy(),
        "simplices":       skeleton_data["simplices"],
        "critical_states": skeleton_data["critical_states"],
        "morse_values":    {
            k: (v.cpu().numpy() if hasattr(v, "cpu") else v)
            for k, v in skeleton_data.get("morse_values", {}).items()
        },
        "phi_values":      {
            k: (v.cpu().numpy() if hasattr(v, "cpu") else v)
            for k, v in skeleton_data.get("phi_values", {}).items()
        },
        "knn_estimator":   skeleton_data.get("knn_estimator"),
    }
    with open(os.path.join(ckpt_dir, "skeleton.pkl"), "wb") as f:
        pickle.dump(skel_save, f)

    # Replay buffer (shared across iterations — save once and overwrite)
    save_replay_buffer(replay_buffer,
                       os.path.join(save_dir, "replay_buffer.npz"))

    # Metrics
    metrics_path = os.path.join(save_dir, "metrics.json")
    existing = {}
    if os.path.exists(metrics_path):
        with open(metrics_path) as f:
            existing = json.load(f)
    existing[str(iteration)] = metrics
    with open(metrics_path, "w") as f:
        json.dump(existing, f, indent=2)

    return ckpt_dir


def load_checkpoint(checkpoint_dir: str, device: str = "cpu") -> dict:
    """
    Load a checkpoint saved by save_checkpoint().

    Returns dict with keys:
        meta_policy_state, meta_value_net_state,
        task_policies_manifest, task_policies_dir, skeleton_save
    """
    result = {}

    _load = lambda p: torch.load(p, map_location=device, weights_only=False)

    mp_path = os.path.join(checkpoint_dir, "meta_policy.pt")
    if os.path.exists(mp_path):
        result["meta_policy_state"]    = _load(mp_path)
        result["meta_value_net_state"] = _load(
            os.path.join(checkpoint_dir, "meta_value_net.pt"))

    # Task policies — store the directory path; restore_models loads them
    tp_dir = os.path.join(checkpoint_dir, "task_policies")
    if os.path.isdir(tp_dir):
        result["task_policies_dir"] = tp_dir
        manifest_path = os.path.join(tp_dir, "manifest.json")
        if os.path.exists(manifest_path):
            with open(manifest_path) as f:
                result["task_policies_manifest"] = json.load(f)

    with open(os.path.join(checkpoint_dir, "skeleton.pkl"), "rb") as f:
        result["skeleton_save"] = pickle.load(f)

    return result


def restore_models(ckpt: dict, state_dim: int, action_dim: int,
                   discrete: bool = True, device: str = "cpu"):
    """
    Reconstruct live model objects from a loaded checkpoint dict.

    Returns (meta_policy, meta_value_net, task_policies, skeleton_data).
    task_policies is a dict {task_id: SB3 model}.
    """
    import sys, os
    _ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _ROOT not in sys.path:
        sys.path.insert(0, _ROOT)

    from models.meta_policy_net import MetaPolicy, MetaValueNetwork

    skel_save = ckpt["skeleton_save"]

    # Meta-policy
    meta_policy = meta_value_net = None
    if "meta_policy_state" in ckpt:
        meta_policy = MetaPolicy(state_dim, action_dim,
                                 discrete=discrete).to(device)
        meta_policy.load_state_dict(ckpt["meta_policy_state"])
        meta_policy.eval()
        meta_value_net = MetaValueNetwork(state_dim,
                                          meta_policy.gru_hidden).to(device)
        meta_value_net.load_state_dict(ckpt["meta_value_net_state"])
        meta_value_net.eval()

    # Task policies (SB3)
    task_policies: dict = {}
    manifest = ckpt.get("task_policies_manifest", {})
    tp_dir   = ckpt.get("task_policies_dir", "")
    if manifest and tp_dir:
        from stable_baselines3 import PPO, SAC
        _algo_map = {"PPO": PPO, "SAC": SAC}
        for task_id_str, info in manifest.items():
            AlgoCls   = _algo_map.get(info.get("algo", "PPO"), PPO)
            zip_path  = os.path.join(tp_dir, f"task_{task_id_str}.zip")
            if os.path.exists(zip_path):
                task_policies[int(task_id_str)] = AlgoCls.load(
                    zip_path, device=device)

    # Skeleton data
    landmarks  = torch.tensor(skel_save["landmarks"], dtype=torch.float32, device=device)
    morse_vals = {k: torch.tensor(v, dtype=torch.float32, device=device)
                  if isinstance(v, np.ndarray) else v
                  for k, v in skel_save.get("morse_values", {}).items()}
    phi_vals   = {k: torch.tensor(v, dtype=torch.float32, device=device)
                  if isinstance(v, np.ndarray) else v
                  for k, v in skel_save.get("phi_values", {}).items()}

    skeleton_data = {
        "landmarks":       landmarks,
        "simplices":       skel_save["simplices"],
        "critical_states": skel_save["critical_states"],
        "knn_estimator":   skel_save.get("knn_estimator"),
        "morse_values":    morse_vals,
        "phi_values":      phi_vals,
        "train_losses":    [],
    }

    return meta_policy, meta_value_net, task_policies, skeleton_data


# ── Best-model tracker ─────────────────────────────────────────────────────

class BestModelTracker:
    """
    Copies the latest checkpoint to <save_dir>/best/ whenever the tracked
    metric improves.
    """

    def __init__(self, save_dir: str, higher_is_better: bool = True):
        self.save_dir        = save_dir
        self.higher_is_better = higher_is_better
        self._best           = -float("inf") if higher_is_better else float("inf")
        self.best_dir        = os.path.join(save_dir, "best")

    def update(self, metric: float, ckpt_dir: str) -> bool:
        """
        Compare metric against best so far. If improved, copy checkpoint to
        best/. Returns True when a new best is saved.
        """
        improved = (metric > self._best) if self.higher_is_better \
                   else (metric < self._best)
        if improved:
            self._best = metric
            if os.path.exists(self.best_dir):
                shutil.rmtree(self.best_dir)
            shutil.copytree(ckpt_dir, self.best_dir)
            best_meta = os.path.join(self.save_dir, "best_metric.json")
            with open(best_meta, "w") as f:
                json.dump({"metric": metric, "source": ckpt_dir}, f, indent=2)
        return improved
