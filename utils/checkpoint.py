"""
Checkpoint and data-persistence utilities.

Saves / loads:
  - meta_policy + meta_value_net   (torch state dicts)
  - sub_policies                   (per-subgoal state dicts + metadata)
  - hitting_nets                   (per-subgoal state dicts)
  - skeleton_data                  (pickle; includes BackwardValueNet)
  - replay_buffer                  (numpy npz)
  - metrics                        (JSON)

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
    sub_policies: dict,
    hitting_nets: dict,
    skeleton_data: dict,
    replay_buffer,
    metrics: dict,
) -> str:
    """
    Write a full checkpoint to <save_dir>/iter_<N>/.
    Returns the directory path of the saved checkpoint.
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

    # Sub-policies
    sp_data = {}
    for c_id, sp in sub_policies.items():
        sp_data[c_id] = {
            "policy_net":       sp.policy_net.state_dict(),
            "value_net":        sp.value_net.state_dict(),
            "target_value_net": sp.target_value_net.state_dict(),
            "c_np":             sp.c.cpu().numpy(),
            "threshold":        sp.threshold,
            "max_steps":        sp.max_steps,
            "reach_bonus":      sp.reach_bonus,
            "discrete":         sp.discrete,
        }
    torch.save(sp_data, os.path.join(ckpt_dir, "sub_policies.pt"))

    # Hitting-time nets
    hn_data = {c_id: net.state_dict() for c_id, net in hitting_nets.items()}
    torch.save(hn_data, os.path.join(ckpt_dir, "hitting_nets.pt"))

    # Skeleton (BackwardValueNet weights + topology arrays — pickle is fine here)
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
        "train_losses":    skeleton_data.get("train_losses", []),
        "value_net_state": skeleton_data["value_net"].state_dict(),
        "value_net_kwargs": {
            # index 0 = first Linear in each Sequential (unchanged by BatchNorm insertion)
            "hidden_dim":        skeleton_data["value_net"].state_encoder[0].out_features,
            "set_embedding_dim": skeleton_data["value_net"].set_encoder[0].out_features,
        },
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
        meta_policy_state, meta_value_net_state, sub_policies_data,
        hitting_nets_data, skeleton_save, metrics
    """
    result = {}

    _load = lambda p: torch.load(p, map_location=device, weights_only=False)

    mp_path = os.path.join(checkpoint_dir, "meta_policy.pt")
    if os.path.exists(mp_path):
        result["meta_policy_state"]    = _load(mp_path)
        result["meta_value_net_state"] = _load(
            os.path.join(checkpoint_dir, "meta_value_net.pt"))

    result["sub_policies_data"] = _load(
        os.path.join(checkpoint_dir, "sub_policies.pt"))
    result["hitting_nets_data"] = _load(
        os.path.join(checkpoint_dir, "hitting_nets.pt"))

    with open(os.path.join(checkpoint_dir, "skeleton.pkl"), "rb") as f:
        result["skeleton_save"] = pickle.load(f)

    return result


def restore_models(ckpt: dict, state_dim: int, action_dim: int,
                   device: str = "cpu"):
    """
    Reconstruct live model objects from a loaded checkpoint dict.

    Returns (meta_policy, meta_value_net, sub_policies, hitting_nets, skeleton_data).
    """
    import sys, os
    _ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _ROOT not in sys.path:
        sys.path.insert(0, _ROOT)

    from algos.MetaPolicy  import MetaPolicy, MetaValueNetwork, SubPolicy
    from models.HTVNet     import HittingTimeValueNet
    from models.networks   import PolicyNetwork, ValueNetwork

    skel_save        = ckpt["skeleton_save"]
    critical_states  = skel_save["critical_states"]
    num_subgoals     = len(critical_states)
    c_list           = list(critical_states.keys())

    # Meta-policy
    meta_policy = meta_value_net = None
    if "meta_policy_state" in ckpt:
        meta_policy = MetaPolicy(state_dim, num_subgoals).to(device)
        meta_policy.load_state_dict(ckpt["meta_policy_state"])
        meta_policy.eval()
        meta_value_net = MetaValueNetwork(state_dim).to(device)
        meta_value_net.load_state_dict(ckpt["meta_value_net_state"])
        meta_value_net.eval()

    # Sub-policies
    sub_policies = {}
    for c_id, sp_data in ckpt["sub_policies_data"].items():
        sp = SubPolicy(
            c_id, sp_data["c_np"],
            state_dim=state_dim, action_dim=action_dim,
            discrete=sp_data.get("discrete", False),
            threshold=sp_data.get("threshold", 0.1),
            max_steps=sp_data.get("max_steps", 100),
            reach_bonus=sp_data.get("reach_bonus", 1.0),
            device=device,
        )
        sp.policy_net.load_state_dict(sp_data["policy_net"])
        sp.value_net.load_state_dict(sp_data["value_net"])
        sp.target_value_net.load_state_dict(sp_data["target_value_net"])
        sp.policy_net.eval()
        sub_policies[c_id] = sp

    # Hitting-time nets
    hitting_nets = {}
    for c_id, state_dict in ckpt["hitting_nets_data"].items():
        net = HittingTimeValueNet(state_dim).to(device)
        net.load_state_dict(state_dict)
        net.eval()
        hitting_nets[c_id] = net

    # Skeleton data
    import sys as _sys, os as _os
    _scripts = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "scripts")
    if _scripts not in _sys.path:
        _sys.path.insert(0, _scripts)
    from morse import BackwardValueNet as _BVNet
    bv_kwargs = skel_save.get("value_net_kwargs", {})
    bv_net = _BVNet(state_dim, **bv_kwargs)
    bv_net.load_state_dict(skel_save["value_net_state"])
    bv_net.to(device)
    bv_net.eval()

    landmarks = torch.tensor(skel_save["landmarks"], dtype=torch.float32, device=device)
    morse_vals = {k: torch.tensor(v, dtype=torch.float32, device=device)
                  if isinstance(v, np.ndarray) else v
                  for k, v in skel_save["morse_values"].items()}
    phi_vals   = {k: torch.tensor(v, dtype=torch.float32, device=device)
                  if isinstance(v, np.ndarray) else v
                  for k, v in skel_save["phi_values"].items()}

    skeleton_data = {
        "landmarks":       landmarks,
        "simplices":       skel_save["simplices"],
        "critical_states": critical_states,
        "value_net":       bv_net,
        "morse_values":    morse_vals,
        "phi_values":      phi_vals,
        "train_losses":    skel_save["train_losses"],
    }

    return meta_policy, meta_value_net, sub_policies, hitting_nets, skeleton_data


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
