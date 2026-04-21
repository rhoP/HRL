"""
Phase 1: Build (and refine) the Morse skeleton from a ReplayBuffer.

Wraps the pipeline in scripts/morse.py so the rest of the project
can call build_skeleton() without knowing the internals.
"""

import sys
import os
import torch
import torch.optim as optim

# Make scripts/morse.py importable
_SCRIPTS = os.path.join(os.path.dirname(__file__), "..", "scripts")
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

from morse import (                          # noqa: E402  (after sys.path insert)
    select_landmarks,
    build_witness_complex,
    BackwardValueNet,
    RewardNormalizer,
    train_backward_value_net,
    compute_morse_function,
    identify_critical_simplices,
)

from utils.utils import eval_mode            # noqa: E402  (after sys.path insert)


def build_skeleton(
    replay_buffer,
    state_dim: int,
    num_landmarks: int = 32,
    nu: int = 1,
    max_dim: int = 1,
    gamma: float = 0.99,
    hidden_dim: int = 64,
    set_embedding_dim: int = 16,
    lr: float = 3e-4,
    n_train_epochs: int = 5,
    batch_size: int = 32,
    threshold_percentile: float = 75.0,
    num_start_states: int = 32,
    target_clip: float = None,
    device: str = "cpu",
    verbose: bool = True,
) -> dict:
    """
    Build the Morse skeleton from experience stored in `replay_buffer`.

    Steps:
        1. Landmark selection via farthest-point sampling.
        2. Lazy witness complex on landmarks.
        3. Train BackwardValueNet on terminal episodes.
        4. Compute discrete Morse function.
        5. Identify critical simplices → subgoal states.

    Args:
        replay_buffer: ReplayBuffer (or any object with get_all_states()
                       and iter_episodes()).
        state_dim:     Observation dimension after flattening.

    Returns dict with keys:
        landmarks, simplices, critical_states, value_net,
        morse_values, phi_values, train_losses
    """
    if len(replay_buffer) == 0:
        raise RuntimeError("Replay buffer is empty; collect data before building skeleton.")

    if verbose:
        print("  [Skeleton] Selecting landmarks via FPS...")
    landmarks = select_landmarks(replay_buffer, num_landmarks=num_landmarks)

    if verbose:
        print("  [Skeleton] Building witness complex...")
    all_states = replay_buffer.get_all_states()
    simplices  = build_witness_complex(landmarks, all_states, nu=nu, max_dim=max_dim)
    if verbose:
        for d, sl in simplices.items():
            print(f"             dim {d}: {len(sl)} simplices")

    if verbose:
        print("  [Skeleton] Training BackwardValueNet...")

    reward_normalizer = RewardNormalizer()
    for ep in replay_buffer.iter_episodes():
        rw = ep["rewards"]
        reward_normalizer.update(rw.tolist() if hasattr(rw, "tolist") else list(rw))
    if verbose:
        print(f"  [Skeleton] Reward stats: mean={reward_normalizer.mean:.4f}  "
              f"std={reward_normalizer.std:.4f}")

    value_net = BackwardValueNet(
        state_dim, set_embedding_dim=set_embedding_dim, hidden_dim=hidden_dim
    ).to(device)
    optimizer = optim.Adam(value_net.parameters(), lr=lr)
    train_losses = train_backward_value_net(
        replay_buffer, value_net, optimizer,
        gamma=gamma, n_epochs=n_train_epochs,
        batch_size=batch_size, device=device,
        reward_normalizer=reward_normalizer,
        target_clip=target_clip,
    )

    if verbose:
        last = f"{train_losses[-1]:.6f}" if train_losses else "N/A"
        print(f"  [Skeleton] Final backward-value loss: {last}")
        print("  [Skeleton] Computing discrete Morse function...")

    n_start = min(num_start_states, len(all_states))
    perm = torch.randperm(len(all_states))[:n_start]
    start_states    = all_states[perm].to(device)
    landmark_states = landmarks.to(device)

    with eval_mode(value_net):
        morse_values, phi_values = compute_morse_function(
            simplices, landmark_states, start_states, value_net
        )

    if verbose:
        print("  [Skeleton] Identifying critical simplices...")
    critical = identify_critical_simplices(
        simplices, morse_values, threshold_percentile=threshold_percentile
    )

    # Extract landmark states at critical 0-simplices as subgoal states
    critical_states: dict = {}
    for sigma in critical.get(0, []):
        v = sigma[0]
        critical_states[v] = landmarks[v].cpu().numpy()

    if verbose:
        print(f"  [Skeleton] {len(critical_states)} critical state(s) found.")

    return {
        "landmarks":       landmarks,
        "simplices":       simplices,
        "critical_states": critical_states,
        "value_net":       value_net,
        "morse_values":    morse_values,
        "phi_values":      phi_values,
        "train_losses":    train_losses,
    }


def refine_skeleton(skeleton_data: dict, replay_buffer, **kwargs) -> dict:
    """
    Rebuild the skeleton using the current (larger) replay buffer.

    Any keyword argument accepted by build_skeleton() can be passed to
    override the defaults; state_dim is inferred from the buffer.
    """
    state_dim = replay_buffer.state_dim
    if state_dim is None:
        raise RuntimeError("Cannot infer state_dim from an empty ReplayBuffer.")
    return build_skeleton(replay_buffer, state_dim=state_dim, **kwargs)
