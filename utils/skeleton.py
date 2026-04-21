"""
Phase 1: Build (and refine) the Meta-Morse skeleton from a ReplayBuffer.

Delegates to build_meta_morse_complex() in scripts/morse.py which runs the
full Meta-MorseComplex pipeline:
  Step 1 — Stratified terminal sampling + FPS landmark selection
  Step 2 — Per-task SA encoder + witness complex + persistent critical simplices
  Step 3 — DBSCAN soft intersection → meta-subgoals → z-score meta-potential
"""

import sys
import os

_SCRIPTS = os.path.join(os.path.dirname(__file__), "..", "scripts")
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

from morse import build_meta_morse_complex   # noqa: E402


def build_skeleton(
    replay_buffer,
    state_dim: int,
    action_dim: int              = None,
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
    Build the Meta-Morse skeleton from experience stored in `replay_buffer`.

    Thin wrapper around build_meta_morse_complex().  All arguments are
    forwarded verbatim; action_dim defaults to replay_buffer.action_dim.

    Returns dict with keys:
        landmarks, landmark_meta, simplices, simplex_task_ids,
        edge_action_labels, critical_states, phi_critical,
        meta_subgoals, meta_phi, task_critical_states, knn_estimator,
        sa_encoder, morse_values, phi_values, phi_values_denorm, train_losses
    """
    if len(replay_buffer) == 0:
        raise RuntimeError("Replay buffer is empty; collect data before building skeleton.")

    a_dim = action_dim
    if a_dim is None:
        a_dim = replay_buffer.action_dim or 1

    return build_meta_morse_complex(
        replay_buffer,
        state_dim=state_dim,
        action_dim=a_dim,
        num_landmarks=num_landmarks,
        nu=nu,
        max_dim=max_dim,
        gamma=gamma,
        sa_embedding_dim=sa_embedding_dim,
        batch_size=batch_size,
        threshold_percentile=threshold_percentile,
        min_task_support=min_task_support,
        persistence_threshold=persistence_threshold,
        centrality_threshold=centrality_threshold,
        sa_lr=sa_lr,
        sa_epochs=sa_epochs,
        knn_k=knn_k,
        state_projection_fn=state_projection_fn,
        device=device,
        verbose=verbose,
    )


def refine_skeleton(skeleton_data: dict, replay_buffer, **kwargs) -> dict:
    """
    Rebuild the skeleton using the current (larger) replay buffer.

    state_dim and action_dim are inferred from the buffer; any keyword
    accepted by build_skeleton() can be passed to override the defaults.
    """
    state_dim  = replay_buffer.state_dim
    action_dim = kwargs.pop("action_dim", replay_buffer.action_dim or 1)
    if state_dim is None:
        raise RuntimeError("Cannot infer state_dim from an empty ReplayBuffer.")
    return build_skeleton(
        replay_buffer,
        state_dim=state_dim,
        action_dim=action_dim,
        **kwargs,
    )
