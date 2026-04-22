"""
Progress-estimation reward shaping.

StateEncoder          — MLP projecting raw states to a latent space
ProgressEstimator     — predicts task progress ∈ [0, 1] from latent state
ProgressPotential     — ShapedRewardWrapper-compatible potential derived from
                        the progress score difference Φ(s') − Φ(s)
MixedPotential        — blends any base potential with ProgressPotential
train_progress_estimator — trains encoder + estimator via temporal ranking loss
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Encoder ────────────────────────────────────────────────────────────────

class StateEncoder(nn.Module):
    def __init__(self, state_dim: int, latent_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, latent_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ── Progress estimator ─────────────────────────────────────────────────────

class ProgressEstimator(nn.Module):
    """
    Predicts "how close" the agent is to task completion.
    Trained via temporal ordering: states later in successful trajectories
    should have higher progress scores.
    """

    def __init__(self, latent_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.net(z))

    def ranking_loss(
        self, z_early: torch.Tensor, z_late: torch.Tensor
    ) -> torch.Tensor:
        """z_late should have strictly higher progress than z_early (margin 0.1)."""
        return F.relu(self.forward(z_early) - self.forward(z_late) + 0.1).mean()


# ── Potential wrappers ─────────────────────────────────────────────────────

class ProgressPotential:
    """
    Potential Φ(s) = progress_estimator(encoder(s)) ∈ [0, 1].

    Compatible with ShapedRewardWrapper.  A single dummy subgoal at the
    origin is used so the wrapper's nearest-subgoal lookup always fires;
    sg_id is ignored in get_intrinsic_reward.
    """

    def __init__(
        self,
        encoder: StateEncoder,
        estimator: ProgressEstimator,
        state_dim: int,
        device: str = "cpu",
    ):
        self.encoder   = encoder.to(device)
        self.estimator = estimator.to(device)
        self._device   = device
        self.subgoals  = {0: {"state": np.zeros(state_dim, dtype=np.float32)}}
        self.encoder.eval()
        self.estimator.eval()

    def _phi(self, s) -> float:
        if hasattr(s, "detach"):
            s_np = s.detach().cpu().numpy().astype(np.float32)
        else:
            s_np = np.asarray(s, dtype=np.float32).flatten()
        x = torch.from_numpy(s_np).unsqueeze(0).to(self._device)
        with torch.no_grad():
            return float(self.estimator(self.encoder(x)).item())

    def get_potential(self, s, sg_id=None) -> float:
        return self._phi(s)

    def get_intrinsic_reward(self, s, s_next, sg_id=None) -> float:
        return self._phi(s_next) - self._phi(s)


class MixedPotential:
    """
    r_int = base_weight  · base.get_intrinsic_reward(s, s', sg_id)
          + prog_weight  · progress.get_intrinsic_reward(s, s')

    Compatible with ShapedRewardWrapper.  Uses subgoals from the base
    potential when present, otherwise the dummy subgoal from progress.
    Falls back to pure progress shaping when base is None.
    """

    def __init__(
        self,
        base_potential,
        progress_potential: ProgressPotential,
        base_weight: float = 0.5,
        prog_weight: float = 0.5,
    ):
        self._base     = base_potential
        self._progress = progress_potential
        self._bw       = base_weight
        self._pw       = prog_weight
        self.subgoals  = (
            base_potential.subgoals
            if base_potential is not None
            else progress_potential.subgoals
        )

    def get_potential(self, s, sg_id=None) -> float:
        base_val = (
            self._base.get_potential(s, sg_id) * self._bw
            if self._base is not None else 0.0
        )
        return base_val + self._progress.get_potential(s) * self._pw

    def get_intrinsic_reward(self, s, s_next, sg_id=None) -> float:
        base_r = (
            self._base.get_intrinsic_reward(s, s_next, sg_id) * self._bw
            if self._base is not None else 0.0
        )
        return base_r + self._progress.get_intrinsic_reward(s, s_next) * self._pw


# ── Training ───────────────────────────────────────────────────────────────

def train_progress_estimator(
    replay_buffer,
    state_dim: int,
    latent_dim: int     = 64,
    epochs: int         = 10,
    lr: float           = 1e-3,
    batch_size: int     = 256,
    n_pairs_per_ep: int = 20,
    survived_only: bool = True,
    device: str         = "cpu",
    verbose: bool       = True,
) -> tuple:
    """
    Train a StateEncoder + ProgressEstimator on temporal ranking pairs sampled
    from the replay buffer.

    survived_only=True (MuJoCo default): only use episodes that ran to the
    time limit (no early termination = survival), since those have meaningful
    progress structure.  Falls back to all episodes when none are available.

    Returns (encoder, estimator) ready for ProgressPotential, or (None, None)
    if no usable episodes were found.
    """
    pairs: list = []
    rng = np.random.default_rng(seed=42)

    for ep in replay_buffer.iter_episodes():
        states     = np.asarray(ep["states"], dtype=np.float32)
        term_flags = np.asarray(ep.get("terminated", ep["dones"]), dtype=bool)
        survived   = not any(term_flags)
        if survived_only and not survived:
            continue
        T = len(states)
        if T < 2:
            continue
        n = min(n_pairs_per_ep, T * (T - 1) // 2)
        for _ in range(n):
            t_early = int(rng.integers(0, T - 1))
            t_late  = int(rng.integers(t_early + 1, T))
            pairs.append((states[t_early], states[t_late]))

    # IS fallback: no survived episodes → use all episodes
    if not pairs and survived_only:
        if verbose:
            print("  [Progress] No survived episodes — using all episodes as fallback.")
        return train_progress_estimator(
            replay_buffer, state_dim,
            latent_dim=latent_dim, epochs=epochs, lr=lr,
            batch_size=batch_size, n_pairs_per_ep=n_pairs_per_ep,
            survived_only=False, device=device, verbose=verbose,
        )

    if not pairs:
        if verbose:
            print("  [Progress] No training pairs found — skipping progress estimator.")
        return None, None

    encoder   = StateEncoder(state_dim, latent_dim).to(device)
    estimator = ProgressEstimator(latent_dim).to(device)
    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(estimator.parameters()), lr=lr
    )
    encoder.train()
    estimator.train()

    if verbose:
        print(f"  [Progress] Training on {len(pairs)} pairs  "
              f"latent_dim={latent_dim}  epochs={epochs}")

    for epoch in range(epochs):
        rng.shuffle(pairs)
        epoch_loss = 0.0
        n_batches  = 0
        for i in range(0, len(pairs), batch_size):
            batch   = pairs[i : i + batch_size]
            early_t = torch.tensor(
                np.stack([p[0] for p in batch]), dtype=torch.float32
            ).to(device)
            late_t  = torch.tensor(
                np.stack([p[1] for p in batch]), dtype=torch.float32
            ).to(device)
            z_early = encoder(early_t)
            z_late  = encoder(late_t)
            loss    = estimator.ranking_loss(z_early, z_late)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches  += 1

        if verbose and (epoch == 0 or (epoch + 1) % 5 == 0 or epoch == epochs - 1):
            print(f"  [Progress] epoch {epoch + 1:3d}/{epochs}  "
                  f"loss={epoch_loss / max(n_batches, 1):.4f}")

    encoder.eval()
    estimator.eval()
    return encoder, estimator
