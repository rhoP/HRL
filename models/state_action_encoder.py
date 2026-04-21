"""
StateActionEncoder — joint (state, action) embedding for task-specific edge detection.
Trained via triplet contrastive loss on consecutive transitions.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class StateActionEncoder(nn.Module):
    """
    Joint (state, action) embedding.
    Consecutive (s_t, a_t) → (s_{t+1}, a_{t+1}) are positive pairs;
    random pairs are negatives.
    """

    def __init__(self, state_dim: int, action_dim: int, embedding_dim: int = 32):
        super().__init__()
        self.state_encoder  = nn.Linear(state_dim,  embedding_dim)
        self.action_encoder = nn.Linear(action_dim, embedding_dim)
        self.joint_encoder  = nn.Linear(embedding_dim * 2, embedding_dim)

    def forward(self, s: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        s_emb = F.relu(self.state_encoder(s))
        a_emb = F.relu(self.action_encoder(a))
        return self.joint_encoder(torch.cat([s_emb, a_emb], dim=-1))


def train_state_action_encoder(
    encoder: StateActionEncoder,
    replay_buffer,
    lr: float       = 1e-3,
    n_epochs: int   = 10,
    batch_size: int = 256,
    margin: float   = 1.0,
    device: str     = "cpu",
) -> list:
    """Train StateActionEncoder via triplet loss on consecutive transitions."""
    anchors_s, anchors_a, positives_s, positives_a = [], [], [], []
    for ep in replay_buffer.iter_episodes():
        states  = np.asarray(ep["states"], dtype=np.float32)
        actions = np.asarray(
            ep.get("actions", np.zeros((len(ep["rewards"]), 1), np.float32)),
            dtype=np.float32,
        )
        T = len(actions)
        for t in range(T - 1):
            anchors_s.append(states[t]);     anchors_a.append(actions[t])
            positives_s.append(states[t+1]); positives_a.append(actions[t+1])

    if len(anchors_s) < batch_size:
        return []

    as_t = torch.tensor(np.array(anchors_s,   dtype=np.float32), device=device)
    aa_t = torch.tensor(np.array(anchors_a,   dtype=np.float32), device=device)
    ps_t = torch.tensor(np.array(positives_s, dtype=np.float32), device=device)
    pa_t = torch.tensor(np.array(positives_a, dtype=np.float32), device=device)
    N    = len(as_t)

    opt          = optim.Adam(encoder.parameters(), lr=lr)
    epoch_losses = []
    encoder.train()

    for _ in range(n_epochs):
        perm = torch.randperm(N, device=device)
        bl   = []
        for start in range(0, N, batch_size):
            idx     = perm[start : start + batch_size]
            if len(idx) < 2:
                continue
            neg_idx = (idx + torch.randint(1, N, (len(idx),), device=device)) % N
            anch = encoder(as_t[idx],     aa_t[idx])
            pos  = encoder(ps_t[idx],     pa_t[idx])
            neg  = encoder(as_t[neg_idx], aa_t[neg_idx])
            loss = F.relu(
                (anch - pos).pow(2).sum(-1) - (anch - neg).pow(2).sum(-1) + margin
            ).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            bl.append(loss.item())
        if bl:
            epoch_losses.append(float(np.mean(bl)))

    encoder.eval()
    return epoch_losses
