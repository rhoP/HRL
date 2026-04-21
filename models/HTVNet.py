import torch
import torch.nn as nn
import torch.nn.functional as F


class HittingTimeValueNet(nn.Module):
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

    def forward(self, s, S_prototype):
        """
        s:           [B, state_dim]
        S_prototype: [B, state_dim]
        Returns:     V_H(s; S) — [B, 1]
        """
        state_feat = self.state_encoder(s)
        set_feat   = self.set_encoder(S_prototype)
        return self.value_head(torch.cat([state_feat, set_feat], dim=-1))


class DeepSetHittingTimeNet(nn.Module):
    def __init__(self, state_dim, hidden_dim=256):
        super().__init__()
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
        )
        self.set_element_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, s, S_elements):
        """
        s:          [B, state_dim]
        S_elements: [B, N, state_dim]
        Returns:    V_H(s; S) — [B, 1]
        """
        state_feat   = self.state_encoder(s)
        element_feats = self.set_element_encoder(S_elements)
        set_feat     = element_feats.mean(dim=1)
        return self.value_head(torch.cat([state_feat, set_feat], dim=-1))
