import torch
import torch.nn as nn


class RewardModel(nn.Module):
    """
    Predicts reward from latent features.

    Input:
        features = concat(z_t, h_t), shape [B, z_dim + h_dim]

    Output:
        reward prediction, shape [B, 1]
    """

    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 2,
    ):
        super().__init__()

        layers = []
        dim = feature_dim

        for _ in range(num_layers):
            layers.append(nn.Linear(dim, hidden_dim))
            layers.append(nn.ELU())
            dim = hidden_dim

        layers.append(nn.Linear(dim, 1))

        self.net = nn.Sequential(*layers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)
