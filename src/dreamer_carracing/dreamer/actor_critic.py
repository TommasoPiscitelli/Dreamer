import torch
import torch.nn as nn
from torch.distributions import Normal


class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 3,
        activation=nn.ELU,
    ):
        super().__init__()

        layers = []
        dim = input_dim

        for _ in range(num_layers):
            layers.append(nn.Linear(dim, hidden_dim))
            layers.append(activation())
            dim = hidden_dim

        layers.append(nn.Linear(dim, output_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Actor(nn.Module):
    """
    Actor for CarRacing in latent feature space.

    Input:
        features = concat(z, h), shape [B, feature_dim]

    Output action:
        steering in [-1, 1]
        gas      in [0, 1]
        brake    in [0, 1]

    We sample in an unconstrained Gaussian space and then squash with tanh.
    """

    def __init__(
        self,
        feature_dim: int,
        action_dim: int = 3,
        hidden_dim: int = 256,
        num_layers: int = 3,
        min_std: float = 0.05,
        max_std: float = 1.0,
    ):
        super().__init__()

        self.feature_dim = feature_dim
        self.action_dim = action_dim
        self.min_std = min_std
        self.max_std = max_std

        self.backbone = MLP(
            input_dim=feature_dim,
            output_dim=hidden_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
        )

        self.mean_head = nn.Linear(hidden_dim, action_dim)
        self.log_std_head = nn.Linear(hidden_dim, action_dim)

    def forward(self, features: torch.Tensor):
        x = self.backbone(features)

        mean = self.mean_head(x)
        log_std = self.log_std_head(x)

        std = torch.sigmoid(log_std)
        std = self.min_std + (self.max_std - self.min_std) * std

        return Normal(mean, std)

    def sample(
        self,
        features: torch.Tensor,
        deterministic: bool = False,
    ):
        dist = self.forward(features)

        if deterministic:
            raw_action = dist.mean
        else:
            raw_action = dist.rsample()

        squashed = torch.tanh(raw_action)

        steering = squashed[..., 0:1]
        gas = 0.5 * (squashed[..., 1:2] + 1.0)
        brake = 0.5 * (squashed[..., 2:3] + 1.0)

        action = torch.cat([steering, gas, brake], dim=-1)

        entropy = dist.entropy().sum(dim=-1, keepdim=True)

        return action, entropy

    def mean_action(self, features: torch.Tensor) -> torch.Tensor:
        action, _ = self.sample(features, deterministic=True)
        return action


class Value(nn.Module):
    """
    Value model in latent feature space.

    Input:
        features = concat(z, h), shape [B, feature_dim]

    Output:
        V(features), shape [B, 1]
    """

    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 3,
    ):
        super().__init__()

        self.net = MLP(
            input_dim=feature_dim,
            output_dim=1,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)