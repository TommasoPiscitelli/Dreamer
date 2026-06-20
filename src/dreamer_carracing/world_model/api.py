from dataclasses import dataclass
from typing import Optional, Protocol

import torch


@dataclass
class LatentState:
    """
    Latent state used by the Dreamer-style API.

    z: current VAE latent vector, shape [B, z_dim]
    h: LSTM hidden state, shape [num_layers, B, h_dim]
    c: LSTM cell state, shape [num_layers, B, h_dim]
    """

    z: torch.Tensor
    h: torch.Tensor
    c: torch.Tensor
    extra: Optional[dict] = None

    @property
    def deter(self) -> torch.Tensor:
        """
        Deterministic recurrent feature used by actor/value/reward models.
        Returns the last LSTM layer hidden state: [B, h_dim].
        """
        return self.h[-1]

    @property
    def features(self) -> torch.Tensor:
        """
        Full latent feature vector: concat(z_t, h_t).
        Shape: [B, z_dim + h_dim].
        """
        return torch.cat([self.z, self.deter], dim=-1)

    def detach(self) -> "LatentState":
        return LatentState(
            z=self.z.detach(),
            h=self.h.detach(),
            c=self.c.detach(),
            extra=self.extra,
        )


@dataclass
class ImagineOutput:
    next_state: LatentState
    reward: torch.Tensor      # [B, 1]
    discount: torch.Tensor    # [B, 1]


class WorldModelBackend(Protocol):
    z_dim: int
    h_dim: int
    action_dim: int
    feature_dim: int

    def encode_obs(self, obs: torch.Tensor) -> torch.Tensor:
        """
        obs: [B, C, H, W]
        returns z: [B, z_dim]
        """
        ...

    def initial_state(self, batch_size: int, device: torch.device) -> LatentState:
        ...

    def observe_step(
        self,
        prev_state: LatentState,
        action: torch.Tensor,
        next_obs: torch.Tensor,
    ) -> LatentState:
        """
        Teacher-forcing transition using the true next observation.
        Used to reconstruct latent states from replay data.
        """
        ...

    def imagine_step(
        self,
        state: LatentState,
        action: torch.Tensor,
    ) -> ImagineOutput:
        """
        Latent imagination transition.
        Used later by Dreamer actor/value training.
        """
        ...
