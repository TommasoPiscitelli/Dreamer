from dataclasses import dataclass
from typing import Optional, Protocol

import torch


@dataclass
class LatentState:
    """
    Latent state used by the Dreamer-style API.

    z: current VAE latent vector
    h: LSTM hidden state
    c: LSTM cell state
    """

    z: torch.Tensor
    h: torch.Tensor
    c: torch.Tensor
    extra: Optional[dict] = None

    @property
    def deter(self) -> torch.Tensor:
        """
        Deterministic recurrent feature used by actor/value/reward models.
        Returns shape [B, h_dim].
        """
        h = self.h

        if h.ndim == 3:
            # Caso standard LSTM: [num_layers, B, h_dim]
            if h.shape[0] == 1:
                h = h[0]

            # Caso alternativo: [B, 1, h_dim]
            elif h.shape[1] == 1:
                h = h[:, 0, :]

            # Fallback: prendi ultimo layer
            else:
                h = h[-1]

        return h

    @property
    def features(self) -> torch.Tensor:
        """
        Full latent feature vector: concat(z_t, h_t).
        Shape: [B, z_dim + h_dim].
        """
        z = self.z

        if z.ndim == 3:
            if z.shape[0] == 1:
                z = z[0]
            elif z.shape[1] == 1:
                z = z[:, 0, :]
            else:
                z = z[-1]

        return torch.cat([z, self.deter], dim=-1)

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
