from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn

from dreamer_carracing.world_model.api import LatentState, ImagineOutput
from dreamer_carracing.world_model.reward_model import RewardModel


class HaWorldModelAdapter(nn.Module):
    """
    Adapter for a Ha & Schmidhuber-style World Model:

        obs_t --VAE--> z_t
        concat(z_t, a_t) --MDN-RNN/LSTM--> distribution over z_{t+1}
        concat(z_{t+1}, h_{t+1}) --RewardModel--> r_t

    This exposes a Dreamer-style API:
        observe_step(...)
        imagine_step(...)
    """

    def __init__(
        self,
        vae: nn.Module,
        mdn_rnn: nn.Module,
        reward_model: RewardModel,
        z_dim: int = 32,
        h_dim: int = 256,
        action_dim: int = 3,
        num_layers: int = 1,
        discount: float = 0.99,
        freeze_vae: bool = True,
        freeze_mdn_rnn: bool = True,
    ):
        super().__init__()

        self.vae = vae
        self.mdn_rnn = mdn_rnn
        self.reward_model = reward_model

        self.z_dim = z_dim
        self.h_dim = h_dim
        self.action_dim = action_dim
        self.num_layers = num_layers
        self.discount = discount
        self.feature_dim = z_dim + h_dim

        if freeze_vae:
            for p in self.vae.parameters():
                p.requires_grad_(False)

        if freeze_mdn_rnn:
            for p in self.mdn_rnn.parameters():
                p.requires_grad_(False)

    @torch.no_grad()
    def encode_obs(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Encode observations with the frozen VAE.

        Expected obs shape:
            [B, C, H, W]

        This assumes your VAE exposes:
            mu, logvar = vae.encode(obs)

        If your ConvVAE has a slightly different API, this is the only
        method we should need to edit.
        """
        encoded = self.vae.encode(obs)

        if isinstance(encoded, tuple):
            mu = encoded[0]
        else:
            mu = encoded

        return mu

    def initial_state(self, batch_size: int, device: torch.device) -> LatentState:
        z = torch.zeros(batch_size, self.z_dim, device=device)
        h = torch.zeros(self.num_layers, batch_size, self.h_dim, device=device)
        c = torch.zeros(self.num_layers, batch_size, self.h_dim, device=device)

        return LatentState(z=z, h=h, c=c)

    @torch.no_grad()
    def observe_step(
        self,
        prev_state: LatentState,
        action: torch.Tensor,
        next_obs: torch.Tensor,
    ) -> LatentState:
        """
        Teacher-forcing step.

        We update the LSTM hidden state using the previous latent z_t
        and action a_t, but we set z_{t+1} from the true next observation.

        This is useful for training the reward model on replay data.
        """
        _, _, _, next_h, next_c = self._mdn_step(
            z=prev_state.z,
            action=action,
            h=prev_state.h,
            c=prev_state.c,
        )

        next_z = self.encode_obs(next_obs)

        return LatentState(z=next_z, h=next_h, c=next_c)

    def imagine_step(
        self,
        state: LatentState,
        action: torch.Tensor,
    ) -> ImagineOutput:
        """
        Differentiable latent imagination step.

        Important:
        - We do NOT call sample_from_mdn here.
        - We use the mixture mean, sum_k pi_k * mu_k, so gradients can flow
          through the predicted latent dynamics into the action.
        """
        log_pi, mu, _, next_h, next_c = self._mdn_step(
            z=state.z,
            action=action,
            h=state.h,
            c=state.c,
        )

        next_z = self._mixture_mean(log_pi=log_pi, mu=mu)

        next_state = LatentState(z=next_z, h=next_h, c=next_c)

        reward = self.reward_model(next_state.features)

        discount = torch.ones_like(reward) * self.discount

        return ImagineOutput(
            next_state=next_state,
            reward=reward,
            discount=discount,
        )

    def predict_reward(self, state: LatentState) -> torch.Tensor:
        return self.reward_model(state.features)

    def _mdn_step(
        self,
        z: torch.Tensor,
        action: torch.Tensor,
        h: torch.Tensor,
        c: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        One recurrent MDN-RNN step.

        z:      [B, z_dim]
        action: [B, action_dim]
        h:      [num_layers, B, h_dim]
        c:      [num_layers, B, h_dim]

        Returns:
            log_pi:  [B, z_dim, K]
            mu:      [B, z_dim, K]
            log_std: [B, z_dim, K]
            next_h:  [num_layers, B, h_dim]
            next_c:  [num_layers, B, h_dim]
        """
        x = torch.cat([z, action], dim=-1)

        expected = self.z_dim + self.action_dim
        if x.shape[-1] != expected:
            raise ValueError(
                f"MDN-RNN input has wrong size: got {x.shape[-1]}, expected {expected}."
            )

        x = x.unsqueeze(1)  # [B, 1, z_dim + action_dim]

        log_pi, mu, log_std, hidden = self.mdn_rnn(x, hidden=(h, c))
        next_h, next_c = hidden

        # Remove the time dimension T=1.
        log_pi = log_pi[:, 0]      # [B, z_dim, K]
        mu = mu[:, 0]              # [B, z_dim, K]
        log_std = log_std[:, 0]    # [B, z_dim, K]

        return log_pi, mu, log_std, next_h, next_c

    @staticmethod
    def _mixture_mean(log_pi: torch.Tensor, mu: torch.Tensor) -> torch.Tensor:
        """
        Differentiable expected value of the MDN distribution.

        log_pi: [B, z_dim, K]
        mu:     [B, z_dim, K]

        returns:
            next_z: [B, z_dim]
        """
        pi = torch.exp(log_pi)
        return torch.sum(pi * mu, dim=-1)
