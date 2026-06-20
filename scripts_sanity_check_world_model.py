import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, Tuple
import math

from dreamer_carracing.world_model import RewardModel, HaWorldModelAdapter


@dataclass
class MDNRNNConfig:
    input_size: int = 35
    output_size: int = 32
    hidden_size: int = 256
    num_layers: int = 1
    num_mixtures: int = 5
    dropout: float = 0.0


class MDNRNN(nn.Module):
    def __init__(self, config: MDNRNNConfig):
        super().__init__()
        self.config = config
        self.rnn = nn.LSTM(
            input_size=config.input_size,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            batch_first=True,
            dropout=config.dropout if config.num_layers > 1 else 0.0,
        )
        self.output = nn.Linear(
            config.hidden_size,
            config.output_size * config.num_mixtures * 3,
        )

    def forward(
        self,
        x: torch.Tensor,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ):
        rnn_out, hidden = self.rnn(x, hidden)
        raw = self.output(rnn_out)

        B, T, _ = raw.shape
        raw = raw.view(
            B,
            T,
            self.config.output_size,
            self.config.num_mixtures,
            3,
        )

        log_pi = raw[..., 0]
        mu = raw[..., 1]
        log_std = raw[..., 2]

        log_pi = F.log_softmax(log_pi, dim=-1)
        log_std = torch.clamp(log_std, min=-7.0, max=5.0)

        return log_pi, mu, log_std, hidden


class DummyVAE(nn.Module):
    def encode(self, obs):
        B = obs.shape[0]
        mu = torch.zeros(B, 32, device=obs.device)
        logvar = torch.zeros(B, 32, device=obs.device)
        return mu, logvar


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    vae = DummyVAE().to(device)
    mdn_rnn = MDNRNN(MDNRNNConfig()).to(device)
    reward_model = RewardModel(feature_dim=32 + 256).to(device)

    wm = HaWorldModelAdapter(
        vae=vae,
        mdn_rnn=mdn_rnn,
        reward_model=reward_model,
        z_dim=32,
        h_dim=256,
        action_dim=3,
        num_layers=1,
    ).to(device)

    B = 4
    obs = torch.randn(B, 3, 64, 64, device=device)
    action = torch.randn(B, 3, device=device).tanh()

    state = wm.initial_state(B, device)
    state = wm.observe_step(state, action, obs)

    out = wm.imagine_step(state, action)

    print("state.z:", state.z.shape)
    print("state.h:", state.h.shape)
    print("state.c:", state.c.shape)
    print("next_state.z:", out.next_state.z.shape)
    print("reward:", out.reward.shape)
    print("discount:", out.discount.shape)

    assert out.next_state.z.shape == (B, 32)
    assert out.reward.shape == (B, 1)
    assert out.discount.shape == (B, 1)

    print("Sanity check passed.")


if __name__ == "__main__":
    main()
