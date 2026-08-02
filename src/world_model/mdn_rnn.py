import math
from dataclasses import dataclass
from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class MDNRNNConfig:
    input_size: int = 35          # z_size + action_size
    output_size: int = 32         # z_size
    hidden_size: int = 256
    num_layers: int = 1
    num_mixtures: int = 5
    dropout: float = 0.0


class MDNRNN(nn.Module):
    """
    PyTorch equivalent of the original World Models MDN-RNN.

    Input:
        x: (B, T, input_size)

    Output:
        log_pi:  (B, T, output_size, num_mixtures)
        mu:      (B, T, output_size, num_mixtures)
        log_std: (B, T, output_size, num_mixtures)
    """

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
        """
        x: (B, T, input_size)
        hidden:
            h: (num_layers, B, hidden_size)
            c: (num_layers, B, hidden_size)
        """
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

        # Optional numerical stability clamp.
        log_std = torch.clamp(log_std, min=-7.0, max=5.0)

        return log_pi, mu, log_std, hidden

    def init_hidden(self, batch_size: int, device: torch.device):
        h = torch.zeros(
            self.config.num_layers,
            batch_size,
            self.config.hidden_size,
            device=device,
        )
        c = torch.zeros(
            self.config.num_layers,
            batch_size,
            self.config.hidden_size,
            device=device,
        )
        return h, c


def mdn_loss(
    log_pi: torch.Tensor,
    mu: torch.Tensor,
    log_std: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """
    Negative log likelihood under a diagonal mixture of Gaussians.

    log_pi:  (B, T, z_size, K)
    mu:      (B, T, z_size, K)
    log_std: (B, T, z_size, K)
    target:  (B, T, z_size)
    """

    target = target.unsqueeze(-1)  # (B, T, z_size, 1)

    log_prob = -0.5 * ((target - mu) / torch.exp(log_std)) ** 2
    log_prob = log_prob - log_std - 0.5 * math.log(2.0 * math.pi)

    log_prob = log_pi + log_prob

    log_prob = torch.logsumexp(log_prob, dim=-1)  # (B, T, z_size)

    return -log_prob.mean()


@torch.no_grad()
def sample_from_mdn(
    log_pi: torch.Tensor,
    mu: torch.Tensor,
    log_std: torch.Tensor,
    temperature: float = 1.0,
) -> torch.Tensor:
    """
    Sample z_{t+1} from MDN output.

    Inputs can be:
        log_pi:  (..., z_size, K)
        mu:      (..., z_size, K)
        log_std: (..., z_size, K)

    Returns:
        sample: (..., z_size)
    """

    if temperature <= 0:
        raise ValueError("temperature must be > 0")

    pi_logits = log_pi / temperature
    mixture_dist = torch.distributions.Categorical(logits=pi_logits)
    mixture_idx = mixture_dist.sample()  # (..., z_size)

    gather_idx = mixture_idx.unsqueeze(-1)

    chosen_mu = torch.gather(mu, dim=-1, index=gather_idx).squeeze(-1)
    chosen_log_std = torch.gather(log_std, dim=-1, index=gather_idx).squeeze(-1)

    eps = torch.randn_like(chosen_mu)
    sample = chosen_mu + torch.exp(chosen_log_std) * eps * math.sqrt(temperature)

    return sample