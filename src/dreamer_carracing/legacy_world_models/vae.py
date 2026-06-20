from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvVAE(nn.Module):
    def __init__(
        self,
        z_size: int = 32,
        kl_tolerance: float = 0.5,
    ) -> None:
        super().__init__()
        self.z_size = z_size
        self.kl_tolerance = kl_tolerance

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Flatten(),
        )

        self.fc_mu = nn.Linear(2 * 2 * 256, z_size)
        self.fc_logvar = nn.Linear(2 * 2 * 256, z_size)

        self.fc_dec = nn.Linear(z_size, 4 * 256)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(1024, 128, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=6, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, kernel_size=6, stride=2),
            nn.Sigmoid(),
        )

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        sigma = torch.exp(logvar / 2.0)
        eps = torch.randn_like(sigma)
        return mu + sigma * eps

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        h = self.fc_dec(z)
        h = h.view(-1, 1024, 1, 1)
        return self.decoder(h)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar

    def loss(
        self,
        recon: torch.Tensor,
        x: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        recon_loss = F.mse_loss(recon, x, reduction="none")
        recon_loss = recon_loss.sum(dim=(1, 2, 3)).mean()

        kl_loss = -0.5 * torch.sum(
            1 + logvar - mu.pow(2) - logvar.exp(),
            dim=1,
        )

        kl_min = self.kl_tolerance * self.z_size
        kl_loss = torch.maximum(
            kl_loss,
            torch.full_like(kl_loss, kl_min),
        )
        kl_loss = kl_loss.mean()

        total_loss = recon_loss + kl_loss
        return total_loss, recon_loss, kl_loss