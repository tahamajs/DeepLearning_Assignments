from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class VAEOutput:
    elbo: torch.Tensor
    recon_loss: torch.Tensor
    kl: torch.Tensor
    mu: torch.Tensor
    logvar: torch.Tensor
    z: torch.Tensor
    x_logits: torch.Tensor


class Encoder(nn.Module):
    def __init__(self, x_dim: int = 784, h_dim: int = 400, z_dim: int = 16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(x_dim, h_dim),
            nn.ReLU(inplace=True),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(inplace=True),
        )
        self.mu = nn.Linear(h_dim, z_dim)
        self.logvar = nn.Linear(h_dim, z_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.net(x)
        return self.mu(h), self.logvar(h)


class Decoder(nn.Module):
    def __init__(self, z_dim: int = 16, h_dim: int = 400, x_dim: int = 784):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim, h_dim),
            nn.ReLU(inplace=True),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(inplace=True),
            nn.Linear(h_dim, x_dim),  # logits for Bernoulli
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class VampPriorVAE(nn.Module):
    """MLP VAE with optional VampPrior (mixture of variational posteriors over pseudo-inputs)."""

    def __init__(
        self,
        z_dim: int = 16,
        x_dim: int = 784,
        h_dim: int = 400,
        vamp_k: int = 0,
        beta: float = 1.0,
    ):
        super().__init__()
        self.z_dim = z_dim
        self.x_dim = x_dim
        self.beta = beta
        self.vamp_k = int(vamp_k)

        self.encoder = Encoder(x_dim=x_dim, h_dim=h_dim, z_dim=z_dim)
        self.decoder = Decoder(z_dim=z_dim, h_dim=h_dim, x_dim=x_dim)

        if self.vamp_k > 0:
            self.pseudo_logits = nn.Parameter(torch.randn(self.vamp_k, x_dim))  # sigmoid -> (0,1)

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def _log_normal_diag(self, z: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        const = torch.log(torch.tensor(2.0 * torch.pi, device=z.device)) * z.size(-1)
        return -0.5 * (const + logvar.sum(dim=-1) + ((z - mu) ** 2 / torch.exp(logvar)).sum(dim=-1))

    def log_q_z_x(self, z: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        return self._log_normal_diag(z, mu, logvar)

    def log_p_z(self, z: torch.Tensor) -> torch.Tensor:
        if self.vamp_k <= 0:
            mu = torch.zeros_like(z)
            logvar = torch.zeros_like(z)
            return self._log_normal_diag(z, mu, logvar)

        u = torch.sigmoid(self.pseudo_logits)  # (K, x_dim) in (0,1)
        mu_k, logvar_k = self.encoder(u)       # (K, z_dim)

        z_bk = z.unsqueeze(1)                  # (B,1,z)
        mu_bk = mu_k.unsqueeze(0)              # (1,K,z)
        logvar_bk = logvar_k.unsqueeze(0)      # (1,K,z)

        log_comp = -0.5 * (
            torch.log(torch.tensor(2.0 * torch.pi, device=z.device)) * self.z_dim
            + logvar_bk.sum(dim=-1)
            + ((z_bk - mu_bk) ** 2 / torch.exp(logvar_bk)).sum(dim=-1)
        )  # (B,K)

        log_p = torch.logsumexp(
            log_comp - torch.log(torch.tensor(float(self.vamp_k), device=z.device)),
            dim=1
        )  # (B,)
        return log_p

    def forward(self, x: torch.Tensor) -> VAEOutput:
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        x_logits = self.decoder(z)

        recon_loss = F.binary_cross_entropy_with_logits(x_logits, x, reduction="sum")
        log_q = self.log_q_z_x(z, mu, logvar).sum()
        log_p = self.log_p_z(z).sum()
        kl = log_q - log_p

        elbo = -recon_loss - self.beta * kl
        return VAEOutput(
            elbo=elbo,
            recon_loss=recon_loss,
            kl=kl,
            mu=mu,
            logvar=logvar,
            z=z,
            x_logits=x_logits,
        )

    @torch.no_grad()
    def sample(self, n: int, device: torch.device) -> torch.Tensor:
        if self.vamp_k <= 0:
            z = torch.randn(n, self.z_dim, device=device)
        else:
            u = torch.sigmoid(self.pseudo_logits).to(device)
            mu_k, logvar_k = self.encoder(u)
            k = torch.randint(0, self.vamp_k, (n,), device=device)
            mu = mu_k[k]
            logvar = logvar_k[k]
            z = self.reparameterize(mu, logvar)

        x_logits = self.decoder(z)
        x = torch.sigmoid(x_logits)
        return x.view(n, 1, 28, 28)
