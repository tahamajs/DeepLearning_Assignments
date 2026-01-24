# deepgen/models/vae.py
from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import ResidualBlock


class Encoder(nn.Module):
    def __init__(self, z_dim: int = 32):
        super().__init__()
        self.stem = nn.Conv2d(1, 32, 3, padding=1)
        self.rb1 = ResidualBlock(32, 64, downsample=True, use_se=True)   # 28 -> 14
        self.rb2 = ResidualBlock(64, 128, downsample=True, use_se=True)  # 14 -> 7
        self.rb3 = ResidualBlock(128, 128, downsample=False, use_se=True)
        self.fc = nn.Linear(128 * 7 * 7, 256)
        self.mu = nn.Linear(256, z_dim)
        self.logvar = nn.Linear(256, z_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = F.relu(self.stem(x), inplace=True)
        h = self.rb1(h)
        h = self.rb2(h)
        h = self.rb3(h)
        h = h.view(h.size(0), -1)
        h = F.relu(self.fc(h), inplace=True)
        return self.mu(h), self.logvar(h)


class Decoder(nn.Module):
    def __init__(self, z_dim: int = 32):
        super().__init__()
        self.fc1 = nn.Linear(z_dim, 256)
        self.fc2 = nn.Linear(256, 128 * 7 * 7)
        self.rb1 = ResidualBlock(128, 128, downsample=False, use_se=True)
        self.up1 = nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1)   # 7 -> 14
        self.up2 = nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1)    # 14 -> 28
        self.out = nn.Conv2d(32, 1, 3, padding=1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.fc1(z), inplace=True)
        h = F.relu(self.fc2(h), inplace=True)
        h = h.view(z.size(0), 128, 7, 7)
        h = self.rb1(h)
        h = F.relu(self.up1(h), inplace=True)
        h = F.relu(self.up2(h), inplace=True)
        x = torch.sigmoid(self.out(h))  # [0,1]
        return x


class VAEVampPrior(nn.Module):
    """
    VAE with VampPrior: p(z) = (1/K) sum_k q(z | u_k), where u_k are learnable pseudo-inputs.
    """
    def __init__(self, z_dim: int = 32, n_pseudos: int = 500):
        super().__init__()
        self.z_dim = z_dim
        self.n_pseudos = n_pseudos

        self.encoder = Encoder(z_dim=z_dim)
        self.decoder = Decoder(z_dim=z_dim)

        # Pseudo-inputs are images in [-1,1] space to match training x
        self.pseudos = nn.Parameter(torch.randn(n_pseudos, 1, 28, 28) * 0.05)

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.encoder(x)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_rec = self.decode(z)
        return x_rec, mu, logvar

    @torch.no_grad()
    def sample(self, n: int, device: torch.device) -> torch.Tensor:
        # Sample from VampPrior by selecting a pseudo input u_k and sampling from q(z|u_k)
        idx = torch.randint(0, self.n_pseudos, (n,), device=device)
        u = torch.tanh(self.pseudos[idx].to(device))  # enforce roughly [-1,1]
        mu, logvar = self.encode(u)
        z = self.reparameterize(mu, logvar)
        x = self.decode(z)  # [0,1]
        return x

    def _log_normal(self, z: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        # log N(z; mu, diag(exp(logvar))) per sample
        return -0.5 * (logvar + (z - mu) ** 2 / torch.exp(logvar) + torch.log(torch.tensor(2.0 * 3.1415926535, device=z.device))).sum(dim=-1)

    def log_pz_vampprior(self, z: torch.Tensor) -> torch.Tensor:
        # log p(z) = log mean_k q(z|u_k)
        # Compute mu_k, logvar_k for all pseudos (chunked if needed)
        u = torch.tanh(self.pseudos)  # (K,1,28,28)
        mu_k, logvar_k = self.encode(u)  # (K, z)
        # z: (B, z) -> (B, K, z)
        z_bk = z.unsqueeze(1)  # (B,1,z)
        mu = mu_k.unsqueeze(0)  # (1,K,z)
        lv = logvar_k.unsqueeze(0)  # (1,K,z)
        log_q = self._log_normal(z_bk.expand(-1, self.n_pseudos, -1), mu, lv)  # (B,K)
        log_p = torch.logsumexp(log_q - torch.log(torch.tensor(float(self.n_pseudos), device=z.device)), dim=1)
        return log_p

    def elbo_loss(self, x: torch.Tensor, x_rec: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor, beta: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Reconstruction term: BCE on [0,1] vs decoded x_rec
        # x is in [-1,1], convert to [0,1] target
        x01 = (x + 1.0) / 2.0
        recon = F.binary_cross_entropy(x_rec, x01, reduction="none").flatten(1).sum(dim=1)  # (B,)
        z = self.reparameterize(mu, logvar)
        log_qzx = self._log_normal(z, mu, logvar)
        log_pz = self.log_pz_vampprior(z)
        kl = (log_qzx - log_pz)  # (B,)
        loss = (recon + beta * kl).mean()
        return loss, recon.mean(), kl.mean()
