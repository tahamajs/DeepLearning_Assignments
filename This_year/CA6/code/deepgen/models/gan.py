# deepgen/models/gan.py
from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm

from .blocks import SelfAttention2d, SEBlock


class GenBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, upsample: bool = True, use_se: bool = True):
        super().__init__()
        self.upsample = upsample
        self.conv1 = spectral_norm(nn.Conv2d(in_ch, out_ch, 3, padding=1))
        self.conv2 = spectral_norm(nn.Conv2d(out_ch, out_ch, 3, padding=1))
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.se = SEBlock(out_ch) if use_se else nn.Identity()
        self.skip = spectral_norm(nn.Conv2d(in_ch, out_ch, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x
        if self.upsample:
            h = F.interpolate(h, scale_factor=2, mode="nearest")
            x = F.interpolate(x, scale_factor=2, mode="nearest")
        h = F.relu(self.bn1(self.conv1(h)), inplace=True)
        h = self.bn2(self.conv2(h))
        h = self.se(h)
        return F.relu(h + self.skip(x), inplace=True)


class Generator(nn.Module):
    def __init__(self, z_dim: int = 128, n_classes: int = 10, base_ch: int = 128):
        super().__init__()
        self.z_dim = z_dim
        self.n_classes = n_classes
        self.embed = nn.Embedding(n_classes, z_dim)

        self.fc = spectral_norm(nn.Linear(z_dim, base_ch * 7 * 7))
        self.block1 = GenBlock(base_ch, base_ch, upsample=True, use_se=True)   # 7 -> 14
        self.attn = SelfAttention2d(base_ch)
        self.block2 = GenBlock(base_ch, base_ch // 2, upsample=True, use_se=True)  # 14 -> 28
        self.out = spectral_norm(nn.Conv2d(base_ch // 2, 1, 3, padding=1))

    def forward(self, z: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # Conditional via label embedding added to z
        z = z + self.embed(y)
        h = self.fc(z).view(z.size(0), -1, 7, 7)
        h = self.block1(h)
        h = self.attn(h)
        h = self.block2(h)
        x = torch.tanh(self.out(h))  # [-1,1]
        return x


class DiscBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, downsample: bool = True, use_se: bool = False):
        super().__init__()
        stride = 2 if downsample else 1
        self.conv1 = spectral_norm(nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1))
        self.conv2 = spectral_norm(nn.Conv2d(out_ch, out_ch, 3, padding=1))
        self.skip = spectral_norm(nn.Conv2d(in_ch, out_ch, 1, stride=stride))
        self.se = SEBlock(out_ch) if use_se else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.leaky_relu(self.conv1(x), 0.2, inplace=True)
        h = F.leaky_relu(self.conv2(h), 0.2, inplace=True)
        h = self.se(h)
        return h + self.skip(x)


class ProjectionDiscriminator(nn.Module):
    """
    Projection discriminator:
      D(x, y) = h(x)^T v(y) + w^T h(x) + b
    where h(x) are pooled features.
    """
    def __init__(self, n_classes: int = 10, base_ch: int = 128):
        super().__init__()
        self.n_classes = n_classes
        self.block1 = DiscBlock(1, base_ch // 2, downsample=True)     # 28 -> 14
        self.attn = SelfAttention2d(base_ch // 2)
        self.block2 = DiscBlock(base_ch // 2, base_ch, downsample=True)  # 14 -> 7
        self.block3 = DiscBlock(base_ch, base_ch, downsample=False)
        self.embed = spectral_norm(nn.Embedding(n_classes, base_ch))
        self.fc = spectral_norm(nn.Linear(base_ch, 1))
        self.bias = nn.Parameter(torch.zeros(1))

    def features(self, x: torch.Tensor) -> torch.Tensor:
        h = self.block1(x)
        h = self.attn(h)
        h = self.block2(h)
        h = self.block3(h)
        h = F.leaky_relu(h, 0.2, inplace=True)
        h = h.sum(dim=(2, 3))  # global sum pooling (B, C)
        return h

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        h = self.features(x)  # (B, C)
        out = self.fc(h) + self.bias  # (B,1)
        proj = (self.embed(y) * h).sum(dim=1, keepdim=True)  # (B,1)
        return out + proj


def gradient_penalty(
    D: nn.Module,
    real_x: torch.Tensor,
    fake_x: torch.Tensor,
    y: torch.Tensor,
) -> torch.Tensor:
    b = real_x.size(0)
    eps = torch.rand(b, 1, 1, 1, device=real_x.device)
    x_hat = eps * real_x + (1.0 - eps) * fake_x
    x_hat.requires_grad_(True)

    d_hat = D(x_hat, y)
    grad = torch.autograd.grad(
        outputs=d_hat,
        inputs=x_hat,
        grad_outputs=torch.ones_like(d_hat),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    grad = grad.view(b, -1)
    gp = ((grad.norm(2, dim=1) - 1.0) ** 2).mean()
    return gp
