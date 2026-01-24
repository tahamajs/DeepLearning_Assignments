from __future__ import annotations

import os
import random
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torchvision.utils import make_grid, save_image


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@torch.no_grad()
def save_image_grid(
    images: torch.Tensor,
    path: str,
    nrow: int = 10,
    normalize: bool = True,
    value_range: Optional[Tuple[float, float]] = None,
) -> None:
    """Save a grid of images (B,C,H,W)."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    grid = make_grid(images, nrow=nrow, normalize=normalize, value_range=value_range)
    save_image(grid, path)


def gradient_penalty(
    discriminator,
    real: torch.Tensor,
    fake: torch.Tensor,
    labels: Optional[torch.Tensor] = None,
    lambda_gp: float = 10.0,
) -> torch.Tensor:
    """WGAN-GP gradient penalty for conditional discriminators."""
    device = real.device
    bsz = real.size(0)
    eps = torch.rand(bsz, 1, 1, 1, device=device)
    x_hat = eps * real + (1 - eps) * fake
    x_hat.requires_grad_(True)

    if labels is None:
        d_hat = discriminator(x_hat)
    else:
        d_hat = discriminator(x_hat, labels)

    if d_hat.dim() == 1:
        d_hat = d_hat.view(-1, 1)

    grads = torch.autograd.grad(
        outputs=d_hat,
        inputs=x_hat,
        grad_outputs=torch.ones_like(d_hat),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    grads = grads.view(bsz, -1)
    gp = ((grads.norm(2, dim=1) - 1.0) ** 2).mean()
    return lambda_gp * gp


@dataclass
class AverageMeter:
    name: str
    fmt: str = ".4f"
    val: float = 0.0
    avg: float = 0.0
    sum: float = 0.0
    count: int = 0

    def reset(self) -> None:
        self.val = self.avg = self.sum = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1) -> None:
        self.val = float(val)
        self.sum += float(val) * n
        self.count += n
        self.avg = self.sum / max(1, self.count)

    def __str__(self) -> str:
        return f"{self.name}: {self.avg:{self.fmt}}"


def bce_with_logits_sum(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Sum BCE over pixels and batch."""
    return F.binary_cross_entropy_with_logits(logits, targets, reduction="sum")
