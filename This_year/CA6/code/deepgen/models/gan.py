from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm

from .attention import ChannelAttention, SelfAttention2d


class ConditionalBatchNorm2d(nn.Module):
    def __init__(self, num_features: int, num_classes: int):
        super().__init__()
        self.bn = nn.BatchNorm2d(num_features, affine=False)
        self.embed = nn.Embedding(num_classes, num_features * 2)
        nn.init.zeros_(self.embed.weight)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        h = self.bn(x)
        gamma_beta = self.embed(y)
        gamma, beta = gamma_beta.chunk(2, dim=1)
        gamma = gamma.view(-1, x.size(1), 1, 1) + 1.0
        beta = beta.view(-1, x.size(1), 1, 1)
        return gamma * h + beta


class GResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, num_classes: int, use_ca: bool = True):
        super().__init__()
        self.cbn1 = ConditionalBatchNorm2d(in_ch, num_classes)
        self.cbn2 = ConditionalBatchNorm2d(out_ch, num_classes)
        self.conv1 = spectral_norm(nn.Conv2d(in_ch, out_ch, 3, padding=1))
        self.conv2 = spectral_norm(nn.Conv2d(out_ch, out_ch, 3, padding=1))
        self.skip = spectral_norm(nn.Conv2d(in_ch, out_ch, 1)) if in_ch != out_ch else None
        self.ca = ChannelAttention(out_ch) if use_ca else nn.Identity()

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.cbn1(x, y), inplace=True)
        h = F.interpolate(h, scale_factor=2, mode="nearest")
        h = self.conv1(h)
        h = F.relu(self.cbn2(h, y), inplace=True)
        h = self.conv2(h)
        h = self.ca(h)

        x_up = F.interpolate(x, scale_factor=2, mode="nearest")
        if self.skip is not None:
            x_up = self.skip(x_up)
        return h + x_up


class ConditionalGenerator(nn.Module):
    """Conditional ResNet Generator for MNIST (28x28)."""

    def __init__(
        self,
        z_dim: int = 128,
        num_classes: int = 10,
        base_ch: int = 128,
        use_attention: bool = True,
        use_ca: bool = True,
    ):
        super().__init__()
        self.z_dim = z_dim
        self.embed = nn.Embedding(num_classes, z_dim)
        self.fc = spectral_norm(nn.Linear(z_dim, base_ch * 4 * 4))
        self.block1 = GResBlock(base_ch, base_ch, num_classes, use_ca=use_ca)        # 4->8
        self.attn = SelfAttention2d(base_ch) if use_attention else nn.Identity()
        self.block2 = GResBlock(base_ch, base_ch // 2, num_classes, use_ca=use_ca)   # 8->16
        self.block3 = GResBlock(base_ch // 2, base_ch // 4, num_classes, use_ca=use_ca)  # 16->32
        self.bn = nn.BatchNorm2d(base_ch // 4)
        self.conv_out = spectral_norm(nn.Conv2d(base_ch // 4, 1, 3, padding=1))

    def forward(self, z: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        z = z + self.embed(y)
        h = self.fc(z).view(z.size(0), -1, 4, 4)
        h = self.block1(h, y)
        h = self.attn(h)
        h = self.block2(h, y)
        h = self.block3(h, y)
        h = F.relu(self.bn(h), inplace=True)
        h = self.conv_out(h)
        x = torch.tanh(h)
        x = x[:, :, 2:30, 2:30]  # center crop 32->28
        return x


class DResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, down: bool = True, use_ca: bool = True):
        super().__init__()
        self.down = down
        self.conv1 = spectral_norm(nn.Conv2d(in_ch, out_ch, 3, padding=1))
        self.conv2 = spectral_norm(nn.Conv2d(out_ch, out_ch, 3, padding=1))
        self.skip = spectral_norm(nn.Conv2d(in_ch, out_ch, 1)) if in_ch != out_ch else None
        self.ca = ChannelAttention(out_ch) if use_ca else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.relu(x, inplace=True)
        h = self.conv1(h)
        h = F.relu(h, inplace=True)
        h = self.conv2(h)
        h = self.ca(h)
        if self.down:
            h = F.avg_pool2d(h, 2)

        x_skip = x
        if self.skip is not None:
            x_skip = self.skip(x_skip)
        if self.down:
            x_skip = F.avg_pool2d(x_skip, 2)
        return h + x_skip


class ProjectionDiscriminator(nn.Module):
    """Projection discriminator for class-conditional GANs."""

    def __init__(
        self,
        num_classes: int = 10,
        base_ch: int = 128,
        use_attention: bool = True,
        use_ca: bool = True,
    ):
        super().__init__()
        self.block1 = DResBlock(1, base_ch // 4, down=True, use_ca=use_ca)       # 28->14
        self.block2 = DResBlock(base_ch // 4, base_ch // 2, down=True, use_ca=use_ca)  # 14->7
        self.attn = SelfAttention2d(base_ch // 2) if use_attention else nn.Identity()
        self.block3 = DResBlock(base_ch // 2, base_ch, down=False, use_ca=use_ca)      # 7->7
        self.block4 = DResBlock(base_ch, base_ch, down=False, use_ca=use_ca)
        self.linear = spectral_norm(nn.Linear(base_ch, 1))
        self.embed = spectral_norm(nn.Embedding(num_classes, base_ch))

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        h = self.block1(x)
        h = self.block2(h)
        h = self.attn(h)
        h = self.block3(h)
        h = self.block4(h)
        h = F.relu(h, inplace=True)
        h = h.sum(dim=(2, 3))  # global sum pool

        out = self.linear(h).view(-1)
        proj = (self.embed(y) * h).sum(dim=1)
        return out + proj
