from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms


class DynamicBinarizeMNIST(Dataset):
    """MNIST with dynamic binarization: each __getitem__ samples Bernoulli(x)."""

    def __init__(
        self,
        root: str,
        train: bool,
        download: bool = True,
        dequantize: bool = False,
        flatten: bool = False,
    ):
        self.flatten = flatten
        self.dequantize = dequantize
        base_tf = transforms.ToTensor()
        self.mnist = datasets.MNIST(root=root, train=train, download=download, transform=base_tf)

    def __len__(self) -> int:
        return len(self.mnist)

    def __getitem__(self, idx: int):
        x, y = self.mnist[idx]  # x in [0,1], (1,28,28)
        if self.dequantize:
            x = (x * 255.0 + torch.rand_like(x)) / 256.0
            x = x.clamp(0.0, 1.0)

        x = torch.bernoulli(x)

        if self.flatten:
            x = x.view(-1)
        return x, y


@dataclass
class DataConfig:
    root: str = "./data"
    batch_size: int = 128
    num_workers: int = 2
    pin_memory: bool = True


def _subset_indices_by_class(targets: torch.Tensor, per_class: int, num_classes: int = 10, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    idxs = []
    targets = targets.clone()
    for c in range(num_classes):
        class_idxs = (targets == c).nonzero(as_tuple=False).view(-1)
        perm = class_idxs[torch.randperm(class_idxs.numel(), generator=g)]
        idxs.append(perm[:per_class])
    return torch.cat(idxs, dim=0).tolist()


def get_dynamic_mnist_loaders(
    cfg: DataConfig,
    train_subset: Optional[int] = None,
    subset_seed: int = 0,
    dequantize: bool = False,
    flatten: bool = False,
) -> Tuple[DataLoader, DataLoader]:
    train_ds = DynamicBinarizeMNIST(cfg.root, train=True, download=True, dequantize=dequantize, flatten=flatten)
    test_ds = DynamicBinarizeMNIST(cfg.root, train=False, download=True, dequantize=dequantize, flatten=flatten)

    if train_subset is not None:
        targets = torch.tensor(train_ds.mnist.targets, dtype=torch.long)
        per_class = max(1, train_subset // 10)
        idxs = _subset_indices_by_class(targets, per_class=per_class, seed=subset_seed)
        train_ds = Subset(train_ds, idxs)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        drop_last=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        drop_last=False,
    )
    return train_loader, test_loader


def get_mnist_loaders_for_gan(
    cfg: DataConfig,
    train_subset: Optional[int] = None,
    subset_seed: int = 0,
) -> Tuple[DataLoader, DataLoader]:
    """MNIST loaders for GAN/classifier: images scaled to [-1, 1]."""
    tf = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: x * 2.0 - 1.0)])
    train_ds = datasets.MNIST(cfg.root, train=True, download=True, transform=tf)
    test_ds = datasets.MNIST(cfg.root, train=False, download=True, transform=tf)

    if train_subset is not None:
        targets = torch.tensor(train_ds.targets, dtype=torch.long)
        per_class = max(1, train_subset // 10)
        idxs = _subset_indices_by_class(targets, per_class=per_class, seed=subset_seed)
        train_ds = Subset(train_ds, idxs)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        drop_last=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        drop_last=False,
    )
    return train_loader, test_loader
