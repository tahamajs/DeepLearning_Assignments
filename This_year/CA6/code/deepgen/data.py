# deepgen/data.py
from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


def mnist_transforms(scale_to_minus1_1: bool = True) -> transforms.Compose:
    t = [transforms.ToTensor()]
    if scale_to_minus1_1:
        # MNIST ToTensor gives [0,1]. Convert to [-1,1].
        t.append(transforms.Normalize(mean=(0.5,), std=(0.5,)))
    return transforms.Compose(t)


def get_mnist_loaders(
    root: str = "./data",
    batch_size: int = 128,
    num_workers: int = 2,
    train_subset: Optional[int] = None,
    scale_to_minus1_1: bool = True,
    pin_memory: bool = True,
) -> Tuple[DataLoader, DataLoader]:
    tfm = mnist_transforms(scale_to_minus1_1=scale_to_minus1_1)

    train_ds = datasets.MNIST(root=root, train=True, download=True, transform=tfm)
    test_ds = datasets.MNIST(root=root, train=False, download=True, transform=tfm)

    if train_subset is not None:
        train_subset = int(train_subset)
        if train_subset <= 0:
            raise ValueError("train_subset must be > 0 when provided.")
        train_ds = Subset(train_ds, list(range(train_subset)))

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory and torch.cuda.is_available(),
        drop_last=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory and torch.cuda.is_available(),
        drop_last=False,
    )
    return train_loader, test_loader
