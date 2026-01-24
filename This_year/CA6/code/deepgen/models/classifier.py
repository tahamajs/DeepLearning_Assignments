# deepgen/models/classifier.py
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class LeNet5(nn.Module):
    """
    LeNet-style classifier for MNIST, with an option to return penultimate features.
    Input expected in [-1,1] range.
    """
    def __init__(self, n_classes: int = 10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)     # 28 -> 24
        self.conv2 = nn.Conv2d(6, 16, 5)    # 12 -> 8
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)       # penultimate
        self.fc3 = nn.Linear(84, n_classes)

    def forward(self, x: torch.Tensor, return_features: bool = False) -> torch.Tensor:
        x = F.relu(self.conv1(x), inplace=True)
        x = F.avg_pool2d(x, 2)  # 24 -> 12
        x = F.relu(self.conv2(x), inplace=True)
        x = F.avg_pool2d(x, 2)  # 8 -> 4
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x), inplace=True)
        feat = F.relu(self.fc2(x), inplace=True)  # (B,84)
        if return_features:
            return feat
        logits = self.fc3(feat)
        return logits
