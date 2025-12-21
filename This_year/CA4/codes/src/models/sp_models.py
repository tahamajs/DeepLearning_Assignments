"""Models for S&P500 forecasting: MLP baseline, Conv+LSTM, Conv+GRU, and a small TCN-like CNN-only model."""
from typing import Optional

import torch
import torch.nn as nn


class MLPRegressor(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x: torch.Tensor):
        # x: (B, input_dim)
        return self.net(x).squeeze(-1)


class ConvFeatureExtractor(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int = 64, kernel_size: int = 3):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv1d(in_channels, hidden_channels, kernel_size, padding=padding)
        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor):
        # x shape: (B, T, C) -> (B, C, T)
        x = x.transpose(1, 2)
        y = self.conv(x)
        y = self.act(y)
        return y.transpose(1, 2)  # back to (B, T, hidden)


class ConvLSTMRegressor(nn.Module):
    def __init__(self, in_channels: int, cnn_hidden: int = 64, lstm_hidden: int = 128, out_dim: int = 1):
        super().__init__()
        self.cnn = ConvFeatureExtractor(in_channels, cnn_hidden)
        self.lstm = nn.LSTM(cnn_hidden, lstm_hidden, batch_first=True)
        self.head = nn.Linear(lstm_hidden, out_dim)

    def forward(self, x: torch.Tensor):
        # x: (B, T, C)
        feat = self.cnn(x)
        out, (hn, cn) = self.lstm(feat)
        last = hn[-1]
        return self.head(last).squeeze(-1)


class ConvGRURegressor(nn.Module):
    def __init__(self, in_channels: int, cnn_hidden: int = 64, gru_hidden: int = 128, out_dim: int = 1):
        super().__init__()
        self.cnn = ConvFeatureExtractor(in_channels, cnn_hidden)
        self.gru = nn.GRU(cnn_hidden, gru_hidden, batch_first=True)
        self.head = nn.Linear(gru_hidden, out_dim)

    def forward(self, x: torch.Tensor):
        feat = self.cnn(x)
        out, hn = self.gru(feat)
        last = hn[-1]
        return self.head(last).squeeze(-1)


class DilatedConvBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 3, dilation: int = 1):
        super().__init__()
        padding = (kernel_size - 1) // 2 * dilation
        self.conv = nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=dilation)
        self.act = nn.ReLU()
        self.bn = nn.BatchNorm1d(channels)

    def forward(self, x: torch.Tensor):
        return self.bn(self.act(self.conv(x)))


class TCNRegressor(nn.Module):
    def __init__(self, in_channels: int, channels: int = 64, kernel_size: int = 3, num_blocks: int = 4):
        super().__init__()
        self.input_proj = nn.Conv1d(in_channels, channels, 1)
        self.blocks = nn.ModuleList([DilatedConvBlock(channels, kernel_size, dilation=2 ** i) for i in range(num_blocks)])
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(channels, 1)

    def forward(self, x: torch.Tensor):
        # x: (B, T, C) -> (B, C, T)
        x = x.transpose(1, 2)
        x = self.input_proj(x)
        for blk in self.blocks:
            x = x + blk(x)
        x = self.pool(x).squeeze(-1)
        return self.head(x).squeeze(-1)
