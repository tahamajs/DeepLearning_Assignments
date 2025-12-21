"""Train script skeleton for S&P baseline MLP (no execution in this commit)."""
import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from src.models.sp_models import MLPRegressor


class SeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def train(args):
    # data loading and splitting is handled in notebook; this script expects prepared numpy arrays
    import numpy as np
    X_train = np.load(args.x_train)
    y_train = np.load(args.y_train)
    X_val = np.load(args.x_val)
    y_val = np.load(args.y_val)

    train_ds = SeriesDataset(X_train, y_train)
    val_ds = SeriesDataset(X_val, y_val)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MLPRegressor(X_train.shape[1], hidden_dim=args.hidden_dim, dropout=args.dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.MSELoss()

    best_val = float('inf')
    for epoch in range(args.epochs):
        model.train()
        for Xb, yb in train_loader:
            Xb, yb = Xb.to(device), yb.to(device)
            opt.zero_grad()
            pred = model(Xb)
            loss = criterion(pred, yb)
            loss.backward()
            opt.step()
        # validation step omitted for brevity


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--x_train')
    parser.add_argument('--y_train')
    parser.add_argument('--x_val')
    parser.add_argument('--y_val')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--dropout', type=float, default=0.2)
    args = parser.parse_args()
    train(args)
