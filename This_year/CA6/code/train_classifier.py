from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from deepgen.data import DataConfig, get_mnist_loaders_for_gan
from deepgen.models import ConditionalGenerator, LeNet5
from deepgen.utils import AverageMeter, set_seed


def parse_args():
    p = argparse.ArgumentParser(description="Train LeNet-5 on MNIST with optional GAN augmentation.")
    p.add_argument("--data_root", type=str, default="./data")
    p.add_argument("--outdir", type=str, default="./runs/classifier")
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--real_n", type=int, default=1000, help="real training samples (balanced).")
    p.add_argument("--gan_ckpt", type=str, default=None, help="path to GAN checkpoint containing key 'G'.")
    p.add_argument("--aug_n", type=int, default=0, help="generated samples to add (balanced).")
    p.add_argument("--z_dim", type=int, default=128, help="must match GAN z_dim if using augmentation.")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


@torch.no_grad()
def generate_balanced_gan_samples(
    G: ConditionalGenerator,
    n: int,
    z_dim: int,
    device: torch.device,
    num_classes: int = 10,
) -> Tuple[torch.Tensor, torch.Tensor]:
    per_class = max(1, n // num_classes)
    ys = torch.arange(0, num_classes, device=device).repeat_interleave(per_class)[:n]
    z = torch.randn(ys.size(0), z_dim, device=device)
    xs = G(z, ys)
    return xs.cpu(), ys.cpu()


@torch.no_grad()
def evaluate(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct = 0
    total = 0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.size(0)
    return correct / max(1, total)


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    cfg = DataConfig(root=args.data_root, batch_size=args.batch_size)
    _, test_loader = get_mnist_loaders_for_gan(cfg, train_subset=None)

    real_train_loader, _ = get_mnist_loaders_for_gan(cfg, train_subset=args.real_n, subset_seed=args.seed)

    real_xs, real_ys = [], []
    for x, y in real_train_loader:
        real_xs.append(x)
        real_ys.append(y)
    real_x = torch.cat(real_xs, dim=0)
    real_y = torch.cat(real_ys, dim=0)

    if args.gan_ckpt is not None and args.aug_n > 0:
        ckpt = torch.load(args.gan_ckpt, map_location=device)
        if "G" not in ckpt:
            raise ValueError("GAN checkpoint must contain key 'G' with generator state_dict.")

        G = ConditionalGenerator(z_dim=args.z_dim).to(device)
        G.load_state_dict(ckpt["G"])
        G.eval()

        aug_x, aug_y = generate_balanced_gan_samples(G, n=args.aug_n, z_dim=args.z_dim, device=device)
        x_train = torch.cat([real_x, aug_x], dim=0)
        y_train = torch.cat([real_y, aug_y], dim=0)
    else:
        x_train, y_train = real_x, real_y

    train_ds = TensorDataset(x_train, y_train)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=False)

    model = LeNet5().to(device)
    opt = Adam(model.parameters(), lr=args.lr)

    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_m = AverageMeter("loss")

        for x, y in tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", leave=False):
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            loss = F.cross_entropy(logits, y)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            loss_m.update(loss.item(), x.size(0))

        acc = evaluate(model, test_loader, device=device)
        print(f"Epoch {epoch:02d} | {loss_m} | test_acc: {acc:.4f}")
        torch.save({"model": model.state_dict(), "args": vars(args), "epoch": epoch, "test_acc": acc}, outdir / "last.pt")

    final_acc = evaluate(model, test_loader, device=device)
    print(f"Final test accuracy: {final_acc:.4f}")


if __name__ == "__main__":
    main()
