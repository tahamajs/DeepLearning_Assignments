# train_classifier.py
from __future__ import annotations

import argparse
import os
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from deepgen.data import get_mnist_loaders
from deepgen.models.classifier import LeNet5
from deepgen.models.gan import Generator
from deepgen.utils import (
    get_device,
    load_checkpoint,
    restore_rng_state,
    save_checkpoint,
    set_seed,
)

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", type=str, default="./data")
    p.add_argument("--out_dir", type=str, default="./runs/classifier")
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="auto")

    # Real-data control
    p.add_argument("--real_n", type=int, default=1000, help="Number of real training samples (use 60000 for full MNIST train).")

    # Augmentation from GAN
    p.add_argument("--aug_source", type=str, choices=["none", "gan"], default="none")
    p.add_argument("--aug_n", type=int, default=0, help="Number of GAN-generated samples to add.")
    p.add_argument("--gan_ckpt", type=str, default=None)
    p.add_argument("--gan_z_dim", type=int, default=128)
    p.add_argument("--save_every", type=int, default=5)
    p.add_argument("--resume", type=str, default=None)
    return p.parse_args()

@torch.no_grad()
def make_gan_aug_dataset(
    gan_ckpt: str,
    n: int,
    z_dim: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    ckpt = load_checkpoint(gan_ckpt, map_location=device)
    G = Generator(z_dim=z_dim, n_classes=10).to(device)
    G.load_state_dict(ckpt["model"]["G"])
    G.eval()

    y = torch.randint(0, 10, (n,), device=device)
    z = torch.randn(n, z_dim, device=device)
    x = G(z, y)  # [-1,1]
    return x.cpu(), y.cpu()

def main() -> None:
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    set_seed(args.seed)
    device = get_device(args.device)

    train_loader, test_loader = get_mnist_loaders(
        root=args.data_root, batch_size=args.batch_size, train_subset=args.real_n, scale_to_minus1_1=True
    )

    # Collect real subset tensors for potential concatenation
    real_x_list, real_y_list = [], []
    for x, y in train_loader:
        real_x_list.append(x)
        real_y_list.append(y)
    real_x = torch.cat(real_x_list, dim=0)
    real_y = torch.cat(real_y_list, dim=0)

    if args.aug_source == "gan":
        if args.aug_n <= 0:
            raise ValueError("--aug_n must be > 0 when --aug_source=gan")
        if args.gan_ckpt is None:
            raise ValueError("--gan_ckpt is required when --aug_source=gan")
        aug_x, aug_y = make_gan_aug_dataset(args.gan_ckpt, args.aug_n, args.gan_z_dim, device)
        x_all = torch.cat([real_x, aug_x], dim=0)
        y_all = torch.cat([real_y, aug_y], dim=0)
    else:
        x_all, y_all = real_x, real_y

    train_ds = TensorDataset(x_all, y_all)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)

    model = LeNet5(n_classes=10).to(device)
    opt = Adam(model.parameters(), lr=args.lr)

    start_epoch = 0
    if args.resume is not None:
        ckpt = load_checkpoint(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        if ckpt.get("optim") is not None:
            opt.load_state_dict(ckpt["optim"])
        extra = ckpt.get("extra", {})
        start_epoch = int(extra.get("epoch", 0))
        if "rng" in ckpt:
            restore_rng_state(ckpt["rng"])
        print(f"[CLS] Resumed from {args.resume} at epoch={start_epoch}")

    def eval_acc() -> float:
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                pred = logits.argmax(dim=1)
                correct += (pred == y).sum().item()
                total += y.numel()
        model.train()
        return correct / max(1, total)

    for epoch in range(start_epoch, args.epochs):
        model.train()
        for x, y in tqdm(train_dl, desc=f"CLS Epoch {epoch+1}/{args.epochs}"):
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = F.cross_entropy(logits, y)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

        acc = eval_acc()
        print(f"[CLS] epoch={epoch+1} test_acc={acc*100:.2f}%")

        if (epoch + 1) % args.save_every == 0 or (epoch + 1) == args.epochs:
            tag = f"real{args.real_n}_aug{args.aug_source}{args.aug_n}"
            ckpt_path = os.path.join(args.out_dir, f"lenet_{tag}_epoch{epoch+1:03d}.pt")
            save_checkpoint(
                ckpt_path,
                model_state=model.state_dict(),
                optim_state=opt.state_dict(),
                extra={"epoch": epoch + 1, "args": vars(args), "test_acc": acc},
            )
            print(f"[CLS] Saved checkpoint: {ckpt_path}")

if __name__ == "__main__":
    main()
