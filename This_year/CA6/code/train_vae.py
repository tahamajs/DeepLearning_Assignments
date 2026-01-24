from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.optim import Adam
from tqdm import tqdm

from deepgen.data import DataConfig, get_dynamic_mnist_loaders
from deepgen.models import VampPriorVAE
from deepgen.utils import AverageMeter, save_image_grid, set_seed


def parse_args():
    p = argparse.ArgumentParser(description="Train VAE (optionally VampPrior) on DynamicMNIST.")
    p.add_argument("--data_root", type=str, default="./data")
    p.add_argument("--outdir", type=str, default="./runs/vae")
    p.add_argument("--z_dim", type=int, default=16)
    p.add_argument("--h_dim", type=int, default=400)
    p.add_argument("--vamp_k", type=int, default=0)
    p.add_argument("--beta", type=float, default=1.0)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--train_subset", type=int, default=None, help="e.g., 100, 1000 (balanced across classes)")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    cfg = DataConfig(root=args.data_root, batch_size=args.batch_size)
    train_loader, test_loader = get_dynamic_mnist_loaders(
        cfg, train_subset=args.train_subset, subset_seed=args.seed, flatten=True
    )

    model = VampPriorVAE(z_dim=args.z_dim, h_dim=args.h_dim, vamp_k=args.vamp_k, beta=args.beta).to(device)
    opt = Adam(model.parameters(), lr=args.lr)

    best_elbo = None

    for epoch in range(1, args.epochs + 1):
        model.train()
        elbo_m = AverageMeter("elbo")
        recon_m = AverageMeter("recon")
        kl_m = AverageMeter("kl")

        for x, _ in tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", leave=False):
            x = x.to(device)
            out = model(x)
            loss = -out.elbo

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            bsz = x.size(0)
            elbo_m.update(out.elbo.item() / bsz, bsz)
            recon_m.update(out.recon_loss.item() / bsz, bsz)
            kl_m.update(out.kl.item() / bsz, bsz)

        model.eval()
        with torch.no_grad():
            test_elbo = 0.0
            test_n = 0
            for x, _ in test_loader:
                x = x.to(device)
                out = model(x)
                test_elbo += out.elbo.item()
                test_n += x.size(0)
            test_elbo /= max(1, test_n)

        samples = model.sample(n=100, device=device)
        save_image_grid(
            samples,
            str(outdir / f"samples_epoch{epoch:03d}.png"),
            nrow=10,
            normalize=True,
            value_range=(0, 1),
        )

        ckpt = {"model": model.state_dict(), "args": vars(args), "epoch": epoch, "test_elbo": test_elbo}
        torch.save(ckpt, outdir / "last.pt")
        if best_elbo is None or test_elbo > best_elbo:
            best_elbo = test_elbo
            torch.save(ckpt, outdir / "best.pt")

        print(f"Epoch {epoch:03d} | train {elbo_m} {recon_m} {kl_m} | test_elbo: {test_elbo:.4f}")


if __name__ == "__main__":
    main()
