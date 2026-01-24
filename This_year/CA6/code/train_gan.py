from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.optim import Adam
from tqdm import tqdm

from deepgen.data import DataConfig, get_mnist_loaders_for_gan
from deepgen.models import ConditionalGenerator, ProjectionDiscriminator
from deepgen.utils import gradient_penalty, save_image_grid, set_seed


def parse_args():
    p = argparse.ArgumentParser(description="Train conditional WGAN-GP on MNIST.")
    p.add_argument("--data_root", type=str, default="./data")
    p.add_argument("--outdir", type=str, default="./runs/gan")
    p.add_argument("--z_dim", type=int, default=128)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--steps", type=int, default=50000)
    p.add_argument("--lr_g", type=float, default=2e-4)
    p.add_argument("--lr_d", type=float, default=2e-4)
    p.add_argument("--beta1", type=float, default=0.0)
    p.add_argument("--beta2", type=float, default=0.9)
    p.add_argument("--n_critic", type=int, default=5)
    p.add_argument("--lambda_gp", type=float, default=10.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--train_subset", type=int, default=None)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    cfg = DataConfig(root=args.data_root, batch_size=args.batch_size)
    train_loader, _ = get_mnist_loaders_for_gan(cfg, train_subset=args.train_subset, subset_seed=args.seed)
    train_iter = iter(train_loader)

    G = ConditionalGenerator(z_dim=args.z_dim).to(device)
    D = ProjectionDiscriminator().to(device)

    opt_g = Adam(G.parameters(), lr=args.lr_g, betas=(args.beta1, args.beta2))
    opt_d = Adam(D.parameters(), lr=args.lr_d, betas=(args.beta1, args.beta2))

    fixed_z = torch.randn(100, args.z_dim, device=device)
    fixed_y = torch.arange(0, 10, device=device).repeat_interleave(10)

    for step in tqdm(range(1, args.steps + 1), desc="Training"):
        for _ in range(args.n_critic):
            try:
                real, y = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                real, y = next(train_iter)

            real = real.to(device)
            y = y.to(device)

            z = torch.randn(real.size(0), args.z_dim, device=device)
            fake = G(z, y).detach()

            d_real = D(real, y).mean()
            d_fake = D(fake, y).mean()
            gp = gradient_penalty(D, real, fake, labels=y, lambda_gp=args.lambda_gp)
            d_loss = -(d_real - d_fake) + gp

            opt_d.zero_grad(set_to_none=True)
            d_loss.backward()
            opt_d.step()

        try:
            _, y = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            _, y = next(train_iter)

        y = y.to(device)
        z = torch.randn(y.size(0), args.z_dim, device=device)
        fake = G(z, y)
        g_loss = -D(fake, y).mean()

        opt_g.zero_grad(set_to_none=True)
        g_loss.backward()
        opt_g.step()

        if step % 500 == 0:
            with torch.no_grad():
                samples = G(fixed_z, fixed_y)
                save_image_grid(
                    samples,
                    str(outdir / f"samples_step{step:06d}.png"),
                    nrow=10,
                    normalize=True,
                    value_range=(-1, 1),
                )
            ckpt = {"G": G.state_dict(), "D": D.state_dict(), "args": vars(args), "step": step}
            torch.save(ckpt, outdir / "last.pt")
            print(f"step {step} | d_loss {d_loss.item():.4f} | g_loss {g_loss.item():.4f}")

    torch.save({"G": G.state_dict(), "D": D.state_dict(), "args": vars(args), "step": args.steps}, outdir / "final.pt")


if __name__ == "__main__":
    main()
