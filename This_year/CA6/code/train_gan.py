# train_gan.py
from __future__ import annotations

import argparse
import os
from typing import Dict, Any

import torch
from torch.optim import Adam
from tqdm import tqdm
from torchvision.utils import save_image

from deepgen.data import get_mnist_loaders
from deepgen.models.gan import Generator, ProjectionDiscriminator, gradient_penalty
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
    p.add_argument("--out_dir", type=str, default="./runs/gan")
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--steps", type=int, default=50000)
    p.add_argument("--z_dim", type=int, default=128)
    p.add_argument("--n_classes", type=int, default=10)
    p.add_argument("--lr_g", type=float, default=2e-4)
    p.add_argument("--lr_d", type=float, default=2e-4)
    p.add_argument("--betas", type=float, nargs=2, default=(0.0, 0.9))
    p.add_argument("--n_critic", type=int, default=5)
    p.add_argument("--gp_lambda", type=float, default=10.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--save_every", type=int, default=5000)
    p.add_argument("--sample_every", type=int, default=1000)
    p.add_argument("--resume", type=str, default=None)
    return p.parse_args()

def main() -> None:
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    set_seed(args.seed)
    device = get_device(args.device)

    train_loader, _ = get_mnist_loaders(
        root=args.data_root, batch_size=args.batch_size, train_subset=None, scale_to_minus1_1=True
    )
    train_iter = iter(train_loader)

    G = Generator(z_dim=args.z_dim, n_classes=args.n_classes).to(device)
    D = ProjectionDiscriminator(n_classes=args.n_classes).to(device)
    opt_g = Adam(G.parameters(), lr=args.lr_g, betas=tuple(args.betas))
    opt_d = Adam(D.parameters(), lr=args.lr_d, betas=tuple(args.betas))

    step0 = 0
    if args.resume is not None:
        ckpt = load_checkpoint(args.resume, map_location=device)
        G.load_state_dict(ckpt["model"]["G"])
        D.load_state_dict(ckpt["model"]["D"])
        if ckpt.get("optim") is not None:
            opt_g.load_state_dict(ckpt["optim"]["opt_g"])
            opt_d.load_state_dict(ckpt["optim"]["opt_d"])
        extra = ckpt.get("extra", {})
        step0 = int(extra.get("step", 0))
        if "rng" in ckpt:
            restore_rng_state(ckpt["rng"])
        print(f"[GAN] Resumed from {args.resume} at step={step0}")

    def sample_noise(batch: int) -> torch.Tensor:
        return torch.randn(batch, args.z_dim, device=device)

    fixed_z = sample_noise(80)
    fixed_y = torch.arange(0, 10, device=device).repeat_interleave(8)  # 8 per class

    for step in range(step0, args.steps):
        # -------------------------
        # Train Discriminator
        # -------------------------
        for _ in range(args.n_critic):
            try:
                real_x, real_y = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                real_x, real_y = next(train_iter)

            real_x = real_x.to(device)
            real_y = real_y.to(device)

            z = sample_noise(real_x.size(0))
            fake_x = G(z, real_y).detach()

            d_real = D(real_x, real_y).mean()
            d_fake = D(fake_x, real_y).mean()
            gp = gradient_penalty(D, real_x, fake_x, real_y)

            # WGAN-GP: maximize d_real - d_fake - lambda*gp
            d_loss = (d_fake - d_real) + args.gp_lambda * gp

            opt_d.zero_grad(set_to_none=True)
            d_loss.backward()
            opt_d.step()

        # -------------------------
        # Train Generator
        # -------------------------
        try:
            _, y = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            _, y = next(train_iter)
        y = y.to(device)

        z = sample_noise(y.size(0))
        fake_x = G(z, y)
        g_loss = -D(fake_x, y).mean()

        opt_g.zero_grad(set_to_none=True)
        g_loss.backward()
        opt_g.step()

        if (step + 1) % 200 == 0:
            print(f"[GAN] step={step+1} d_loss={d_loss.item():.3f} g_loss={g_loss.item():.3f} gp={gp.item():.3f}")

        if (step + 1) % args.sample_every == 0:
            G.eval()
            with torch.no_grad():
                imgs = G(fixed_z, fixed_y)
                # imgs are [-1,1]; convert to [0,1] for visualization
                imgs01 = (imgs + 1.0) / 2.0
                grid_path = os.path.join(args.out_dir, f"samples_step{step+1:06d}.png")
                save_image(imgs01, grid_path, nrow=10)
            G.train()

        if (step + 1) % args.save_every == 0 or (step + 1) == args.steps:
            ckpt_path = os.path.join(args.out_dir, f"gan_step{step+1:06d}.pt")
            save_checkpoint(
                ckpt_path,
                model_state={"G": G.state_dict(), "D": D.state_dict()},
                optim_state={"opt_g": opt_g.state_dict(), "opt_d": opt_d.state_dict()},
                extra={"step": step + 1, "args": vars(args)},
            )
            print(f"[GAN] Saved checkpoint: {ckpt_path}")

if __name__ == "__main__":
    main()
