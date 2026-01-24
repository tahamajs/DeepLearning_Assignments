# train_vae.py
from __future__ import annotations

import argparse
import os
from typing import Dict, Any

import torch
from torch.optim import Adam
from tqdm import tqdm
from torchvision.utils import save_image

from deepgen.data import get_mnist_loaders
from deepgen.models.vae import VAEVampPrior
from deepgen.utils import (
    AverageMeter,
    get_device,
    load_checkpoint,
    restore_rng_state,
    save_checkpoint,
    set_seed,
)

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", type=str, default="./data")
    p.add_argument("--out_dir", type=str, default="./runs/vae")
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--z_dim", type=int, default=32)
    p.add_argument("--n_pseudos", type=int, default=500)
    p.add_argument("--beta", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--save_every", type=int, default=5)
    p.add_argument("--sample_every", type=int, default=1)
    p.add_argument("--resume", type=str, default=None)
    return p.parse_args()

def main() -> None:
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    set_seed(args.seed)
    device = get_device(args.device)

    train_loader, test_loader = get_mnist_loaders(
        root=args.data_root, batch_size=args.batch_size, train_subset=None, scale_to_minus1_1=True
    )

    model = VAEVampPrior(z_dim=args.z_dim, n_pseudos=args.n_pseudos).to(device)
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
        print(f"[VAE] Resumed from {args.resume} at epoch={start_epoch}")

    for epoch in range(start_epoch, args.epochs):
        model.train()
        loss_m = AverageMeter("loss")
        recon_m = AverageMeter("recon")
        kl_m = AverageMeter("kl")

        for x, _ in tqdm(train_loader, desc=f"VAE Epoch {epoch+1}/{args.epochs}"):
            x = x.to(device)
            x_rec, mu, logvar = model(x)
            loss, recon, kl = model.elbo_loss(x, x_rec, mu, logvar, beta=args.beta)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            loss_m.update(loss.item(), x.size(0))
            recon_m.update(recon.item(), x.size(0))
            kl_m.update(kl.item(), x.size(0))

        print(f"[VAE] epoch={epoch+1} loss={loss_m.avg:.3f} recon={recon_m.avg:.3f} kl={kl_m.avg:.3f}")

        if (epoch + 1) % args.sample_every == 0:
            model.eval()
            with torch.no_grad():
                samples = model.sample(64, device=device)  # [0,1]
                grid_path = os.path.join(args.out_dir, f"samples_epoch{epoch+1:03d}.png")
                save_image(samples, grid_path, nrow=8)

        if (epoch + 1) % args.save_every == 0 or (epoch + 1) == args.epochs:
            ckpt_path = os.path.join(args.out_dir, f"vae_epoch{epoch+1:03d}.pt")
            save_checkpoint(
                ckpt_path,
                model_state=model.state_dict(),
                optim_state=opt.state_dict(),
                extra={"epoch": epoch + 1, "args": vars(args)},
            )
            print(f"[VAE] Saved checkpoint: {ckpt_path}")

if __name__ == "__main__":
    main()
