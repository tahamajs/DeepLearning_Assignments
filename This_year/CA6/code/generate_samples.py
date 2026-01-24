from __future__ import annotations

import argparse

import torch

from deepgen.models import ConditionalGenerator, VampPriorVAE
from deepgen.utils import save_image_grid


def parse_args():
    p = argparse.ArgumentParser(description="Generate samples from trained VAE or GAN.")
    p.add_argument("--ckpt", type=str, required=True, help="checkpoint path")
    p.add_argument("--out", type=str, default="./samples.png")
    p.add_argument("--model", type=str, choices=["vae", "gan"], required=True)
    p.add_argument("--n", type=int, default=100)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--z_dim", type=int, default=128, help="GAN z_dim (if model=gan)")
    return p.parse_args()


@torch.no_grad()
def main():
    args = parse_args()
    device = torch.device(args.device)
    ckpt = torch.load(args.ckpt, map_location=device)

    if args.model == "vae":
        margs = ckpt.get("args", {})
        model = VampPriorVAE(
            z_dim=int(margs.get("z_dim", 16)),
            h_dim=int(margs.get("h_dim", 400)),
            vamp_k=int(margs.get("vamp_k", 0)),
            beta=float(margs.get("beta", 1.0)),
        ).to(device)
        model.load_state_dict(ckpt["model"])
        model.eval()

        samples = model.sample(args.n, device=device)
        save_image_grid(samples, args.out, nrow=10, normalize=True, value_range=(0, 1))

    else:
        if "G" not in ckpt:
            raise ValueError("GAN checkpoint must contain key 'G'.")
        model = ConditionalGenerator(z_dim=args.z_dim).to(device)
        model.load_state_dict(ckpt["G"])
        model.eval()

        y = torch.arange(0, 10, device=device).repeat_interleave(args.n // 10 + 1)[: args.n]
        z = torch.randn(args.n, args.z_dim, device=device)
        samples = model(z, y)
        save_image_grid(samples, args.out, nrow=10, normalize=True, value_range=(-1, 1))


if __name__ == "__main__":
    main()
