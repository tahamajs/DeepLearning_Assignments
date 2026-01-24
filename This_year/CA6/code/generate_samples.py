# generate_samples.py
from __future__ import annotations

import argparse
import os

import torch
from torchvision.utils import save_image

from deepgen.models.gan import Generator
from deepgen.utils import get_device, load_checkpoint, set_seed


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--out", type=str, default="./samples.png")
    p.add_argument("--z_dim", type=int, default=128)
    p.add_argument("--n_per_class", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="auto")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = get_device(args.device)

    ckpt = load_checkpoint(args.ckpt, map_location=device)
    G = Generator(z_dim=args.z_dim, n_classes=10).to(device)
    G.load_state_dict(ckpt["model"]["G"])
    G.eval()

    y = torch.arange(0, 10, device=device).repeat_interleave(args.n_per_class)
    z = torch.randn(y.size(0), args.z_dim, device=device)
    with torch.no_grad():
        x = G(z, y)
        x01 = (x + 1.0) / 2.0

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    save_image(x01, args.out, nrow=10)
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
