# evaluate_fid.py
from __future__ import annotations

import argparse
import os
from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from deepgen.data import get_mnist_loaders
from deepgen.metrics import extract_features, fid_like
from deepgen.models.classifier import LeNet5
from deepgen.models.gan import Generator
from deepgen.models.vae import VAEVampPrior
from deepgen.utils import get_device, load_checkpoint, set_seed


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", type=str, default="./data")
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--seed", type=int, default=42)

    # Feature extractor (LeNet)
    p.add_argument("--feat_ckpt", type=str, required=True, help="Checkpoint from train_classifier.py (trained on real MNIST, ideally full 60000).")
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--n_real", type=int, default=10000, help="How many real images to use (test set recommended).")
    p.add_argument("--split", type=str, choices=["test", "train"], default="test")

    # Generated model choice
    p.add_argument("--model", type=str, choices=["gan", "vae"], required=True)
    p.add_argument("--gen_ckpt", type=str, required=True)

    # GAN options
    p.add_argument("--gan_z_dim", type=int, default=128)
    p.add_argument("--n_gen", type=int, default=10000, help="How many generated images to use.")
    p.add_argument("--classwise", action="store_true", help="Compute per-class FID-like (GAN only).")
    p.add_argument("--out_path", type=str, default=None)
    return p.parse_args()


@torch.no_grad()
def sample_from_gan(gen_ckpt: str, n: int, z_dim: int, device: torch.device) -> torch.Tensor:
    ckpt = load_checkpoint(gen_ckpt, map_location=device)
    G = Generator(z_dim=z_dim, n_classes=10).to(device)
    G.load_state_dict(ckpt["model"]["G"])
    G.eval()
    y = torch.randint(0, 10, (n,), device=device)
    z = torch.randn(n, z_dim, device=device)
    x = G(z, y)  # [-1,1]
    return x


@torch.no_grad()
def sample_from_vae(vae_ckpt: str, n: int, device: torch.device) -> torch.Tensor:
    ckpt = load_checkpoint(vae_ckpt, map_location=device)
    # Recover dimensions if possible
    extra = ckpt.get("extra", {})
    args = extra.get("args", {})
    z_dim = int(args.get("z_dim", 32))
    n_pseudos = int(args.get("n_pseudos", 500))
    V = VAEVampPrior(z_dim=z_dim, n_pseudos=n_pseudos).to(device)
    V.load_state_dict(ckpt["model"])
    V.eval()
    x01 = V.sample(n, device=device)  # [0,1]
    x = x01 * 2.0 - 1.0               # [-1,1] for LeNet
    return x


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = get_device(args.device)

    # Load feature extractor
    feat = LeNet5(n_classes=10).to(device)
    feat_ckpt = load_checkpoint(args.feat_ckpt, map_location=device)
    feat.load_state_dict(feat_ckpt["model"])
    feat.eval()

    # Real loader
    train_loader, test_loader = get_mnist_loaders(
        root=args.data_root, batch_size=args.batch_size, train_subset=None, scale_to_minus1_1=True
    )
    real_loader = test_loader if args.split == "test" else train_loader
    real_feats = extract_features(feat, real_loader, device=device, max_items=args.n_real)

    if args.model == "gan":
        gen_x = sample_from_gan(args.gen_ckpt, args.n_gen, args.gan_z_dim, device)
        # Need a loader to compute features in batches
        gen_ds = TensorDataset(gen_x.cpu(), torch.zeros(gen_x.size(0), dtype=torch.long))
        gen_loader = DataLoader(gen_ds, batch_size=args.batch_size, shuffle=False, drop_last=False)
        gen_feats = extract_features(feat, gen_loader, device=device, max_items=args.n_gen)

        score = fid_like(real_feats, gen_feats)
        print(f"[FID-like] GAN vs {args.split} (n_real={len(real_feats)}, n_gen={len(gen_feats)}): {score:.4f}")

        if args.classwise:
            # Compute per-class: sample fixed labels and compute FID per label
            per_scores = {}
            for cls in range(10):
                y = torch.full((args.n_gen,), cls, device=device, dtype=torch.long)
                z = torch.randn(args.n_gen, args.gan_z_dim, device=device)

                ckpt = load_checkpoint(args.gen_ckpt, map_location=device)
                G = Generator(z_dim=args.gan_z_dim, n_classes=10).to(device)
                G.load_state_dict(ckpt["model"]["G"])
                G.eval()
                x = G(z, y)

                ds = TensorDataset(x.cpu(), torch.zeros(x.size(0), dtype=torch.long))
                dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, drop_last=False)
                f = extract_features(feat, dl, device=device, max_items=args.n_gen)

                # Real class features: filter from dataset split
                # For efficiency, collect class-specific real samples once using the split loader.
                # Here: approximate by iterating split loader and selecting.
                real_cls = []
                for bx, by in real_loader:
                    mask = (by == cls)
                    if mask.any():
                        real_cls.append(bx[mask])
                    if sum(t.size(0) for t in real_cls) >= args.n_real:
                        break
                if len(real_cls) == 0:
                    continue
                real_cls_x = torch.cat(real_cls, dim=0)[:args.n_real]
                rds = TensorDataset(real_cls_x.cpu(), torch.zeros(real_cls_x.size(0), dtype=torch.long))
                rdl = DataLoader(rds, batch_size=args.batch_size, shuffle=False, drop_last=False)
                rf = extract_features(feat, rdl, device=device, max_items=args.n_real)

                per_scores[int(cls)] = fid_like(rf, f)
                print(f"  class={cls} fid_like={per_scores[int(cls)]:.4f}")

    else:
        gen_x = sample_from_vae(args.gen_ckpt, args.n_gen, device)
        gen_ds = TensorDataset(gen_x.cpu(), torch.zeros(gen_x.size(0), dtype=torch.long))
        gen_loader = DataLoader(gen_ds, batch_size=args.batch_size, shuffle=False, drop_last=False)
        gen_feats = extract_features(feat, gen_loader, device=device, max_items=args.n_gen)

        score = fid_like(real_feats, gen_feats)
        print(f"[FID-like] VAE vs {args.split} (n_real={len(real_feats)}, n_gen={len(gen_feats)}): {score:.4f}")

    if args.out_path is not None:
        os.makedirs(os.path.dirname(args.out_path), exist_ok=True)
        with open(args.out_path, "w", encoding="utf-8") as f:
            f.write(str(score) + "\n")


if __name__ == "__main__":
    main()
