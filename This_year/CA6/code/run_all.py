# run_all.py
from __future__ import annotations

import argparse
import os
import subprocess
import sys


def run(cmd: list[str]) -> None:
    print("\n>>>", " ".join(cmd))
    subprocess.check_call(cmd)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", type=str, default="./data")
    p.add_argument("--runs_root", type=str, default="./runs")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="auto")

    # Fast defaults (edit as needed)
    p.add_argument("--vae_epochs", type=int, default=20)
    p.add_argument("--gan_steps", type=int, default=30000)

    p.add_argument("--real_n_small", type=int, default=1000)
    p.add_argument("--aug_n", type=int, default=30000)

    p.add_argument("--fid_n", type=int, default=10000)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.runs_root, exist_ok=True)
    py = sys.executable

    vae_dir = os.path.join(args.runs_root, "vae")
    gan_dir = os.path.join(args.runs_root, "gan")
    cls_dir = os.path.join(args.runs_root, "classifier")
    fid_dir = os.path.join(args.runs_root, "fid")
    os.makedirs(fid_dir, exist_ok=True)

    # 1) Train VAE
    run([
        py, "train_vae.py",
        "--data_root", args.data_root,
        "--out_dir", vae_dir,
        "--epochs", str(args.vae_epochs),
        "--seed", str(args.seed),
        "--device", args.device,
    ])
    vae_ckpt = os.path.join(vae_dir, f"vae_epoch{args.vae_epochs:03d}.pt")

    # 2) Train GAN
    run([
        py, "train_gan.py",
        "--data_root", args.data_root,
        "--out_dir", gan_dir,
        "--steps", str(args.gan_steps),
        "--seed", str(args.seed),
        "--device", args.device,
    ])
    gan_ckpt = os.path.join(gan_dir, f"gan_step{args.gan_steps:06d}.pt")

    # 3) Train feature extractor (LeNet) on full MNIST (for FID-like)
    run([
        py, "train_classifier.py",
        "--data_root", args.data_root,
        "--out_dir", cls_dir,
        "--real_n", "60000",
        "--aug_source", "none",
        "--epochs", "10",
        "--seed", str(args.seed),
        "--device", args.device,
    ])
    feat_ckpt = os.path.join(cls_dir, "lenet_real60000_augnone0_epoch010.pt")

    # 4) Classifier baseline with small real_n
    run([
        py, "train_classifier.py",
        "--data_root", args.data_root,
        "--out_dir", cls_dir,
        "--real_n", str(args.real_n_small),
        "--aug_source", "none",
        "--epochs", "10",
        "--seed", str(args.seed),
        "--device", args.device,
    ])

    # 5) Classifier with GAN augmentation
    run([
        py, "train_classifier.py",
        "--data_root", args.data_root,
        "--out_dir", cls_dir,
        "--real_n", str(args.real_n_small),
        "--aug_source", "gan",
        "--aug_n", str(args.aug_n),
        "--gan_ckpt", gan_ckpt,
        "--epochs", "10",
        "--seed", str(args.seed),
        "--device", args.device,
    ])

    # 6) FID-like evaluation (GAN)
    run([
        py, "evaluate_fid.py",
        "--data_root", args.data_root,
        "--device", args.device,
        "--feat_ckpt", feat_ckpt,
        "--model", "gan",
        "--gen_ckpt", gan_ckpt,
        "--n_real", str(args.fid_n),
        "--n_gen", str(args.fid_n),
        "--split", "test",
        "--out_path", os.path.join(fid_dir, "fid_like_gan.txt"),
    ])

    # 7) FID-like evaluation (VAE)
    run([
        py, "evaluate_fid.py",
        "--data_root", args.data_root,
        "--device", args.device,
        "--feat_ckpt", feat_ckpt,
        "--model", "vae",
        "--gen_ckpt", vae_ckpt,
        "--n_real", str(args.fid_n),
        "--n_gen", str(args.fid_n),
        "--split", "test",
        "--out_path", os.path.join(fid_dir, "fid_like_vae.txt"),
    ])

    # 8) Generate a sample grid from GAN
    run([
        py, "generate_samples.py",
        "--ckpt", gan_ckpt,
        "--out", os.path.join(args.runs_root, "gan_samples.png"),
        "--seed", str(args.seed),
        "--device", args.device,
    ])

    print("\nAll done.")
    print("Key outputs:")
    print("  GAN samples:", os.path.join(args.runs_root, "gan_samples.png"))
    print("  FID-like GAN:", os.path.join(fid_dir, "fid_like_gan.txt"))
    print("  FID-like VAE:", os.path.join(fid_dir, "fid_like_vae.txt"))


if __name__ == "__main__":
    main()
