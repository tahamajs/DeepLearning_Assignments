# deepgen/metrics.py
from __future__ import annotations

from typing import Tuple, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader


@torch.no_grad()
def extract_features(
    feature_model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    max_items: Optional[int] = None,
) -> np.ndarray:
    feature_model.eval()
    feats = []
    seen = 0
    for x, _y in loader:
        x = x.to(device)
        f = feature_model(x, return_features=True)  # (B, D)
        f = f.detach().float().cpu().numpy()
        feats.append(f)
        seen += f.shape[0]
        if max_items is not None and seen >= max_items:
            break
    out = np.concatenate(feats, axis=0)
    if max_items is not None:
        out = out[:max_items]
    return out.astype(np.float64)


def _covariance(features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    # features: (N, D)
    mu = np.mean(features, axis=0)
    X = features - mu
    cov = (X.T @ X) / max(1, (features.shape[0] - 1))
    return mu, cov


def _sqrtm_psd(mat: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    # Symmetric PSD matrix square root via eigen-decomposition.
    mat = 0.5 * (mat + mat.T)
    w, v = np.linalg.eigh(mat)
    w = np.clip(w, eps, None)
    return (v * np.sqrt(w)) @ v.T


def fid_like(features_real: np.ndarray, features_fake: np.ndarray, eps: float = 1e-6) -> float:
    mu_r, cov_r = _covariance(features_real)
    mu_f, cov_f = _covariance(features_fake)

    cov_r = cov_r + np.eye(cov_r.shape[0]) * eps
    cov_f = cov_f + np.eye(cov_f.shape[0]) * eps

    diff = mu_r - mu_f
    diff_sq = diff @ diff

    sqrt_cov_r = _sqrtm_psd(cov_r)
    # Symmetric middle term
    mid = sqrt_cov_r @ cov_f @ sqrt_cov_r
    mid = 0.5 * (mid + mid.T)
    sqrt_mid = _sqrtm_psd(mid)

    trace = np.trace(cov_r + cov_f - 2.0 * sqrt_mid)
    return float(diff_sq + trace)
