"""
Common helper functions: metrics, seeding, save/load
"""
import os
import random
import torch
import numpy as np


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def save_checkpoint(model, path):
    ensure_dir(os.path.dirname(path))
    torch.save(model.state_dict(), path)


def load_checkpoint(model, path, map_location=None):
    model.load_state_dict(torch.load(path, map_location=map_location))
    return model


def save_figure(fig, path, dpi=300, tight=True):
    """Save a matplotlib figure to disk ensuring directories exist.

    Args:
        fig: matplotlib.figure.Figure or pyplot object
        path: destination file path (including filename).
        dpi: resolution for publication-quality images.
        tight: whether to call plt.tight_layout() before saving.
    """
    ensure_dir(os.path.dirname(path))
    try:
        # If fig is a pyplot module (plt), call savefig on plt
        fig.savefig(path, dpi=dpi, bbox_inches='tight' if tight else None)
    except Exception:
        # If fig is a Figure instance
        fig.figure.savefig(path, dpi=dpi, bbox_inches='tight' if tight else None)
    return path
