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


def make_fig_name(section, metric, desc, ext='png', images_dir=None):
    """Create a canonical filename for figures with timestamp and normalization."""
    from datetime import datetime
    if images_dir is None:
        images_dir = os.path.join(os.path.dirname(__file__), '..', 'q1_image_captioning', 'images')
    ts = datetime.now().strftime('%Y%m%d-%H%M%S')
    name = f"fig-{section}-{metric}-{desc}-{ts}.{ext}"
    name = name.replace(' ', '-').lower()
    ensure_dir(images_dir)
    return os.path.join(images_dir, name)


def save_asset_manifest(manifest, images_dir):
    """Save a manifest (list of dicts) to images_dir/manifest.csv and return path."""
    import csv
    ensure_dir(images_dir)
    path = os.path.join(images_dir, 'manifest.csv')
    with open(path, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=['filename','width_in','height_in','dpi','caption_placeholder'])
        writer.writeheader()
        for r in manifest:
            writer.writerow(r)
    return path
