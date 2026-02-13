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


def seed_everything(seed=42):
    """Compatibility wrapper: alias for set_seed (used across notebooks)."""
    set_seed(seed)


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def save_checkpoint(model, path):
    ensure_dir(os.path.dirname(path))
    torch.save(model.state_dict(), path)


def load_checkpoint(model, path, map_location=None):
    model.load_state_dict(torch.load(path, map_location=map_location))
    return model


def save_figure(fig, path, dpi=300, tight=True, save_pdf=True):
    """Save a matplotlib figure or PIL image to PNG and optionally PDF.

    Returns a dict with 'png' and 'pdf' keys (pdf may be None if not created).
    """
    from PIL import Image
    ensure_dir(os.path.dirname(path))
    png_path = path
    pdf_path = None
    try:
        # Matplotlib Figure or pyplot
        try:
            fig.savefig(png_path, dpi=dpi, bbox_inches='tight' if tight else None)
        except Exception:
            # If fig is matplotlib.pyplot module
            fig.figure.savefig(png_path, dpi=dpi, bbox_inches='tight' if tight else None)
        if save_pdf:
            pdf_path = os.path.splitext(png_path)[0] + '.pdf'
            try:
                # try saving as vector/pdf directly
                try:
                    fig.savefig(pdf_path, format='pdf', bbox_inches='tight' if tight else None)
                except Exception:
                    # fallback: rasterize and save
                    fig.figure.savefig(pdf_path, format='pdf', bbox_inches='tight' if tight else None)
            except Exception:
                pdf_path = None
    except Exception:
        # If fig is a PIL Image
        if isinstance(fig, Image.Image):
            fig.save(png_path)
            if save_pdf:
                pdf_path = os.path.splitext(png_path)[0] + '.pdf'
                try:
                    fig.convert('RGB').save(pdf_path, 'PDF', resolution=dpi)
                except Exception:
                    pdf_path = None
        else:
            raise
    return {'png': png_path, 'pdf': pdf_path}



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


def _latex_safe_label(fname):
    import re
    base = os.path.splitext(os.path.basename(fname))[0]
    label = re.sub(r'[^0-9a-zA-Z]+', '_', base)
    return 'fig:' + label


def write_latex_figure_snippet(pdf_path, caption, label=None, width='\\columnwidth', tex_path=None):
    """Write a LaTeX figure snippet referencing the PDF file. Returns path of tex file."""
    if label is None:
        label = _latex_safe_label(pdf_path)
    if tex_path is None:
        tex_path = os.path.splitext(pdf_path)[0] + '.tex'
    tex = f"""\\begin{{figure}}[t]
\\centering
\\includegraphics[width={width}]{{{os.path.basename(pdf_path)}}}
\\caption{{{caption}}}
\\label{{{label}}}
\\end{{figure}}
"""
    with open(tex_path, 'w') as f:
        f.write(tex)
    return tex_path


def save_asset_manifest(manifest, images_dir, create_ieee_assets=True):
    """Save manifest CSV; copy PDFs to an `ieee_assets/` folder and write LaTeX snippets.

    Returns path to manifest CSV and a list of generated tex files.
    """
    import csv
    from shutil import copy2
    ensure_dir(images_dir)
    path = os.path.join(images_dir, 'manifest.csv')
    with open(path, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=['filename','width_in','height_in','dpi','caption_placeholder'])
        writer.writeheader()
        for r in manifest:
            writer.writerow(r)

    tex_files = []
    if create_ieee_assets:
        assets_dir = os.path.join(images_dir, 'ieee_assets')
        ensure_dir(assets_dir)
        for r in manifest:
            fname = r['filename']
            if isinstance(fname, str):
                base = os.path.basename(fname)
                pdf_src = os.path.splitext(fname)[0] + '.pdf'
                if os.path.exists(pdf_src):
                    pdf_dst = os.path.join(assets_dir, os.path.basename(pdf_src))
                    copy2(pdf_src, pdf_dst)
                    # write TEX snippet
                    caption = r.get('caption_placeholder', 'Figure')
                    tex_path = os.path.join(assets_dir, os.path.splitext(os.path.basename(pdf_src))[0] + '.tex')
                    write_latex_figure_snippet(pdf_dst, caption, label=None, width='0.48\\textwidth', tex_path=tex_path)
                    tex_files.append(tex_path)
    return path, tex_files
