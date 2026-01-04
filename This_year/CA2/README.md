# CA2 — Variational Autoencoders & PGMs 🧠🔬

**Short description:**
This folder contains assignment code, notebooks, generated figures and the written report for Assignment 2 of the Deep Generative Models course (VAE, β-VAE experiments on dSprites, MIG metric, PCA analysis, and probabilistic graphical model questions).

---

## 🔧 Quick Start

1. Clone repository (if not already local)

2. Create Python environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

(If your system uses conda, create a conda env and install same packages.)

3. Generate representative figures (fast / synthetic results):

```bash
cd This_year/CA2/codes
python generate_results.py
```

Output images will be saved to: `This_year/CA2/pictures/`.

4. To re-run full experiments / reproduce results (may take substantial time):

- Open and run the notebook: `This_year/CA2/codes/vae_implementation.ipynb` (run cells sequentially). This downloads dSprites, trains VAE and β-VAEs, computes MIG and PCA, and saves models & results under:
  - `This_year/CA2/models/`
  - `This_year/CA2/results/`
  - `This_year/CA2/pictures/`

---

## 📁 Repository structure (CA2-focused)

- `This_year/CA2/`
  - `codes/` — notebooks and scripts
    - `vae_implementation.ipynb` — main training & evaluation notebook
    - `generate_results.py` — script to synthesize representative figures (fast)
    - `code.ipynb` — supporting material / experiments
  - `pictures/` — generated figures (`training_curves_comparison.png`, `reconstructions_comparison.png`, `mig_comparison.png`, `pca_β1.png` etc.)
  - `models/` — saved model checkpoints from training
  - `results/` — numeric outputs saved as `.npy` or `.csv` (e.g., `mig_results.npy`)
  - `report/` — LaTeX source and `report.tex` for PDF generation

---

## 📝 What is included in this assignment

- Part 1: Probabilistic Graphical Models — theory, joint distributions, d-separation reasoning and proofs.
- Part 2: Variational Autoencoders — implementation and experiments on dSprites:
  - Standard VAE (β=1) and β-VAE (β>1) variants
  - Training curves (reconstruction, KL, total loss)
  - Reconstructions (original vs reconstructed images)
  - MIG (Mutual Information Gap) computation and comparison
  - PCA analysis of latent spaces
  - Short surveys of VQ-VAE, VampPrior, SC-VAE

---

## ▶️ Reproducibility & commands

- Run quick generation (no heavy training): `python generate_results.py`
- Full training (notebook): run `vae_implementation.ipynb` from top to bottom.
- Compile the report (PDF) from `This_year/CA2/report/`:

```bash
cd This_year/CA2/report
latexmk -pdf report.tex
# or
pdflatex report.tex && bibtex report && pdflatex report.tex
```

Notes:
- Full training uses PyTorch; GPU recommended for reasonable runtimes. Use the `device` selection in notebooks (CUDA / MPS / CPU).
- The `generate_results.py` script provides representative figures if you cannot train models locally.

---

## ✅ Results / Outputs

- Figures: saved to `This_year/CA2/pictures/` (e.g. `training_curves_comparison.png`, `reconstructions_comparison.png`, `mig_comparison.png`, `pca_β4.png`).
- MIG results saved to `This_year/CA2/results/mig_results.npy`.
- Model checkpoints in `This_year/CA2/models/` (if training was executed).

---

## ⚙️ Environment & dependencies

Minimum recommended:
- Python 3.8+
- torch, torchvision
- numpy, scipy, scikit-learn, matplotlib
- requests, tqdm

See top-level `requirements.txt` for an exact list.

---

## ✍️ Report & Attribution

- The LaTeX report is in `This_year/CA2/report/report.tex` and references the generated images.
- Author: Mahdi Aghamohammadi (Student ID: 810102365)

---

## 📚 References

Key papers referenced in the report:
- Kingma & Welling (2013) — Auto-Encoding Variational Bayes
- Higgins et al. (2016) — β-VAE
- Van Den Oord et al. (2017) — VQ-VAE
- Tomczak & Welling (2017) — VampPrior
- Acharya et al. (2019) — SC-VAE

---

## 🆘 Need help?

If you want me to:
- Re-run the full experiments and regenerate figures (GPU required) ✅ — reply “Regenerate figures”
- Improve/expand any section of the report, or run LaTeX compilation and fix issues ✅ — reply what to prioritize

Contact: use the author emails listed in the report files (or provide your preferred contact) for course queries.

---

## License

This repository uses the license included in the top-level file `LICENSE`.


*Generated automatically by the project assistant.*
