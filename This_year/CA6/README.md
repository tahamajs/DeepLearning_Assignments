# CA6: Generative Models (This Year Track)

## Overview

This assignment contains two generative modeling pipelines:

1. `Q1`: GAN-based image generation/adaptation workflow
2. `Q2`: VAE-based representation and reconstruction workflow

The repository already includes training/evaluation scripts, bilingual report sources, and compiled report PDFs.

## Structure

```text
CA6/
├── code/
│   ├── train_gan.py
│   ├── train_vae.py
│   ├── evaluate_fid.py
│   ├── generate_samples.py
│   ├── run_all.py
│   └── requirements.txt
├── description/
│   ├── Assignment6.pdf
│   ├── Q1.pdf
│   ├── Q2.pdf
│   └── en.md
├── report/
│   ├── EN/ (LaTeX source + main.pdf)
│   ├── FA/ (LaTeX source + main.pdf)
│   └── REPORT_COMPLETION.md
└── zip_file/
    └── NNDL_Assignment6.zip
```

## Quick Run

```bash
cd This_year/CA6/code
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python run_all.py
```

## Report Assets

- EN PDF: `This_year/CA6/report/EN/main.pdf`
- FA PDF: `This_year/CA6/report/FA/main.pdf`
- Sample figure used for report completeness:
  - `This_year/CA6/report/EN/images/vae_samples_epoch001.png`
  - `This_year/CA6/report/FA/images/vae_samples_epoch001.png`
