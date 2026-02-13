# CA6 Code Package

This folder contains executable scripts for the CA6 generative-model assignment.

## Entry Points

- `train_gan.py`: train GAN pipeline for Q1
- `train_vae.py`: train VAE pipeline for Q2
- `evaluate_fid.py`: compute/report image-quality score for generated samples
- `generate_samples.py`: export generated outputs for qualitative analysis
- `train_classifier.py`: downstream classifier support for evaluation
- `run_all.py`: convenience orchestration script

## Internal Package

- `deepgen/`: shared data loaders, model definitions, metrics, and utility helpers

## Outputs

- Training outputs are written under `runs/`
- Report-linked sample output currently available at:
  - `runs/vae/samples_epoch001.png`

## Environment

Install dependencies with:

```bash
pip install -r requirements.txt
```
