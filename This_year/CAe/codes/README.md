# CAe — Deep Learning & Neural Networks (Course Project)

This folder contains implementation code and notebooks for the course assignment (Image Captioning, CLIP contrastive learning, Urban Sound classification, LoRA fine-tuning, and Adversarial Attacks).

Structure
- `q1_image_captioning/` — data, tokenizer, model (encoder/decoder + attention), training and evaluation scripts
- `q1_clip/` — ViT-based image encoder + text encoder (contrastive) and InfoNCE training
- `q2_urban_sound/` — UrbanSound data wrappers, CNN and Wav2Vec experiments
- `q3_lora/` — LoRA fine-tuning scripts for Llama-based models
- `q4_adversarial/` — attacks (FGSM/PGD), adversarial training, evaluation
- `utils/` — shared utilities (logging, metrics, plotting)
- `scripts/` — dataset downloaders and helper scripts
- `notebooks/` — example notebooks with demos and evaluation plots

Getting started
1. Create a Python environment (recommend conda) and install dependencies:

```bash
python -m pip install -r requirements.txt
```

2. Place datasets in `data/` as instructed in `scripts/download_datasets.py` or run the downloader scripts.
3. See each subfolder `README.md` (TODO) for specific run instructions.

Notes
- Training full models requires GPU resources. The repository provides trainer scripts and sample checkpoints to reproduce results.
- If you prefer I can add example checkpoints and short smoke-run scripts to validate the pipeline on CPU (fast, low-accuracy) as well.
