# CAe — Deep Learning & Neural Networks (Course Project)

This folder contains implementation code and notebooks for the course assignment, covering advanced topics in Deep Learning and Neural Networks. The assignments focus on novel syntheses of state-of-the-art research papers, aiming to develop publication-worthy projects.

## Structure

This repository is organized into distinct sub-assignments, each focusing on a specific advanced topic. The structure is designed to promote modularity, reproducibility, and clarity in complex research projects.

-   `q1_image_captioning/` — Implements an image captioning system, typically involving an encoder-decoder architecture with attention mechanisms. This assignment delves into combining computer vision and natural language processing techniques.
-   `q1_clip/` — Explores Contrastive Language-Image Pre-training (CLIP) using Vision Transformer (ViT) based image encoders and text encoders, trained with InfoNCE loss. This section focuses on multimodal learning and representation alignment.
-   `q2_urban_sound/` — Addresses Urban Sound classification using various approaches, including CNNs and Wav2Vec experiments. This assignment focuses on audio processing and classification.
-   `q3_lora/` — Contains scripts for Low-Rank Adaptation (LoRA) fine-tuning of large language models, specifically Llama-based architectures. This section explores efficient fine-tuning techniques for pre-trained models.
-   `q4_adversarial/` — Investigates adversarial attacks (e.g., FGSM, PGD), adversarial training, and evaluation methods in deep learning models. This assignment focuses on robustness and security in AI.
-   `utils/` — A collection of shared utility functions and classes used across multiple assignments, including logging, metrics computation, and plotting functionalities.
-   `scripts/` — Contains helper scripts for tasks such as dataset downloading, environment setup, and other automation needs.
-   `notebooks/` — Houses Jupyter notebooks for demonstrating implementations, running experiments, and generating visualizations. These notebooks are the exclusive location for code execution and visualization.

## Getting Started

To set up the environment and get started with the projects, follow these steps:

1.  **Create a Python environment**: It is recommended to use `conda` or `venv` to create a dedicated Python environment.

2.  **Install dependencies**: Navigate to the `codes` directory and install the required Python packages using `pip`:

    ```bash
    python -m pip install -r requirements.txt
    ```

    The `requirements.txt` file lists all necessary Python libraries and their versions to ensure a consistent development environment. It includes:
    -   `torch`, `torchvision`, `torchaudio`: Core PyTorch libraries for deep learning.
    -   `timm`: PyTorch Image Models, providing a collection of pre-trained models.
    -   `transformers`, `datasets`, `peft`, `bitsandbytes`, `accelerate`: Libraries for working with Hugging Face models, datasets, parameter-efficient fine-tuning, and optimized training.
    -   `sacrebleu`, `nltk`: Libraries for natural language processing tasks, particularly for evaluation metrics like BLEU score.
    -   `pandas`, `numpy`, `Pillow`, `scikit-learn`: Essential libraries for data manipulation, numerical operations, image processing, and machine learning utilities.
    -   `matplotlib`, `seaborn`: Libraries for creating high-quality visualizations.
    -   `tqdm`: A fast, extensible progress bar for loops.
    -   `librosa`, `soundfile`: Libraries for audio analysis and manipulation.
    -   `wandb`: Weights & Biases for experiment tracking and visualization.

3.  **Download datasets**: Datasets are typically managed through scripts located in the `scripts/` directory. For example, you might run specific downloaders as instructed in `scripts/download_datasets.py`.

4.  **Run Smoke Tests**: To quickly verify that the environment is set up correctly and core modules can be imported, you can execute the `run_all.sh` script. This script runs a series of basic Python commands to check module imports for each sub-assignment.

    ```bash
    bash run_all.sh
    ```

    The `run_all.sh` script currently performs the following checks:
    -   Attempts to download UrbanSound and Flickr datasets (with `|| true` to prevent script failure if downloaders are not fully set up).
    -   Verifies the import of `tokenizer` from `q1_image_captioning`.
    -   Verifies the import of `models` from `q1_clip`.
    -   Verifies the import of `data` from `q2_urban_sound`.
    -   Verifies the import of `train` from `q3_lora`.
    -   Verifies the import of `attacks` from `q4_adversarial`.
    
    A successful run will output "Smoke run completed".

5.  **Refer to subfolder READMEs**: Each subfolder (`q1_image_captioning/`, `q1_clip/`, etc.) will contain its own detailed `README.md` with specific instructions on running and evaluating that particular assignment.

## .gitignore

The `.gitignore` file specifies intentionally untracked files that Git should ignore. This ensures a clean repository and avoids committing large or temporary files. Key ignored patterns include:

-   **Python-specific files**: `__pycache__/`, `*.py[cod]`, `*.so`, `*.egg-info/`, `.env`, `venv/`, `.env*`.
-   **Jupyter Notebook checkpoints**: `.ipynb_checkpoints/`.
-   **Model checkpoints and saved models**: `checkpoints/`, `*.ckpt`, `*.pt`, `*.pth`.
-   **Data directories**: `data/`. This prevents large datasets from being committed to the repository.
-   **Experiment logs**: `wandb/`, `logs/`.
-   **macOS specific files**: `.DS_Store`.

## Notes

-   Training full models and achieving state-of-the-art results typically requires significant GPU resources. The repository provides comprehensive trainer scripts, and in some cases, sample checkpoints may be provided to reproduce results.
-   If needed, "smoke-run" scripts (fast, low-accuracy CPU runs) can be added to validate pipeline functionality without extensive resources.
