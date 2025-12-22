# `utils/` — Shared Utility Functions and Classes

This directory contains a collection of shared utility functions and classes that are used across various assignments within the CAe project. The goal of this module is to centralize common functionalities, promote code reusability, and maintain consistency throughout the codebase.

## Purpose

The `utils/` module provides essential tools and helper functions that support different aspects of the deep learning pipeline, including:

-   **Configuration Management**: Centralized handling of hyperparameters and settings.
-   **Reproducibility**: Functions for setting random seeds to ensure consistent experimental results.
-   **Device Management**: Utilities for detecting and managing computational devices (CPU, CUDA, MPS).
-   **Logging**: Tools for recording training progress, metrics, and other relevant information.
-   **Metrics Calculation**: Common evaluation metrics that can be applied to different tasks.
-   **Plotting and Visualization**: Helper functions for generating various types of plots (e.g., loss curves, confusion matrices).
-   **Checkpointing**: Functions for saving and loading model states during training.

## Structure

The `utils/` directory typically includes the following types of files and functionalities:

-   `config.py`: Defines a centralized `CONFIG` dictionary for all hyperparameters. This ensures that all assignment-specific and global settings are easily accessible and modifiable from a single location.
-   `reproducibility.py`: Contains functions like `seed_everything()` to set random seeds for Python, NumPy, and PyTorch (CPU/CUDA) to ensure deterministic results.
-   `device.py`: Provides functions to automatically select the most suitable device (CUDA > MPS > CPU) and move models/data accordingly.
-   `logger.py`: Implements logging functionalities to record training and evaluation progress, potentially integrating with tools like Weights & Biases (wandb).
-   `metrics.py`: Houses reusable functions for calculating common metrics such as Dice Score, IoU, BLEU, and accuracy.
-   `plotter.py`: Contains helper functions for generating various plots, adhering to specific visualization rules (e.g., DPI, format, labeling).
-   `checkpoint.py`: Includes utilities for saving and loading model states, optimizer states, and training history, facilitating the resumption of training or deployment of best models.

## Key Guidelines

-   **Modularity**: Each utility function or class should be self-contained and perform a specific task.
-   **Generality**: Functions should be designed to be general enough to be used across different assignments without significant modifications.
-   **Type Hinting and Docstrings**: All functions and classes in `utils/` must be fully type-hinted and include comprehensive docstrings explaining their purpose, parameters, and return values.
-   **No Hardcoding**: Avoid hardcoding values; instead, use the `CONFIG` dictionary or pass parameters explicitly.
-   **Rule References**: When applicable, reference the project's `CLAUDE.md` rules in comments or docstrings to ensure compliance with coding standards and best practices.

## Example Usage

```python
# From within an assignment file (e.g., q1_image_captioning/train.py)

import torch
from utils.config import CONFIG
from utils.reproducibility import seed_everything
from utils.device import get_device
from utils.metrics import calculate_dice_score # Assuming a segmentation task

# Set global seed for reproducibility
seed_everything(CONFIG['seed'])

# Get the appropriate device
device = get_device()
print(f"Running on: {device}")

# Example of using a metric function
predictions = torch.randn(10, 2, 256, 256).to(device)
targets = torch.randint(0, 2, (10, 256, 256)).to(device)

dice = calculate_dice_score(predictions, targets, num_classes=CONFIG['num_classes'])
print(f"Calculated Dice Score: {dice:.4f}")
```
