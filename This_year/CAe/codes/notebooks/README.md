# `notebooks/` — Jupyter Notebooks for Experimentation and Visualization

This directory serves as the **exclusive home for Jupyter notebooks** within the CAe project. These notebooks are designed for demonstrating implementations, conducting experiments, and generating high-quality visualizations.

## Purpose

The primary roles of the notebooks in this directory are:

1.  **Experimentation and Prototyping**: Rapidly test and iterate on new ideas, model architectures, or training strategies.
2.  **Code Execution**: The notebooks are the *only* place where training loops, evaluation routines, and other executable code snippets should be run. All core logic (models, datasets, losses, utilities) should be imported from the `src/` or respective assignment directories (e.g., `q1_image_captioning/`).
3.  **Visualization**: Generate publication-quality figures and plots for analysis and reporting. This includes loss curves, generated samples, decision boundaries, attention maps, and statistical visualizations. All generated plots **must be saved programmatically** (e.g., `plt.savefig('../pictures/fig_01_convergence.png', dpi=300)`).
4.  **Demonstration**: Provide clear, step-by-step demonstrations of how to use the implemented models and utilities.

## Structure and Guidelines

Each notebook should ideally follow a structured workflow:

1.  **Imports & Setup**: Begin with necessary imports from the project's `src/` modules or assignment-specific directories. Set random seeds, detect hardware (CUDA/MPS/CPU), and configure logging.
2.  **Data Loading & Visualization**: Load raw data and perform initial visualizations to understand its characteristics. This may include plotting sample images, audio waveforms, or textual data.
3.  **Model Training**: Import models, loss functions, and optimizers from their respective `src/` files. Implement and execute the training loop *within the notebook*. This includes tracking metrics, applying early stopping, and saving model checkpoints.
4.  **Testing and Evaluation**: Run evaluation on dedicated test sets and compute comprehensive metrics (e.g., Dice, IoU, BLEU, accuracy). Ensure models are in `.eval()` mode and `torch.no_grad()` context is used.
5.  **Results Visualization**: Generate detailed plots for training curves (loss, metrics over epochs), qualitative examples of model predictions, error maps, and statistical summaries of results.

### Key Rules:

-   **No Core Logic in Notebooks**: The notebooks should *import* models, datasets, and loss functions from the `src/` directory (or specific assignment subdirectories). They should *not* contain the core implementations of these components.
-   **Visualization Mandate**: All figures and plots generated in the notebooks **must be saved to the `pictures/` directory** (or a similar designated output folder) at 300 DPI in PNG format.
-   **Reproducibility**: Ensure all notebooks are reproducible by setting random seeds at the beginning.
-   **Clarity and Documentation**: Use Markdown cells to provide clear explanations, section headers, and comments to describe the code and its purpose.

## Example Workflow in a Notebook

```python
# Cell 1: Imports and Setup
import torch
import matplotlib.pyplot as plt
from q1_image_captioning.models import Encoder, DecoderWithAttention
from q1_image_captioning.data import ImageCaptioningDataset, get_data_loaders
from q1_image_captioning.losses import CrossEntropyLoss, CombinedLoss
from utils.metrics import calculate_bleu_score
from utils.config import CONFIG # Assuming a config file exists

seed_everything(CONFIG['seed'])
device = get_device()
print(f"Using device: {device}")

# Cell 2: Data Loading and Preprocessing
# ... load and visualize data ...

# Cell 3: Model Initialization and Training Loop
encoder = Encoder().to(device)
decoder = DecoderWithAttention().to(device)
optimizer = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=CONFIG['lr'])
criterion = CombinedLoss()

# ... training loop implementation ...

# Cell 4: Evaluation and Prediction Visualization
# ... evaluate model on test set ...
# ... plot sample predictions ...

plt.savefig('../pictures/fig_01_sample_predictions.png', dpi=300)

# Cell 5: Training Curves Visualization
# ... plot loss and metric curves ...

plt.savefig('../pictures/fig_02_training_curves.png', dpi=300)
```
