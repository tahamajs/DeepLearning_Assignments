# CA5: Vision Transformers & Large Language Diffusion Models

This project implements two main components:

1. **Image Re-identification using Transformers (BotNet)**: A ResNet50-based model with Multi-Head Self-Attention for person re-identification.

2. **Large Language Diffusion (LLaDA)**: A diffusion-based text generation model for Text-to-SQL tasks.

## Project Structure

```
CA5_Vision_Transformers/
├── data.py          # Data loading and preprocessing
├── models.py        # Model architectures (ResNet, BotNet, MHSA)
├── utils.py         # Utility functions (visualization, post-processing)
├── train.py         # Training and evaluation functions
├── generate.py      # Generation functions for LLaDA
├── main.py          # Main script with setup and examples
└── requirements.txt # Dependencies
```

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Re-ID Training

```python
from main import *

# Models are initialized in main.py
train_model(resnet_model, resnet_optimizer, train_loader)
evaluate_model(resnet_model, test_loader)
```

### LLaDA Generation

```python
from main import *

example = dataset['test'][0]
prompt, _ = format_example(example, tokenizer)
generated = generate_block_diffusion(model, tokenizer, prompt)
print("Generated SQL:", generated)
```

## Key Components

### BotNet Architecture
- Replaces ResNet's final convolutional stage with Multi-Head Self-Attention
- Captures global spatial dependencies for better Re-ID performance
- Includes attention visualization for interpretability

### LLaDA Implementation
- Forward masking diffusion process for discrete text
- Block-wise iterative generation
- LoRA fine-tuning for efficiency
- SQL normalization and exact match evaluation

## Results

- **Re-ID**: BotNet achieves superior accuracy over baseline ResNet through global attention
- **LLaDA**: Generates coherent SQL queries via diffusion-based denoising

## Authors

[Your Name] - University of Tehran