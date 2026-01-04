# CA5: Vision Transformers & Large Language Diffusion Models

This directory contains the extracted Python modules from the CA5 notebook, allowing you to run the code independently without the Jupyter notebook environment.

## Project Structure

```
notebook/
├── data.py              # Data loading, preprocessing, and dataset classes
├── models.py            # Model architectures (ResNet, BotNet, MHSA)
├── utils.py             # Utility functions (visualization, post-processing)
├── train.py             # Training and evaluation functions
├── generate.py          # Generation functions for LLaDA
├── main.py              # Main script with complete setup and examples
└── requirements.txt     # Dependencies
```

## Installation

```bash
pip install -r requirements.txt
```

## Validation

Before running the code, you can validate the structure:

```bash
python validate_structure.py
```

This will check that all files are present and have correct syntax.

## Core Components

### 1. Person Re-identification (Re-ID)

#### Models
- **ResNet50**: Baseline CNN architecture
- **BotNet50**: ResNet50 enhanced with Multi-Head Self-Attention (MHSA)

#### Key Features
- Attention visualization for interpretability
- Comprehensive evaluation with standard Re-ID metrics

### 2. Large Language Diffusion (LLaDA)

#### Features
- Forward diffusion masking process
- LoRA fine-tuning on quantized LLaMA model
- Block diffusion sampling for generation
- Exact match scoring for SQL evaluation

## Usage Examples

### Basic Re-ID Training

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

### Running Everything

```bash
python main.py
```

This will:
1. Set up the Re-ID datasets and models
2. Train both ResNet50 and BotNet50 models
3. Evaluate their performance
4. Demonstrate LLaDA SQL generation

## Key Functions

### Data Processing
- `download_market1501()`: Download Market-1501 dataset
- `ReIDDataset`: Custom dataset class for Re-ID
- `format_example()`: Prepare examples for LLaDA training

### Model Architectures
- `get_resnet50()`: Create ResNet50 model
- `MHSA`: Multi-Head Self-Attention module
- `BotNet50`: Bottleneck Transformer network

### Training & Evaluation
- `train_model()`: Basic training loop
- `evaluate_model()`: Basic accuracy evaluation
- `train_step()`: Single training step for LLaDA
- `evaluate_pipeline()`: LLaDA evaluation pipeline

### Generation
- `generate_block_diffusion()`: LLaDA text generation

### Utilities
- `visualize_attention()`: Attention map visualization
- `post_process_sql()`: SQL post-processing

## Requirements

- Python 3.8+
- PyTorch 2.0+
- CUDA-compatible GPU (recommended)
- 8GB+ VRAM for LLaDA model
- See `requirements.txt` for full dependencies

## Data Setup

### Re-ID Data
The code uses synthetic data by default. For real data:
1. Download Market-1501 dataset
2. Extract to `./data/Market-1501/`
3. Update data paths in `main.py`

### Text-to-SQL Data
Automatically downloaded via Hugging Face Datasets.

## Expected Results

### Re-ID Performance
- **ResNet50**: ~70-80% accuracy
- **BotNet50**: ~75-85% accuracy (improvement from attention)

### LLaDA Performance
- **Exact Match**: ~20-40% accuracy on synthetic Text-to-SQL
- **Generation Quality**: Reasonable SQL queries for simple cases

## Notes

- The code is extracted directly from the notebook cells
- All functions maintain the same interfaces and behavior
- CUDA device handling is included for GPU acceleration
- Memory-efficient implementations for large models