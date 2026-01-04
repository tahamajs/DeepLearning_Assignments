# CA5: Vision Transformers & Large Language Diffusion Models

This project implements two main components:

1. **Image Re-identification using Transformers (BotNet)**: A ResNet50-based model with Multi-Head Self-Attention for person re-identification.

2. **Large Language Diffusion (LLaDA)**: A diffusion-based text generation model for Text-to-SQL tasks.

## Project Structure

```
CA5_Vision_Transformers/
├── data.py              # Data loading, preprocessing, and dataset classes
├── models.py            # Model architectures (ResNet, BotNet, MHSA)
├── utils.py             # Utility functions (visualization, interpretability)
├── train.py             # Training and evaluation functions
├── generate.py          # Generation functions for LLaDA
├── evaluation.py        # Advanced evaluation metrics (CMC, mAP)
├── ablation.py          # Ablation study functions
├── optimization.py      # Model optimization and deployment
├── research.py          # Advanced research demonstrations
├── main.py              # Main script with comprehensive examples
└── requirements.txt     # Dependencies
```

## Installation

```bash
pip install -r requirements.txt
```

## Core Components

### 1. Person Re-identification (Re-ID)

#### Models
- **ResNet50**: Baseline CNN architecture
- **BotNet50**: ResNet50 enhanced with Multi-Head Self-Attention (MHSA)

#### Key Features
- Attention visualization for interpretability
- Comprehensive evaluation with CMC and mAP metrics
- Ablation studies comparing different architectures

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

### Advanced Evaluation

```python
from evaluation import evaluate_reid_advanced

# Comprehensive evaluation with CMC and mAP
cmc_scores, map_score = evaluate_reid_advanced(botnet_model, test_loader, device, "BotNet50")
```

### Interpretability Analysis

```python
from utils import comprehensive_interpretability_analysis

# Full interpretability analysis
sample_image, _ = next(iter(test_loader))
results = comprehensive_interpretability_analysis(botnet_model, sample_image, pil_image, "BotNet50")
```

### Ablation Studies

```python
from ablation import ablation_study_reid, plot_ablation_results

# Compare different model configurations
model_configs = {
    'ResNet_Pretrained': {'model_type': 'resnet_pretrained', 'num_classes': num_classes},
    'BotNet_4Heads': {'model_type': 'botnet_4', 'num_classes': num_classes},
}
results = ablation_study_reid(model_configs, train_loader, test_loader)
plot_ablation_results(results)
```

### LLaDA Generation

```python
from main import *

example = dataset['test'][0]
prompt, _ = format_example(example, tokenizer)
generated = generate_block_diffusion(model, tokenizer, prompt)
print("Generated SQL:", generated)
```

### Model Optimization

```python
from optimization import benchmark_inference_speed, quantize_model_dynamic

# Benchmark inference speed
benchmark_results = benchmark_inference_speed(model, test_loader, device, "BotNet50")

# Quantize model for efficiency
quantized_model = quantize_model_dynamic(model)
```

### Research Demonstrations

```python
from research import demonstrate_clip_reid_integration, research_directions_summary

# Explore advanced research directions
demonstrate_clip_reid_integration()
research_directions_summary()
```

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
- `evaluate_reid_advanced()`: CMC and mAP evaluation
- `train_step()`: Single training step for LLaDA

### Analysis & Visualization
- `visualize_attention()`: Attention map visualization
- `comprehensive_interpretability_analysis()`: Full model interpretability
- `plot_cmc_curve()`: Plot CMC evaluation results

### Advanced Features
- `ablation_study_reid()`: Compare model configurations
- `benchmark_inference_speed()`: Performance benchmarking
- `generate_block_diffusion()`: LLaDA text generation

## Requirements

- Python 3.8+
- PyTorch 2.0+
- CUDA-compatible GPU (recommended)
- 8GB+ VRAM for LLaDA model
- See `requirements.txt` for full dependencies

## Research Contributions

This implementation explores:
- **Attention mechanisms** in CNN architectures
- **Diffusion models** for text generation
- **Model interpretability** techniques
- **Efficient training** methods
- **Advanced evaluation** metrics

The code provides a foundation for research in vision transformers, generative diffusion models, and their applications in computer vision and natural language processing.

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