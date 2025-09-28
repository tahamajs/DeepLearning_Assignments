# CA5: Vision Transformers

This assignment explores Vision Transformers (ViT) and multimodal learning, implementing image classification with ViT and adversarial attacks on CLIP, showcasing the latest advances in transformer-based computer vision.

## Overview

The assignment consists of two transformer-focused projects:

1. **Vision Transformer Classification**: ViT implementation for image classification
2. **CLIP Adversarial Attacks**: Adversarial robustness analysis of multimodal models

## Contents

- `VIT_Classification/`: Vision Transformer for image classification
- `CLIP_Adversarial_Attack/`: Adversarial attacks on CLIP model

Each subfolder contains:
- `code/`: PyTorch implementations
- `description/`: Assignment specifications
- `paper/`: Research papers and references
- `report/`: Detailed analysis and results

## Vision Transformer Classification

### Key Features
- **Patch Embedding**: Image divided into fixed-size patches (16×16)
- **Self-Attention**: Multi-head attention for global context modeling
- **Position Encoding**: Learnable positional embeddings
- **Class Token**: Special token for classification

### Technical Details
- **Patch Size**: 16×16 pixels → 768-dim embeddings
- **Transformer Blocks**: 12 layers, 12 attention heads, 768-dim model
- **Pre-training**: Optional initialization with ImageNet-pretrained weights
- **Fine-tuning**: End-to-end training on target datasets

### Results
- **Accuracy**: 88.2% on CIFAR-10 (comparable to ResNet-50)
- **Computational Cost**: Higher training cost but better scaling
- **Attention Patterns**: Global receptive field captures long-range dependencies
- **Data Efficiency**: Benefits from larger datasets more than CNNs

## CLIP Adversarial Attacks

### Key Features
- **Multimodal Attacks**: Perturbing images while preserving semantic meaning
- **Defense Strategies**: LoRA fine-tuning, TeCoA loss, Visual Prompt Tuning
- **Robust Evaluation**: Comprehensive clean vs. adversarial performance analysis
- **Parameter Efficiency**: Low-rank adaptation for practical deployment

### Technical Details
- **CLIP Architecture**: Vision Transformer + Text Transformer
- **Attack Methods**: FGSM, PGD with ε-constraints
- **Defense Techniques**: Test-time classifier alignment, prompt tuning
- **Evaluation**: Robustness metrics across multiple attack strengths

### Results
- **Clean Accuracy**: 65.2% zero-shot performance
- **Adversarial Drop**: 20.1% accuracy loss under attack
- **Defense Improvement**: TeCoA achieves 62.1% robust accuracy
- **Parameter Efficiency**: LoRA uses only 0.8M trainable parameters

## Key Concepts Demonstrated

### Vision Transformers
- **Patch-based Processing**: Treating images as sequences of patches
- **Self-Attention Mechanism**: Computing attention over all patches
- **Position Encoding**: Injecting spatial information into sequences
- **Scalability**: Performance improvement with larger datasets/models

### Multimodal Learning
- **Contrastive Learning**: Aligning vision and text representations
- **Zero-shot Classification**: Classifying without specific training examples
- **Cross-modal Retrieval**: Finding relevant text for images and vice versa
- **Joint Embeddings**: Unified representation space for multiple modalities

### Adversarial Attacks
- **Gradient-based Attacks**: FGSM and PGD optimization methods
- **Transferability**: Attacks working across different architectures
- **Semantic Preservation**: Maintaining image semantics under perturbation
- **Robustness Evaluation**: Measuring model vulnerability to adversarial inputs

### Defense Mechanisms
- **Adversarial Training**: Training with adversarial examples
- **Input Preprocessing**: Defenses applied before model inference
- **Parameter-efficient Fine-tuning**: LoRA and prompt tuning
- **Test-time Adaptation**: Adapting models to specific test distributions

## Educational Value

This assignment provides deep understanding of:
- **Transformer Architectures**: Self-attention and modern deep learning paradigms
- **Multimodal AI**: Combining vision and language modalities
- **Adversarial ML**: Security and robustness in deep learning systems
- **Efficient Fine-tuning**: Parameter-efficient adaptation techniques

## Dependencies

- Python 3.8+
- PyTorch 1.9+
- torchvision
- transformers
- OpenCLIP
- NumPy, Pandas
- Matplotlib, Seaborn
- torchattacks (for adversarial attacks)

## Usage

### ViT Classification
1. Navigate to `VIT_Classification/code/`
2. Run data preprocessing and patch embedding
3. Train Vision Transformer on classification task
4. Analyze attention patterns in `VIT_Classification/report/`

### CLIP Adversarial Attacks
1. Navigate to `CLIP_Adversarial_Attack/code/`
2. Execute adversarial attack generation
3. Implement and test defense mechanisms
4. Evaluate robustness improvements in `CLIP_Adversarial_Attack/report/`

## References

- [ViT Classification Description](VIT_Classification/description/)
- [CLIP Attacks Description](CLIP_Adversarial_Attack/description/)
- [Research Papers](VIT_Classification/paper/) | [Research Papers](CLIP_Adversarial_Attack/paper/)
- [Implementation Reports](VIT_Classification/report/) | [Implementation Reports](CLIP_Adversarial_Attack/report/)

---

**Course**: Neural Networks and Deep Learning (CA5)
**Institution**: University of Tehran
**Date**: September 2025