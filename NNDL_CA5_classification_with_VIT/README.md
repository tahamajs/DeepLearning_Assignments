# NNDL_CA5_classification_with_VIT

This folder contains the implementation of image classification using Vision Transformer (ViT). Part of Neural Networks and Deep Learning course assignment 5.

## Concepts Covered

### Vision Transformer (ViT)

Transforms images into sequences of patches for transformer-based processing.

Key components:

- **Patch Embedding**: Divides image into fixed-size patches
- **Position Encoding**: Adds positional information
- **Transformer Encoder**: Self-attention layers process patch sequences
- **Classification Head**: MLP for final prediction

### Self-Attention Mechanism

- Computes attention weights between all patches
- Captures global dependencies
- Enables modeling long-range interactions

### Training ViT

- Requires large datasets (pretrained on ImageNet)
- Data augmentation crucial
- Fine-tuning for downstream tasks

### Comparison to CNNs

- ViT: Global receptive field, better scalability
- CNNs: Inductive biases for local patterns, data efficiency
- ViT excels on large data, CNNs on small data

### Evaluation

- Top-1/Top-5 accuracy
- Computational efficiency
- Interpretability via attention maps

### Applications

- Image classification
- Object detection (DETR)
- Segmentation tasks

## Files

- `code/`: ViT implementation
- Dataset: ImageNet or CIFAR

## Results

ViT achieves state-of-the-art performance on image classification, especially with sufficient data.
