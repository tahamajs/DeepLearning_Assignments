# NNDL_CAe_CNN_VIT_adversarial_attack

This folder contains the implementation of adversarial attacks on CNN and Vision Transformer (ViT) models. Part of Neural Networks and Deep Learning course extra assignment.

## Concepts Covered

### Adversarial Attacks

Small perturbations that mislead neural networks.

### CNN vs. ViT Vulnerabilities

- **CNNs**: Local receptive fields, potentially more robust to global perturbations
- **ViTs**: Global attention, may be sensitive to patch-level attacks

### Attack Methods

- **Gradient-based**: FGSM, PGD, CW
- **Patch attacks**: Perturbing specific image regions
- **Transfer attacks**: Attacks crafted on one model, tested on another

### Evaluation

- Attack success rate
- Robustness comparison between architectures
- Transferability across models

### Defenses

- Adversarial training
- Input transformations
- Certified robustness

### Multimodal Aspects

- Attacks on vision models
- Comparison of inductive biases

### Applications

- Model security assessment
- Robust model development
- Understanding architectural differences

## Files

- `code/`: Attack implementations
- CNN and ViT models

## Results

ViT may show different vulnerability patterns compared to CNNs, informing architecture choices for robustness.
