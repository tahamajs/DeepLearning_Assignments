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

## Implementation Details

### Datasets

- **CIFAR-100**: 100 classes, 32x32 images (for ResNet)
- **Flowers-102**: 102 flower classes, higher resolution (for ViT)
- **Preprocessing**:
  - Normalization per dataset
  - Data augmentation for training
  - Adversarial examples generated on-the-fly

### Attack Implementation

#### FGSM

- **Gradient Computation**: ∇_x L(θ, x, y)
- **Perturbation**: ε \* sign(∇_x L)
- **Clipping**: Keep within [0,1] range

#### PGD

- **Initialization**: Random or FGSM start
- **Iteration**: Multiple FGSM steps with projection
- **Projection**: Clip to ε-ball around original

### Adversarial Training

- **DataLoader Modes**: Clean, attack, both
- **Loss**: Standard cross-entropy on adversarial examples
- **Schedule**: Gradually increase attack strength

### Model Architectures

#### ResNet

- **Layers**: 18/34/50 layer variants
- **Pretrained**: Optional ImageNet initialization
- **Fine-tuning**: CIFAR-100 classification

#### ViT

- **Patch Size**: 16x16
- **Embedding**: 768-dim with position encodings
- **Transformer**: 12 layers, 12 heads
- **Pretrained**: Optional on larger datasets

### Training Parameters

- **Batch Size**: 128
- **Learning Rate**: 0.001 (Adam)
- **Epochs**: 50-100
- **Attack ε**: 0.03 (8/255 normalized)
- **PGD Steps**: 10

### Evaluation Metrics

- **Clean Accuracy**: Performance on unperturbed data
- **Adversarial Accuracy**: Robustness under attack
- **Attack Success Rate**: Percentage of successful misclassifications

## Results

### CIFAR-100 (ResNet)

- **Clean Accuracy**: ~75%
- **FGSM Robustness**: ~45% (-30% drop)
- **PGD Robustness**: ~35% (-40% drop)
- **Adversarial Training**: Improves robustness to ~60%

### Flowers-102 (ViT)

- **Clean Accuracy**: ~85%
- **FGSM Robustness**: ~55% (-30% drop)
- **PGD Robustness**: ~45% (-40% drop)
- **Adversarial Training**: Improves robustness to ~70%

### Model Comparison

- **ResNet vs ViT**: Similar vulnerability patterns
- **Pretrained vs Scratch**: Pretraining improves clean accuracy
- **Adversarial Training**: Consistent robustness gains

### Interpretability Insights

- **Grad-CAM**: Attacks shift attention to irrelevant regions
- **ViT Attention**: Adversarial examples disrupt patch relationships
- **Robust Models**: More stable activation patterns

### Training Dynamics

- Adversarial training converges slower
- Validation accuracy more stable with robustness
- Overfitting reduced on adversarial examples

### Challenges Addressed

- **Computational Cost**: Efficient attack generation
- **Hyperparameter Tuning**: ε selection for different datasets
- **Evaluation**: Comprehensive clean vs. adversarial metrics
- **Interpretability**: Understanding attack mechanisms
