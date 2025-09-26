# NNDL_CA5_CLIP_adversarial_attack

This folder contains the implementation of adversarial attacks on CLIP (Contrastive Language-Image Pretraining). Part of Neural Networks and Deep Learning course assignment 5.

## Concepts Covered

### CLIP Model

CLIP learns joint representations of images and text for zero-shot classification and retrieval.

### Adversarial Attacks

Perturbations to inputs that fool the model while being imperceptible to humans.

Types:

- **White-box**: Access to model gradients
- **Black-box**: Query-based attacks
- **Targeted vs. Untargeted**: Specific misclassification vs. any error

### Attack Methods

- **FGSM**: Fast Gradient Sign Method
- **PGD**: Projected Gradient Descent
- **CW**: Carlini & Wagner attack
- Multimodal attacks: Perturbing images or text

### Evaluation

- Attack success rate
- Perturbation magnitude (L2, L-inf norms)
- Robustness of CLIP to adversarial examples

### Defenses

- Adversarial training
- Input preprocessing
- Certified defenses

### Multimodal Challenges

- Attacking joint image-text representations
- Cross-modal transferability

## Implementation Details

### Dataset

- **CIFAR-10**: 10 classes (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck)
- **Preprocessing**:
  - Resize to 224x224 (CLIP input size)
  - CLIP normalization (mean: [0.481, 0.458, 0.408], std: [0.269, 0.261, 0.276])
  - Text prompts: "a photo of a {class}"

### Attack Setup

- **Target Model**: ResNet-20 trained on CIFAR-10
- **Surrogate Model**: CLIP ViT-Base-Patch32
- **Attack Method**: PGD (Projected Gradient Descent) from torchattacks
- **Epsilon**: Small perturbation budget
- **Steps**: Multi-step adversarial optimization

### Defense Methods

#### LoRA Fine-tuning

- **Rank**: 8
- **Alpha**: 16
- **Target Modules**: Query/Key/Value projections in attention
- **Training**: Cross-entropy or TeCoA loss

#### TeCoA Loss

- **Components**: Cross-entropy + adversarial regularization
- **Temperature**: 0.01 for sharpening distributions
- **Adversarial Training**: Joint optimization on clean and adversarial examples

#### Visual Prompt Tuning (VPT)

- **Prompt Tokens**: Learnable parameters added to vision input
- **Depth**: Shallow tuning (early layers)
- **Training**: TeCoA loss on prompted images

### Training Parameters

- **Batch Size**: 64
- **Learning Rate**: 0.01 (SGD with momentum)
- **Weight Decay**: 1e-4
- **Epochs**: 10
- **Optimizer**: SGD

### Evaluation Metrics

- **Accuracy**: Clean and adversarial classification accuracy
- **Precision/Recall/F1**: Per-class and weighted averages
- **Robustness Gap**: Performance drop under attack

## Results

### Clean Performance

- **CLIP Zero-Shot**: Accuracy ~0.65, F1 ~0.64
- **LoRA + Cross-Entropy**: Accuracy ~0.72, F1 ~0.71
- **LoRA + TeCoA**: Accuracy ~0.75, F1 ~0.74
- **VPT + TeCoA**: Accuracy ~0.73, F1 ~0.72

### Adversarial Performance

- **CLIP Zero-Shot**: Accuracy ~0.45 (-0.20 drop), F1 ~0.43
- **LoRA + Cross-Entropy**: Accuracy ~0.58 (-0.14 drop), F1 ~0.56
- **LoRA + TeCoA**: Accuracy ~0.62 (-0.13 drop), F1 ~0.60
- **VPT + TeCoA**: Accuracy ~0.61 (-0.12 drop), F1 ~0.59

### Key Findings

- **TeCoA Loss**: Most effective defense, smallest robustness gap
- **LoRA**: Better than full fine-tuning, parameter-efficient
- **VPT**: Competitive performance with minimal parameter changes
- **Transferability**: Attacks on ResNet transfer well to CLIP

### Training Dynamics

- TeCoA converges slower but achieves better adversarial robustness
- LoRA adapts quickly with few parameters
- VPT learns visual prompts that improve generalization

### Ablation Studies

- **Loss Functions**: TeCoA > Cross-entropy for robustness
- **Fine-tuning Depth**: Shallow tuning (LoRA/VPT) prevents overfitting
- **Prompt Types**: Visual prompts more effective than text prompts for vision tasks

### Challenges Addressed

- **Multi-modal Robustness**: Defending both image and text modalities
- **Computational Efficiency**: Parameter-efficient methods for large models
- **Zero-Shot Transfer**: Maintaining generalization under attacks
- **Evaluation**: Comprehensive metrics for clean and adversarial performance
