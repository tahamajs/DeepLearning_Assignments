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

### Applications

- Understanding model vulnerabilities
- Improving robustness
- Security in vision-language systems

## Files

- `code/`: Attack implementations on CLIP
- CLIP model and datasets

## Results

CLIP shows varying robustness to adversarial attacks, highlighting the need for defenses in multimodal models.
