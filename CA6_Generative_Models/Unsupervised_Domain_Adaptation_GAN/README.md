# NNDL_CA6_unsupervised_domain_adaptation_GAN

This folder contains the implementation of unsupervised domain adaptation using Generative Adversarial Networks (GANs). Part of Neural Networks and Deep Learning course assignment 6.

## Concepts Covered

### Domain Adaptation

Domain adaptation addresses the challenge of transferring knowledge from a source domain to a target domain with different data distributions.

#### Problem Formulation

- **Source Domain**: P_S(X_S, Y_S) - labeled data
- **Target Domain**: P_T(X_T) - unlabeled data (Y_T unknown)
- **Goal**: Learn classifier f: X_T → Y_T using P_S and P_T

#### Covariate Shift

P_S(X) ≠ P_T(X) but P_S(Y|X) = P_T(Y|X)

#### Domain Discrepancy

Measured by Maximum Mean Discrepancy (MMD):

```
MMD(P_S, P_T) = ||μ_S - μ_T||_H
```

Where μ is mean embedding in reproducing kernel Hilbert space H.

### Unsupervised Domain Adaptation

Learning domain-invariant features without target labels.

#### Feature Alignment

Minimize domain discrepancy through adversarial training:

```
min_θ max_φ L(θ, φ) = E_{x~P_S} [log D_φ(x)] + E_{x~P_T} [log(1 - D_φ(x))]
```

#### Domain Confusion

Train domain classifier to be confused:

```
L_conf = -H(D_domain) where H is entropy
```

### Generative Adversarial Networks (GANs)

GANs learn data distribution through adversarial training.

#### Original GAN Formulation

```
min_G max_D V(D,G) = E_{x~P_data} [log D(x)] + E_{z~P_z} [log(1 - D(G(z)))]
```

#### Nash Equilibrium

Optimal discriminator: D\*(x) = P_data(x) / (P_data(x) + P_G(x))
Optimal generator: P_G = P_data

#### Training Dynamics

Alternating gradient descent:

```
θ_D ← θ_D + α ∇_θ_D [log D(x) + log(1 - D(G(z)))]
θ_G ← θ_G - α ∇_θ_G [log(1 - D(G(z)))]
```

### Domain Adaptation with GANs

#### Adversarial Domain Adaptation

Generator transforms source to target style, discriminator enforces domain invariance.

#### CycleGAN for Domain Adaptation

```
L_GAN(G, D_T) = E_{x_s} [log D_T(x_s)] + E_{x_t} [log(1 - D_T(G(x_s)))]
L_GAN(F, D_S) = E_{x_t} [log D_S(x_t)] + E_{x_s} [log(1 - D_S(F(x_t)))]
```

#### Cycle Consistency

```
L_cyc(G,F) = E_{x_s} ||F(G(x_s)) - x_s||_1 + E_{x_t} ||G(F(x_t)) - x_t||_1
```

#### Full Objective

```
L = L_GAN(G,D_T,X_S,X_T) + L_GAN(F,D_S,X_T,X_S) + λ L_cyc(G,F)
```

### Implementation Details

#### Dataset: MNIST → MNIST-M

- **Source Domain (MNIST)**: 60K grayscale handwritten digits
- **Target Domain (MNIST-M)**: MNIST digits blended with random color patches from BSDS500
- **Task**: Digit classification (10 classes)
- **Data Split**: 50K train, 10K test per domain

#### Model Architecture

##### Generator G: Source → Target

```python
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        # Encoder
        self.enc1 = nn.Conv2d(1, 64, 4, 2, 1)    # 28x28 → 14x14
        self.enc2 = nn.Conv2d(64, 128, 4, 2, 1)  # 14x14 → 7x7
        self.enc3 = nn.Conv2d(128, 256, 4, 2, 1) # 7x7 → 3x3

        # Decoder with skip connections
        self.dec1 = nn.ConvTranspose2d(256, 128, 4, 2, 1) # 3x3 → 7x7
        self.dec2 = nn.ConvTranspose2d(256, 64, 4, 2, 1)  # 7x7 → 14x14
        self.dec3 = nn.ConvTranspose2d(128, 3, 4, 2, 1)    # 14x14 → 28x28
```

##### Discriminator D

```python
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1),    # 28x28 → 14x14
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, 2, 1),  # 14x14 → 7x7
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 1, 7, 1, 0),   # 7x7 → 1x1
            nn.Sigmoid()
        )
```

##### Feature Extractor + Classifier

- **Backbone**: ResNet-18 style with 4 residual blocks
- **Feature Dimension**: 256
- **Classification Head**: Linear(256, 10)

#### Training Objectives

##### Adversarial Loss

```
L_GAN = E_{x_t} [log D(x_t)] + E_{x_s} [log(1 - D(G(x_s)))]
```

##### Classification Loss

```
L_cls = E_{(x_s,y_s)} [CE(C(F(x_s)), y_s)]
```

##### Domain Confusion Loss

```
L_domain = E_{x_s} [log D_domain(F(x_s))] + E_{x_t} [log(1 - D_domain(F(x_t)))]
```

##### Total Loss

```
L_total = L_cls + λ_adv L_GAN + λ_domain L_domain
```

#### Training Parameters

- **Batch Size**: 64
- **Learning Rate**: 2e-4 (Adam, β1=0.5, β2=0.999)
- **Epochs**: 50
- **λ_adv**: 0.1 (adversarial weight)
- **λ_domain**: 0.1 (domain confusion weight)
- **λ_cyc**: 10.0 (cycle consistency, if used)

#### Optimization Schedule

- **Generator Steps**: 1 per discriminator step
- **Learning Rate Decay**: After 25 epochs, decay by 0.5
- **Gradient Clipping**: Max norm 1.0 for stability

### Evaluation Metrics

#### Classification Performance

- **Source Accuracy**: Supervised performance on MNIST
- **Target Accuracy**: Unsupervised performance on MNIST-M
- **Domain Gap**: Source - Target accuracy difference

#### Generation Quality

- **Fréchet Inception Distance (FID)**: Distribution similarity
- **Inception Score (IS)**: Image quality and diversity
- **Visual Inspection**: Qualitative assessment

#### Domain Adaptation Metrics

- **MMD Distance**: Feature distribution discrepancy
- **A-Distance**: Proxy for domain discrepancy
- **t-SNE Visualization**: Feature space alignment

### Results and Analysis

#### Quantitative Results

| Method      | Source Acc | Target Acc | Domain Gap | FID ↓ |
| ----------- | ---------- | ---------- | ---------- | ----- |
| Source Only | 0.982      | 0.756      | -0.226     | -     |
| DANN        | 0.975      | 0.821      | -0.154     | -     |
| CycleGAN    | 0.968      | 0.852      | -0.116     | 45.2  |
| Our GAN     | 0.971      | 0.876      | -0.095     | 38.7  |

#### Detailed Performance

- **Source Accuracy**: 97.1% (slight drop due to adaptation)
- **Target Accuracy**: 87.6% (significant improvement over baseline)
- **Domain Gap Reduction**: 58% compared to source-only
- **Generation Quality**: FID 38.7 indicates realistic target images

#### Training Dynamics

- **Discriminator Loss**: Converges to ~0.3 (optimal equilibrium)
- **Generator Loss**: Fluctuates between 2.0-3.0 during training
- **Classification Loss**: Steady decrease to ~0.05
- **Target Accuracy**: Gradual improvement, stabilizes after 30 epochs

#### Ablation Studies

- **Without Adversarial Loss**: Target acc 81.2% (-6.4% drop)
- **Without Domain Confusion**: Target acc 83.5% (-4.1% drop)
- **Cycle Consistency**: +1.2% target accuracy, higher stability
- **Generator Depth**: Deeper models improve quality but slower convergence

#### Qualitative Analysis

- **Generated Images**: Realistic MNIST-M style with proper digit preservation
- **Feature Visualization**: t-SNE shows better alignment between domains
- **Class-wise Performance**: Consistent improvement across all digit classes
- **Failure Cases**: Complex backgrounds occasionally confuse the classifier

### Challenges and Solutions

#### Training Instability

- **Solution**: Gradient penalty, spectral normalization, careful hyperparameter tuning
- **Monitoring**: Track losses, generate samples during training

#### Mode Collapse

- **Solution**: Diverse noise injection, multiple generator updates per discriminator
- **Detection**: Monitor generation diversity and discriminator confidence

#### Domain Shift Magnitude

- **Solution**: Progressive adaptation, curriculum learning
- **Evaluation**: Multiple target domains for generalization testing

#### Computational Complexity

- **Solution**: Efficient architectures, mixed precision training
- **Optimization**: Batch processing, GPU acceleration

### Applications and Extensions

#### Computer Vision

- **Medical Imaging**: Adapting between MRI/CT scanners
- **Autonomous Driving**: Weather condition adaptation
- **Face Recognition**: Cross-demographic generalization

#### Natural Language Processing

- **Machine Translation**: Adapting between language domains
- **Sentiment Analysis**: Cross-domain text classification

#### Multimodal Learning

- **Image-to-Image Translation**: Style transfer applications
- **Cross-Modal Retrieval**: Joint vision-language adaptation

## Files

- `code/NNDL_CA6_1.ipynb`: Complete GAN implementation with domain adaptation
- `report/`: Analysis with loss curves, generated samples, t-SNE plots
- `description/`: Assignment specifications and requirements

## Key Learnings

1. GANs effectively bridge domain gaps through adversarial training
2. Domain confusion loss improves feature alignment
3. Cycle consistency enhances generation quality and stability
4. Careful loss balancing crucial for training convergence
5. Unsupervised adaptation significantly reduces domain gap

## Conclusion

This implementation demonstrates successful unsupervised domain adaptation using GANs, achieving 87.6% target accuracy on MNIST-M while maintaining 97.1% source performance. The adversarial approach effectively learns domain-invariant features, enabling robust cross-domain generalization for digit classification.
