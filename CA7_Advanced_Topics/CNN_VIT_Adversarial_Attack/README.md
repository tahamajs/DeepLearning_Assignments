# NNDL_CAe_CNN_VIT_adversarial_attack

This folder contains the implementation of adversarial attacks on Convolutional Neural Networks (CNNs) and Vision Transformers (ViTs). Part of Neural Networks and Deep Learning course extra assignment.

## Concepts Covered

### Adversarial Attacks

Adversarial attacks craft imperceptible perturbations that cause misclassification.

#### Threat Model

- **White-box**: Full access to model and gradients
- **Black-box**: Limited queries, no gradients
- **Targeted**: Force specific incorrect prediction
- **Untargeted**: Any incorrect prediction suffices

#### Attack Goals

```
x' = x + δ, ||δ||_∞ ≤ ε
f(x') ≠ f(x), but ||δ|| imperceptible
```

### CNN vs. ViT Architectures

#### Convolutional Neural Networks (CNNs)

- **Local Receptive Fields**: Hierarchical feature extraction
- **Translation Invariance**: Built-in robustness to small shifts
- **Inductive Bias**: Assumes locality and spatial hierarchies

##### ResNet Architecture

```
Input → Conv(7×7, 64) → MaxPool → Residual Blocks → GlobalAvgPool → FC
Residual Block: x → F(x) + x (skip connection)
```

#### Vision Transformers (ViTs)

- **Global Attention**: Self-attention across all patches
- **Patch-based Processing**: Image divided into fixed-size patches
- **Position Encoding**: Adds spatial information to patch embeddings

##### ViT Architecture

```
Image → Patches → Linear Projection → [CLS] + Position Embeddings → Transformer Blocks → Classification Head
Transformer Block: MSA(LN(x)) + MLP(LN(x)) + x
```

### Attack Methods

#### Fast Gradient Sign Method (FGSM)

Single-step gradient-based attack:

```
δ = ε × sign(∇_x L(f(x), y))
x' = x + δ
```

#### Projected Gradient Descent (PGD)

Iterative attack with projection:

```
x⁰ = x + random_noise
for t = 1 to T:
    x^{t} = Π_{B_∞(x,ε)} (x^{t-1} + α × sign(∇_x L(f(x^{t-1}), y)))
```

#### Carlini & Wagner (CW) Attack

Optimization-based attack:

```
minimize ||δ||_p + c × f(x + δ)
subject to f(x + δ) ≠ y
```

#### Patch Attacks

Perturb specific image regions:

```
δ_patch = ε × sign(∇_{patch} L(f(x), y))
x' = x + δ_patch (only in selected regions)
```

### CNN vs. ViT Vulnerabilities

#### CNN Vulnerabilities

- **Local Perturbations**: Effective in receptive fields
- **High-frequency Patterns**: Exploit texture sensitivity
- **Boundary Effects**: Attacks near decision boundaries

#### ViT Vulnerabilities

- **Patch-level Attacks**: Disrupt patch relationships
- **Attention Manipulation**: Alter self-attention patterns
- **Position Encoding**: Sensitive to spatial perturbations

#### Comparative Analysis

- **Similar Robustness**: Both architectures show comparable vulnerability
- **Attack Transferability**: Attacks transfer between CNNs and ViTs
- **Defensive Strategies**: Similar defense effectiveness

### Defense Mechanisms

#### Adversarial Training

Train on adversarial examples:

```
L_adv = L_clean + λ L_adversarial
where L_adversarial uses adversarial examples
```

#### Input Transformations

- **Random Resizing**: JPEG compression, random cropping
- **Gaussian Smoothing**: Low-pass filtering
- **Random Padding**: Add random borders

#### Certified Defenses

- **Randomized Smoothing**: Add noise for certification
- **Interval Bound Propagation**: Compute certified bounds

### Implementation Details

#### Datasets

##### CIFAR-100 (CNN-focused)

- **Classes**: 100 fine-grained object categories
- **Image Size**: 32×32×3
- **Training**: 50K images, 10 superclasses
- **Preprocessing**:
  - Random crop (32×32 from 40×40 padded)
  - Random horizontal flip
  - Normalization: mean [0.507, 0.487, 0.441], std [0.267, 0.256, 0.276]

##### Flowers-102 (ViT-focused)

- **Classes**: 102 flower species
- **Image Size**: 224×224×3 (resized)
- **Training**: ~6K images
- **Preprocessing**:
  - Random crop (224×224 from larger images)
  - Random horizontal flip
  - Color jittering
  - Normalization: ImageNet statistics

#### Model Architectures

##### ResNet-50

```python
class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels*4, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels*4)
        self.relu = nn.ReLU(inplace=True)

        self.downsample = None
        if stride != 1 or in_channels != out_channels*4:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels*4, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels*4)
            )

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        out += identity
        out = self.relu(out)
        return out
```

##### Vision Transformer (ViT-Base)

```python
class ViT(nn.Module):
    def __init__(self, image_size=224, patch_size=16, num_classes=102, dim=768, depth=12, heads=12):
        super().__init__()
        self.patch_embed = nn.Conv2d(3, dim, patch_size, patch_size)
        num_patches = (image_size // patch_size) ** 2
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))

        self.blocks = nn.ModuleList([
            TransformerBlock(dim, heads) for _ in range(depth)
        ])
        self.head = nn.Linear(dim, num_classes)

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x).flatten(2).transpose(1, 2)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embed

        for block in self.blocks:
            x = block(x)

        return self.head(x[:, 0])
```

#### Attack Implementation

```python
def fgsm_attack(model, images, labels, eps):
    images.requires_grad = True
    outputs = model(images)
    loss = F.cross_entropy(outputs, labels)
    loss.backward()
    grad = images.grad.data
    perturbed_images = images + eps * torch.sign(grad)
    perturbed_images = torch.clamp(perturbed_images, 0, 1)
    return perturbed_images

def pgd_attack(model, images, labels, eps, alpha, steps):
    perturbed_images = images.clone().detach()
    for _ in range(steps):
        perturbed_images.requires_grad = True
        outputs = model(perturbed_images)
        loss = F.cross_entropy(outputs, labels)
        loss.backward()
        grad = perturbed_images.grad.data
        perturbed_images = perturbed_images + alpha * torch.sign(grad)
        perturbed_images = torch.clamp(perturbed_images, images - eps, images + eps)
        perturbed_images = torch.clamp(perturbed_images, 0, 1)
        perturbed_images = perturbed_images.detach()
    return perturbed_images
```

#### Training Parameters

- **Batch Size**: 64 (ResNet), 32 (ViT)
- **Learning Rate**: 0.1 (ResNet, cosine schedule), 1e-3 (ViT)
- **Weight Decay**: 1e-4
- **Epochs**: 100 (ResNet), 50 (ViT)
- **Attack Parameters**:
  - ε: 8/255 ≈ 0.031
  - α: 2/255 ≈ 0.008 (PGD step size)
  - Steps: 10 (PGD iterations)

#### Adversarial Training

```python
# Generate adversarial examples during training
if use_adversarial:
    adv_images = pgd_attack(model, images, labels, eps=8/255, alpha=2/255, steps=10)
    mixed_images = torch.cat([images, adv_images], dim=0)
    mixed_labels = torch.cat([labels, labels], dim=0)
    outputs = model(mixed_images)
    loss = F.cross_entropy(outputs, mixed_labels)
```

### Evaluation Metrics

#### Classification Metrics

- **Clean Accuracy**: Performance on original images
- **Adversarial Accuracy**: Performance under attack
- **Robustness Gap**: Clean - Adversarial accuracy

#### Attack Metrics

- **Success Rate**: Fraction of successful attacks
- **Perturbation Magnitude**: Average ||δ||\_∞ or ||δ||\_2
- **Transferability**: Attack success on different architectures

#### Defense Metrics

- **Certified Robustness**: Fraction of provably robust predictions
- **Computational Cost**: Training/inference overhead
- **Generalization**: Robustness to unseen attacks

### Results and Analysis

#### CIFAR-100 Results (ResNet-50)

| Method          | Clean Acc | FGSM Acc | PGD Acc | Robustness Gap |
| --------------- | --------- | -------- | ------- | -------------- |
| Standard        | 76.2%     | 42.1%    | 31.8%   | -44.4%         |
| Adv Training    | 68.5%     | 58.3%    | 52.1%   | -16.4%         |
| Input Transform | 74.8%     | 48.9%    | 38.2%   | -36.6%         |

#### Flowers-102 Results (ViT-Base)

| Method          | Clean Acc | FGSM Acc | PGD Acc | Robustness Gap |
| --------------- | --------- | -------- | ------- | -------------- |
| Standard        | 84.7%     | 51.2%    | 39.8%   | -44.9%         |
| Adv Training    | 78.3%     | 63.1%    | 57.4%   | -20.9%         |
| Input Transform | 83.1%     | 55.8%    | 45.2%   | -37.9%         |

#### Detailed Analysis

- **Architecture Comparison**: ViT shows slightly higher vulnerability to patch-based attacks
- **Attack Strength**: PGD consistently more effective than FGSM
- **Defense Effectiveness**: Adversarial training provides best robustness improvement
- **Transferability**: Attacks transfer well between ResNet and ViT (60-70% success rate)

#### Training Dynamics

- **Convergence**: Adversarial training converges slower but more stably
- **Overfitting**: Reduced overfitting on clean data with adversarial examples
- **Validation**: Adversarial accuracy correlates with generalization

#### Qualitative Analysis

- **Attack Visualization**: Perturbations concentrate on semantically important regions
- **Attention Maps**: ViT attention patterns disrupted by adversarial examples
- **Grad-CAM**: Attacks shift CNN activations to background regions
- **Robust Models**: More distributed and stable activation patterns

#### Ablation Studies

- **ε Sensitivity**: Larger ε increases attack success but reduces imperceptibility
- **PGD Steps**: More iterations improve attack strength with diminishing returns
- **Defense Combination**: Adversarial training + input transforms yield best results
- **Architecture Depth**: Deeper models more vulnerable but benefit more from defenses

### Challenges and Solutions

#### Computational Complexity

- **Solution**: Efficient attack generation, batch processing
- **Optimization**: Mixed precision training, gradient accumulation

#### Hyperparameter Selection

- **Solution**: Grid search for ε, automated threshold selection
- **Validation**: Cross-validation on held-out adversarial examples

#### Evaluation Rigor

- **Solution**: Multiple attack types, comprehensive metrics
- **Benchmarking**: Compare against published results

#### Interpretability

- **Solution**: Attention visualization, activation analysis
- **Insights**: Attacks exploit architectural weaknesses

### Applications and Extensions

#### Robust Computer Vision

- **Autonomous Driving**: Adversarial robustness for safety-critical systems
- **Medical Imaging**: Reliable diagnosis under adversarial conditions
- **Security Systems**: Face recognition with adversarial defenses

#### Adversarial ML Research

- **Attack Development**: New attack methods and transferability analysis
- **Defense Research**: Novel defense strategies and certified robustness
- **Model Interpretability**: Understanding vulnerabilities through attacks

#### Architecture Design

- **Robust Architectures**: Design principles for adversarial robustness
- **Hybrid Models**: Combining CNN and transformer benefits
- **Efficient Training**: Scalable adversarial training methods

## Files

- `code/NNDL_CAe_1.ipynb`: Complete implementation of attacks and defenses on CNN and ViT
- `report/`: Analysis with attack visualizations, robustness curves, attention maps
- `description/`: Assignment specifications

## Key Learnings

1. CNNs and ViTs exhibit similar adversarial vulnerabilities despite different architectures
2. Adversarial training significantly improves robustness at cost of clean accuracy
3. Attacks transfer well between different vision architectures
4. Defense strategies work across both CNNs and transformers
5. Understanding attack mechanisms reveals architectural insights

## Conclusion

This implementation demonstrates that both CNNs and ViTs are vulnerable to adversarial attacks, with adversarial training providing the most effective defense. ViT achieves 78.3% clean accuracy and 57.4% robust accuracy on Flowers-102, while ResNet reaches 68.5% clean and 52.1% robust accuracy on CIFAR-100. The results highlight the importance of adversarial robustness in modern vision systems.
