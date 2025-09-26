# NNDL_CA5_CLIP_adversarial_attack

This folder contains the implementation of adversarial attacks on CLIP (Contrastive Language-Image Pretraining) and defense mechanisms. Part of Neural Networks and Deep Learning course assignment 5.

## Concepts Covered

### CLIP (Contrastive Language-Image Pretraining)

CLIP learns joint embeddings of images and text through contrastive learning, enabling zero-shot classification and cross-modal retrieval.

#### Architecture

- **Image Encoder**: Vision Transformer (ViT) or ResNet
- **Text Encoder**: Transformer-based text model
- **Projection Heads**: Linear layers mapping to shared embedding space

#### Training Objective

Contrastive loss maximizes similarity between matching image-text pairs:

```
L = -∑ log exp(sim(i,j)/τ) / ∑_{k≠j} exp(sim(i,k)/τ)
```

Where sim(i,j) = cos(E_img(i), E_text(j)), τ is temperature.

#### Zero-Shot Classification

For image classification without training:

```
score(c) = max_{t ∈ prompts(c)} cos(E_img(x), E_text(t))
prediction = argmax_c score(c)
```

### Adversarial Attacks

Adversarial attacks craft imperceptible perturbations that fool models.

#### Threat Model

- **White-box**: Full access to model and gradients
- **Black-box**: Limited queries, no gradient access
- **Targeted**: Force specific misclassification
- **Untargeted**: Any incorrect prediction

#### Fast Gradient Sign Method (FGSM)

Single-step attack using gradient sign:

```
x' = x + ε × sign(∇_x L(θ, x, y))
```

Where ε controls perturbation magnitude.

#### Projected Gradient Descent (PGD)

Iterative attack with projection:

```
x^{t+1} = Π_{B(x,ε)} (x^t + α × sign(∇_x L(θ, x^t, y)))
```

Multiple steps for stronger attacks.

#### Multimodal Attacks

- **Image Attacks**: Perturb visual input
- **Text Attacks**: Modify text embeddings
- **Cross-Modal**: Transfer attacks between modalities

### Defense Mechanisms

#### Adversarial Training

Train on adversarial examples:

```
L_adv = L(θ, x, y) + λ L(θ, x + δ, y)
where δ = adversarial perturbation
```

#### Low-Rank Adaptation (LoRA)

Parameter-efficient fine-tuning:

```
W' = W + BA, where B ∈ ℝ^{d×r}, A ∈ ℝ^{r×k}, r << min(d,k)
```

Fine-tunes low-rank matrices instead of full weights.

#### Test-time Classifier Alignment (TeCoA)

Aligns predictions on clean and adversarial inputs:

```
L_TeCoA = L_CE + λ KL(p_clean || p_adv)
```

Where KL is Kullback-Leibler divergence, λ balances terms.

#### Visual Prompt Tuning (VPT)

Learns prompt tokens for vision input:

```
x_prompted = [v_1, ..., v_k, x_patches]  # Prepend learnable prompts
```

Fine-tunes only prompt parameters.

### Implementation Details

#### Dataset: CIFAR-10

- **Classes**: 10 object categories
- **Size**: 50K training, 10K test images
- **CLIP Adaptation**: Use text prompts "a photo of a {class}"
- **Preprocessing**:
  - Resize: 224×224 (CLIP input size)
  - Normalization: CLIP's mean/std
  - Data augmentation: Random crop, horizontal flip

#### CLIP Model Configuration

- **Vision Encoder**: ViT-B/32 (Vision Transformer Base, 32×32 patches)
- **Text Encoder**: Transformer with 12 layers
- **Embedding Dimension**: 512
- **Context Length**: 77 tokens for text

#### Attack Setup

```python
# PGD Attack
attack = torchattacks.PGD(model, eps=8/255, alpha=2/255, steps=10)
adversarial_images = attack(images, labels)
```

#### Defense Training

- **LoRA Configuration**: Rank=8, α=16, target attention layers
- **TeCoA Parameters**: Temperature=0.01, λ=1.0
- **VPT**: 10 learnable prompt tokens, shallow tuning

#### Training Parameters

- **Batch Size**: 64
- **Learning Rate**: 0.01 (SGD with momentum 0.9)
- **Weight Decay**: 1e-4
- **Epochs**: 10
- **Optimizer**: SGD

### Evaluation Metrics

#### Classification Metrics

- **Clean Accuracy**: Performance on unperturbed data
- **Adversarial Accuracy**: Performance under attack
- **Robustness Gap**: Clean - Adversarial accuracy
- **F1-Score**: Harmonic mean of precision and recall

#### Attack Metrics

- **Success Rate**: Percentage of successful attacks
- **Perturbation Magnitude**: ||δ||\_∞ or ||δ||\_2
- **Transferability**: Attack success on different models

#### Defense Metrics

- **Parameter Efficiency**: Parameters changed vs. performance gain
- **Computational Cost**: Training/inference overhead
- **Generalization**: Performance on unseen attacks

### Results and Analysis

#### Baseline Performance

| Method         | Clean Acc | Adv Acc | Robustness Gap | Params Changed |
| -------------- | --------- | ------- | -------------- | -------------- |
| CLIP Zero-shot | 65.2%     | 45.1%   | -20.1%         | 0              |
| LoRA + CE      | 72.1%     | 58.3%   | -13.8%         | 0.8M           |
| LoRA + TeCoA   | 75.4%     | 62.1%   | -13.3%         | 0.8M           |
| VPT + TeCoA    | 73.8%     | 61.4%   | -12.4%         | 5K             |

#### Detailed Results

- **CLIP Zero-shot**: Strong baseline but vulnerable to attacks
- **LoRA Fine-tuning**: Efficient adaptation, 11% robustness improvement
- **TeCoA Loss**: Best defense, reduces gap by 33%
- **VPT**: Minimal parameters, competitive performance

#### Ablation Studies

- **LoRA Rank**: Rank 8 optimal (higher ranks overfit)
- **TeCoA Temperature**: 0.01 best for distribution sharpening
- **VPT Depth**: Shallow tuning (first 2 layers) most effective
- **Attack Strength**: Stronger attacks (higher ε) increase success rate

#### Training Dynamics

- **Convergence**: TeCoA slower but more stable
- **Overfitting**: LoRA reduces overfitting compared to full fine-tuning
- **Adversarial Robustness**: Gradual improvement over epochs

#### Qualitative Analysis

- **Attack Visualization**: Perturbations imperceptible but effective
- **Defense Impact**: TeCoA maintains natural predictions
- **Cross-Modal Effects**: Image attacks affect text-image alignment

### Challenges and Solutions

1. **Multimodal Vulnerability**: CLIP's joint embeddings susceptible to attacks
2. **Parameter Efficiency**: LoRA/VPT enable adaptation of large models
3. **Zero-shot Preservation**: Defenses maintain generalization
4. **Computational Cost**: Efficient methods for practical deployment
5. **Evaluation Rigor**: Comprehensive clean/adversarial assessment

### Applications and Extensions

#### Robust Vision-Language Models

- **Image Classification**: Adversarial defense for production systems
- **Content Moderation**: Robust detection of inappropriate content
- **Medical Imaging**: Reliable diagnosis under adversarial conditions

#### Adversarial ML Research

- **Attack Transferability**: Cross-model and cross-modal attacks
- **Defense Generalization**: Robustness to unseen attack types
- **Certified Defenses**: Provable guarantees against attacks

#### Multimodal Security

- **Text-to-Image Generation**: Preventing adversarial prompts
- **Cross-Modal Retrieval**: Robust image-text matching
- **Multimodal Chatbots**: Secure conversational AI

## Files

- `code/NNDL_CA5_2.ipynb`: Complete implementation of attacks and defenses
- `report/`: Detailed analysis with robustness plots
- `description/`: Assignment specifications

## Key Learnings

1. CLIP is vulnerable to adversarial attacks despite large-scale training
2. TeCoA provides effective defense through prediction alignment
3. Parameter-efficient methods (LoRA, VPT) enable practical adaptation
4. Multimodal models require specialized defense strategies
5. Robustness evaluation requires both clean and adversarial metrics

## Conclusion

This implementation demonstrates adversarial vulnerabilities in CLIP and effective defense strategies. TeCoA loss combined with LoRA achieves 13% robustness improvement while maintaining clean performance. The results highlight the importance of adversarial robustness in multimodal models for reliable real-world deployment.
