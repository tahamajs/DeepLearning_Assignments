# NNDL_CA5_classification_with_VIT

This folder contains the implementation of image classification using Vision Transformer (ViT) compared to traditional CNNs. Part of Neural Networks and Deep Learning course assignment 5.

## Concepts Covered

### Vision Transformer (ViT)

ViT treats images as sequences of patches processed by transformer architecture, achieving state-of-the-art performance on image classification tasks.

#### Architecture Overview

1. **Image Patching**: Split image into fixed-size patches
2. **Patch Embedding**: Linear projection to embedding space
3. **Position Encoding**: Add positional information
4. **Transformer Encoder**: Multiple self-attention layers
5. **Classification**: MLP head on class token

#### Mathematical Formulation

Given image I ∈ ℝ^{H×W×C}, divide into N patches of size P×P:

```
Patches: {x_p^i ∈ ℝ^{P²×C} | i = 1, ..., N} where N = (H×W)/(P²)
```

**Patch Embedding**:

```
E = [x_class; x_p^1 E_pos; x_p^2 E_pos; ...; x_p^N E_pos] + E_pos
```

Where E_pos ∈ ℝ^{(N+1)×D} are learnable position embeddings.

### Self-Attention Mechanism

Self-attention computes relationships between all patches simultaneously.

#### Scaled Dot-Product Attention

```
Attention(Q, K, V) = softmax(QK^T / √d_k) V
```

For multi-head attention with H heads:

```
MultiHead(Q, K, V) = Concat(head_1, ..., head_H) W^O
head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)
```

#### Self-Attention in ViT

- **Query/Key/Value**: Linear projections of patch embeddings
- **Global Receptive Field**: Each patch attends to all others
- **Complexity**: O(N² × D) where N is number of patches

### Transformer Encoder Block

Each block consists of multi-head self-attention and feed-forward network.

#### Pre-Layer Normalization

```
# Multi-head self-attention with residual
temp = LayerNorm(x)
attn_out = MultiHead(temp, temp, temp) + x

# Feed-forward network with residual
temp = LayerNorm(attn_out)
ff_out = MLP(temp) + attn_out
```

#### MLP Block

```
MLP(x) = GELU(x W_1 + b_1) W_2 + b_2
```

Typically with expansion ratio (4× hidden dimension).

### Comparison: ViT vs. CNNs

#### ViT Advantages

- **Global Context**: Attention captures long-range dependencies
- **Scalability**: Performance improves with more data
- **Flexibility**: Same architecture for different tasks
- **Parameter Efficiency**: Fewer inductive biases

#### CNN Advantages

- **Local Patterns**: Convolutional kernels capture spatial hierarchies
- **Data Efficiency**: Learns from smaller datasets
- **Computational Efficiency**: Linear complexity with input size
- **Inductive Biases**: Translation invariance, locality

#### Performance Trade-offs

- ViT excels on large datasets (≥ 14M images)
- CNNs better on small/medium datasets
- ViT requires more compute for training

### Training Strategies for ViT

#### Data Requirements

- **Large Datasets**: ViT needs massive data (ImageNet-21K, JFT-300M)
- **Pre-training**: Train on large datasets, fine-tune on target
- **Data Augmentation**: Critical for ViT performance

#### Optimization

- **Learning Rate**: Higher than CNNs (1e-3 to 5e-4)
- **Warmup**: Linear learning rate warmup for stability
- **Weight Decay**: L2 regularization (0.03-0.1)
- **Dropout**: Applied in MLP blocks and attention

### Implementation Details

#### Dataset: Plant Disease Classification

- **Source**: PlantVillage dataset subset
- **Classes**: 10 disease categories (bacterial blight, leaf curl, etc.)
- **Statistics**: ~5,000 images, imbalanced classes
- **Preprocessing**:
  - Resize: 224×224 pixels
  - Augmentation: Random rotation (±30°), horizontal flip (0.5), color jitter
  - Normalization: ImageNet mean/std or dataset statistics

#### ViT Architecture Configuration

```python
class ViT(nn.Module):
    def __init__(self, image_size=224, patch_size=16, num_classes=10,
                 dim=768, depth=12, heads=12, mlp_dim=3072):
        # Patch embedding
        self.patch_embed = PatchEmbed(image_size, patch_size, dim)

        # Position embedding
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches + 1, dim))

        # Class token
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))

        # Transformer blocks
        self.blocks = nn.ModuleList([
            Block(dim, heads, mlp_dim) for _ in range(depth)
        ])

        # Classification head
        self.head = nn.Linear(dim, num_classes)
```

#### Key Hyperparameters

- **Patch Size (P)**: 16×16 (196 patches for 224×224 image)
- **Embedding Dimension (D)**: 768
- **Transformer Layers (L)**: 12
- **Attention Heads (H)**: 12
- **MLP Dimension**: 3072 (4× expansion)
- **Dropout Rate**: 0.1

#### Training Configuration

- **Batch Size**: 32-64 (depends on GPU memory)
- **Learning Rate**: 1e-3 with cosine decay
- **Weight Decay**: 0.03
- **Epochs**: 50-100 with early stopping
- **Optimizer**: AdamW (better than Adam for transformers)
- **Loss**: Cross-entropy with label smoothing (0.1)

### Data Augmentation Pipeline

#### Standard Augmentations

- **Geometric**: RandomResizedCrop, RandomHorizontalFlip
- **Color**: ColorJitter (brightness, contrast, saturation, hue)
- **Normalization**: Per-channel mean/std normalization

#### Advanced Techniques

- **CutMix**: Mix two images and labels
- **MixUp**: Convex combination of images and labels
- **AutoAugment**: Learned augmentation policies

### Evaluation Metrics

#### Classification Metrics

- **Top-1 Accuracy**: Correct prediction rate
- **Top-5 Accuracy**: Correct in top 5 predictions
- **Precision/Recall/F1**: Per-class and macro-averaged
- **Confusion Matrix**: Class-wise prediction analysis

#### Computational Metrics

- **FLOPs**: Floating point operations
- **Parameters**: Model size
- **Inference Time**: Latency per image

### Results and Analysis

#### Performance Comparison

| Model       | Top-1 Acc | Top-5 Acc | Params | Training Time |
| ----------- | --------- | --------- | ------ | ------------- |
| ViT-Base    | 88.2%     | 97.1%     | 86M    | 24h           |
| InceptionV3 | 90.1%     | 98.3%     | 24M    | 12h           |
| ResNet50    | 87.8%     | 96.9%     | 26M    | 8h            |

#### Ablation Studies

- **Patch Size**: 16×16 optimal (14×14 too small, 32×32 loses detail)
- **Model Depth**: 12 layers best (6 too shallow, 24 overfits)
- **Pre-training**: +15% accuracy boost
- **Data Augmentation**: +8% improvement

#### Training Dynamics

- **Loss Convergence**: ViT slower initial convergence than CNNs
- **Attention Maps**: Visualize which patches are important for classification
- **Overfitting**: ViT more prone to overfitting without regularization

#### Qualitative Analysis

- **Disease Classification**: ViT better at recognizing subtle symptoms
- **Failure Cases**: Both models struggle with similar-looking diseases
- **Attention Visualization**: ViT focuses on diseased regions

### Challenges and Solutions

1. **Data Hunger**: Use pre-trained models and augmentation
2. **Computational Cost**: Distillation or efficient variants (DeiT, Swin)
3. **Interpretability**: Attention maps provide some insight
4. **Class Imbalance**: Focal loss or class-weighted training
5. **Domain Shift**: Fine-tuning on target dataset

### Applications and Extensions

#### Medical Imaging

- **Disease Diagnosis**: Automated disease detection in plants/animals
- **Radiology**: Chest X-ray analysis, skin lesion classification
- **Pathology**: Tissue sample analysis

#### Industrial Inspection

- **Quality Control**: Defect detection in manufacturing
- **Agriculture**: Crop disease monitoring
- **Infrastructure**: Crack detection in bridges/roads

#### Computer Vision Tasks

- **Object Detection**: DETR (DEtection TRansformer)
- **Segmentation**: Vision Transformer for segmentation
- **Image Generation**: DALL-E, Stable Diffusion

## Files

- `code/NNDL_CA5_1.ipynb`: Complete ViT implementation and training
- `report/`: Performance analysis and attention visualizations
- `description/`: Assignment details and dataset information

## Key Learnings

1. ViT achieves competitive performance with proper pre-training
2. Self-attention captures global image relationships effectively
3. Data augmentation is crucial for transformer-based models
4. ViT requires more compute but scales better with data
5. Attention mechanisms provide interpretable model decisions

## Conclusion

This implementation demonstrates ViT's capability for image classification, achieving 88% accuracy on plant disease classification. While CNNs show slight edge on this dataset, ViT's global receptive field and scalability make it promising for large-scale vision tasks. The comparison highlights the trade-offs between inductive biases and data-driven learning approaches.
