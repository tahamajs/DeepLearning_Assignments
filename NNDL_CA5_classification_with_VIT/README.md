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

## Implementation Details

### Dataset

- **Plant Disease Dataset**: Images of plant leaves with disease labels
- **Classes**: 10 disease types (e.g., various bacterial/fungal infections)
- **Preprocessing**:
  - Resize to 224x224
  - Data augmentation (rotation, flip, color jitter)
  - Oversampling for minority classes
  - Normalization with dataset mean/variance

### ViT Architecture

- **Patch Size**: 16x16
- **Embedding Dimension**: 64
- **Transformer Layers**: 6 projection layers
- **Attention Heads**: 8
- **MLP Dimensions**: 128 → 64
- **Class Token**: Prepended to patch sequence

### Key Components

#### Patches Layer

- Splits image into patches
- Output shape: (num_patches, patch_size*patch_size*channels)

#### Patch Encoder

- Linear projection to embedding_dim
- Adds position embeddings
- Includes class token

#### Transformer Block

- Multi-head attention (8 heads, key_dim=64)
- Feed-forward: Dense(128) → Dense(64)
- Residual connections and layer norm
- Dropout for regularization

#### Classification Head

- Flatten transformer output
- Dense layers: 2048 → 1024 → num_classes
- Softmax activation

### Training Parameters

- **Batch Size**: 32
- **Learning Rate**: 0.001 with weight decay
- **Epochs**: 50
- **Optimizer**: Adam
- **Loss**: Categorical cross-entropy
- **Metrics**: Accuracy, precision, recall, F1-score

### Data Augmentation

- Random rotation, horizontal flip
- Brightness/contrast adjustment
- Normalization per channel

## Results

### ViT Performance

- **Training Accuracy**: ~95%
- **Validation Accuracy**: ~88%
- **Precision**: 0.87
- **Recall**: 0.86
- **F1-Score**: 0.86

### InceptionV3 Comparison

- **Training Accuracy**: ~97%
- **Validation Accuracy**: ~90%
- **Precision**: 0.89
- **Recall**: 0.88
- **F1-Score**: 0.88

### Training Dynamics

- ViT converges slower initially but reaches similar performance
- InceptionV3 has faster convergence due to inductive biases
- Both models benefit from data augmentation
- Oversampling improves minority class performance

### Ablation Studies

- **Patch Size**: Smaller patches (8x8) increase sequence length, higher accuracy but slower
- **Embedding Dim**: Higher dimensions improve performance but increase parameters
- **Transformer Layers**: More layers improve capacity but risk overfitting
- **Attention Heads**: More heads better capture multi-scale features

### Model Parameters

- **ViT**: ~2.1M parameters
- **InceptionV3**: ~23.8M parameters
- ViT more parameter-efficient for large datasets

### Challenges Addressed

- **Data Efficiency**: ViTs need more data than CNNs
- **Computational Cost**: Attention scales quadratically with patches
- **Class Imbalance**: Oversampling minority disease classes
- **Overfitting**: Dropout, weight decay, augmentation
