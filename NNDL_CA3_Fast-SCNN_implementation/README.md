# NNDL_CA3_Fast-SCNN_implementation

This folder contains the implementation of Fast-SCNN (Fast Semantic Segmentation Convolutional Neural Network) for real-time semantic segmentation. Part of Neural Networks and Deep Learning course assignment 3.

## Concepts Covered

### Semantic Segmentation

Semantic segmentation assigns a class label to each pixel in an image, enabling detailed scene understanding. Unlike classification (one label per image) or detection (bounding boxes), segmentation provides pixel-level granularity.

#### Mathematical Formulation

For an input image I ∈ ℝ^{H×W×C} and ground truth segmentation mask Y ∈ ℤ^{H×W}, the goal is to learn a function f such that f(I) ≈ Y, where each pixel (i,j) in the output has a class probability distribution.

#### Evaluation Metrics

- **Pixel Accuracy**: (TP + TN) / Total pixels
- **Mean IoU**: Average of IoU scores across classes
  ```
  IoU_c = TP_c / (TP_c + FP_c + FN_c)
  mIoU = (1/C) ∑_c IoU_c
  ```
- **Dice Coefficient**: 2TP / (2TP + FP + FN)

### Fast-SCNN Architecture

Fast-SCNN is designed for efficient, real-time segmentation on mobile devices, balancing speed and accuracy.

#### Key Components:

1. **Learning to Downsample Module**:

   - **Standard Convolution**: Conv2D(32, 3×3, stride=2) + ReLU
   - **Depthwise Separable Convolution (DSConv)**: Depthwise conv + Pointwise conv
     - Depthwise: Conv2D(in_channels, 3×3, groups=in_channels)
     - Pointwise: Conv2D(in_channels, 1×1, groups=1)
   - Reduces spatial resolution while extracting multi-scale features
   - Uses ReLU activation and L2 regularization

2. **Global Feature Extractor**:

   - **Bottleneck Blocks**: Inspired by MobileNetV2
     - Expansion: 1×1 conv to increase channels (e.g., 32→96)
     - Depthwise: 3×3 depthwise conv
     - Projection: 1×1 conv to reduce channels (96→16)
     - Residual connection if input/output channels match
   - **Pyramid Pooling Module (PPM)**: Captures multi-scale context
     - Average pooling at scales: 1×1, 2×2, 3×3, 6×6
     - Bilinear upsampling to original size
     - Concatenation and 1×1 conv for fusion
   - Captures global context for better segmentation

3. **Feature Fusion Module**:

   - Combines high-resolution features (1/8 size) from downsampling module
   - With low-resolution features (1/32 size) from global extractor
   - Upsampling of low-res features via bilinear interpolation
   - Dilated depthwise conv: Conv with dilation rate >1 for larger receptive field
     - Effective kernel size: (dilation-1)×(kernel-1) + kernel
   - Element-wise addition and ReLU

4. **Classifier**:
   - DSConv layers for efficient computation
   - Final Conv2D(num_classes, 1×1) with 11 classes (for CamVid dataset)
   - Dropout(0.3) and Softmax for pixel-wise classification

#### Computational Efficiency

- **Depthwise Separable Convolutions**: Reduce parameters by ~9× compared to standard conv
- **Total Parameters**: ~1.2M (vs. ~50M for DeepLabV3+)
- **Inference Speed**: ~120 FPS on mobile GPUs

### Training Details

#### Dataset

- **CamVid Dataset**: 367 training, 101 validation, 233 test images
- **Classes**: 11 semantic classes (Sky, Building, Pole, Road, Pavement, Tree, SignSymbol, Fence, Car, Pedestrian, Bicyclist)
- **Resolution**: 360×480 pixels

#### Loss Functions

- **Cross-Entropy Loss**: Pixel-wise classification loss
  ```
  L_CE = -∑_c y_c log(ŷ_c)
  ```
- **IoU Loss**: 1 - IoU score, encourages better overlap
- **Dice Loss**: 1 - Dice coefficient, good for imbalanced classes
  ```
  L_Dice = 1 - (2∑ y_i ŷ_i + ε) / (∑ y_i + ∑ ŷ_i + ε)
  ```

#### Optimization

- **Optimizer**: Adam (β1=0.9, β2=0.999)
- **Learning Rate**: Polynomial decay from 0.045 to 0
  ```
  lr = initial_lr × (1 - current_step/max_steps)^power
  ```
- **Batch Size**: 16
- **Epochs**: 100 with early stopping
- **Weight Decay**: 4e-5 for L2 regularization

#### Data Augmentation

- **Geometric**: Random horizontal flip (p=0.5)
- **Photometric**: Random brightness ±0.2, contrast ±0.2, saturation ±0.2
- **Noise**: Gaussian noise with σ=0.1
- **Color Jitter**: Random hue shifts

### Results

#### Quantitative Results

| Loss Function  | Val Accuracy | Dice Coefficient | IoU  | Params |
| -------------- | ------------ | ---------------- | ---- | ------ |
| Cross-Entropy  | 0.85         | 0.65             | 0.55 | 1.2M   |
| IoU Loss       | 0.87         | 0.70             | 0.60 | 1.2M   |
| Dice Loss      | 0.88         | 0.75             | 0.62 | 1.2M   |
| + Augmentation | 0.90         | 0.78             | 0.65 | 1.2M   |

#### Qualitative Analysis

- **Road Segmentation**: High accuracy on straight roads, challenges with complex intersections
- **Vehicle Detection**: Good separation of cars from background
- **Class Imbalance**: Void class (unlabeled) affects overall metrics
- **Real-time Performance**: Maintains >30 FPS on edge devices

#### Ablation Study

- **Without PPM**: IoU drops by 8%
- **Without Feature Fusion**: IoU drops by 12%
- **Standard Conv vs DSConv**: 3× parameter increase with minimal accuracy gain

### Model Parameters Breakdown

- **Learning to Downsample**: ~50K parameters
- **Global Feature Extractor**: ~800K parameters
  - Bottleneck blocks: ~600K
  - PPM: ~200K
- **Feature Fusion**: ~100K parameters
- **Classifier**: ~250K parameters
- **Total**: ~1.2M parameters

### Applications

- **Autonomous Vehicles**: Real-time road scene understanding
- **Mobile Applications**: Portrait mode, object removal, AR filters
- **Robotics and Drones**: Navigation and obstacle avoidance
- **Medical Imaging**: Organ segmentation in ultrasound/CT

### Challenges and Solutions

1. **Speed-Accuracy Trade-off**: DSConv and efficient design maintain real-time performance
2. **Class Imbalance**: Dice loss helps with rare classes
3. **Limited Receptive Field**: PPM provides global context
4. **Edge Device Deployment**: Quantization and pruning for mobile optimization

## Files

- `code/NNDL_CA3_1.ipynb`: Complete implementation with model definition, training loops, and evaluation
- `report/`: Detailed analysis with plots and comparisons
- `paper/`: Original Fast-SCNN paper by Poudel et al.
- `description/`: Assignment requirements and dataset details

## Key Learnings

1. Efficient architectures can achieve high accuracy with low computational cost
2. Depthwise separable convolutions are crucial for mobile deployment
3. Multi-scale feature fusion improves segmentation quality
4. Appropriate loss functions are essential for imbalanced segmentation tasks
5. Data augmentation significantly improves generalization

## Conclusion

Fast-SCNN demonstrates effective real-time semantic segmentation, achieving competitive accuracy (IoU ~0.62) with low computational cost (~1.2M parameters), making it suitable for mobile and embedded applications. The implementation showcases modern CNN design principles for efficient deep learning on resource-constrained devices.
