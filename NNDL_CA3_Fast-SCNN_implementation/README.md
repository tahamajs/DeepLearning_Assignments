# NNDL_CA3_Fast-SCNN_implementation

This folder contains the implementation of Fast-SCNN (Fast Semantic Segmentation Convolutional Neural Network) for real-time semantic segmentation. Part of Neural Networks and Deep Learning course assignment 3.

## Concepts Covered

### Semantic Segmentation

Semantic segmentation assigns a class label to each pixel in an image, enabling detailed scene understanding. Unlike classification (one label per image) or detection (bounding boxes), segmentation provides pixel-level granularity.

### Fast-SCNN Architecture

Fast-SCNN is designed for efficient, real-time segmentation on mobile devices, balancing speed and accuracy.

Key components:

- **Learning to Downsample Module**:

  - Conv2D + DSConv (Depthwise Separable Convolution) + DSConv
  - Reduces spatial resolution while extracting multi-scale features
  - Uses ReLU activation and L2 regularization

- **Global Feature Extractor**:

  - Bottleneck blocks with expansion, depthwise conv, and residual connections
  - Pyramid Pooling Module (PPM): Applies average pooling at different scales (1x1, 2x2, 3x3, 6x6), bilinear upsampling, and concatenation
  - Captures global context for better segmentation

- **Feature Fusion Module**:

  - Combines high-resolution features from downsampling module with low-resolution features from global extractor
  - Uses dilated depthwise conv for larger receptive field
  - Element-wise addition and ReLU

- **Classifier**:
  - DSConv layers for efficient computation
  - Final Conv2D with 11 classes (for CamVid dataset)
  - Dropout and Softmax

### Training Details

- **Dataset**: CamVid dataset with 11 classes (Sky, Building, Pole, Road, etc.)
- **Loss Functions**:
  - Cross-Entropy Loss
  - IoU Loss: 1 - IoU score
  - Dice Loss: 1 - Dice coefficient
- **Metrics**: Accuracy, Dice Coefficient, IoU Score
- **Optimizer**: Adam with PolynomialDecay learning rate
- **Batch Size**: 16
- **Epochs**: 100
- **Data Augmentation**: Horizontal flip, brightness adjustment, Gaussian noise

### Results

#### Cross-Entropy Loss

- Final validation accuracy: ~0.85
- Dice coefficient: ~0.65
- IoU: ~0.55
- Training converges steadily, with some overfitting indicated by validation plateau

#### IoU Loss

- Improved IoU: ~0.60
- Dice: ~0.70
- Better alignment with segmentation metrics

#### Dice Loss

- Highest Dice: ~0.75
- IoU: ~0.62
- Optimized directly for Dice metric, good for imbalanced classes

#### With Data Augmentation

- Reduced overfitting
- Higher validation Dice/IoU
- More robust to variations

#### Final Model (All Data)

- Trained on train+val, tested on test set
- Achieves real-time performance with ~1.2M parameters
- Qualitative results show accurate segmentation of roads, buildings, vehicles

### Model Parameters

- Total parameters: ~1.2 million
- Learning to Downsample: ~50K
- Global Feature Extractor: ~800K
- Feature Fusion: ~100K
- Classifier: ~250K

### Applications

- Autonomous vehicles (road scene understanding)
- Mobile applications (portrait mode, object removal)
- Robotics and drones

### Challenges Addressed

- Balancing speed and accuracy for real-time use
- Handling class imbalance (e.g., void class)
- Efficient computation on edge devices

## Files

- `code/`: Implementation notebook with Fast-SCNN model, training, and evaluation
- `report/`: Detailed report with results and analysis
- `paper/`: Original Fast-SCNN paper
- `description/`: Assignment description

## Conclusion

Fast-SCNN demonstrates effective real-time semantic segmentation, achieving competitive accuracy with low computational cost, making it suitable for mobile and embedded applications.
