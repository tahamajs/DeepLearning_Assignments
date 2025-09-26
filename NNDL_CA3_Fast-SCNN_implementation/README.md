# NNDL_CA3_Fast-SCNN_implementation

This folder contains the implementation of Fast-SCNN (Fast Semantic Segmentation Convolutional Neural Network) for real-time semantic segmentation. Part of Neural Networks and Deep Learning course assignment 3.

## Concepts Covered

### Semantic Segmentation

Semantic segmentation assigns a class label to each pixel in an image, enabling detailed scene understanding.

### Fast-SCNN Architecture

Fast-SCNN is designed for efficient, real-time segmentation on mobile devices.

Key components:

- **Learning to Downsample Module**: Efficient feature extraction with depthwise separable convolutions
- **Global Feature Extractor**: Pyramid Pooling Module (PPM) for multi-scale context
- **Feature Fusion Module**: Combines high-resolution and low-resolution features
- **Classifier**: Lightweight segmentation head

### Advantages

- **Speed**: Optimized for real-time performance
- **Efficiency**: Low computational cost and memory usage
- **Accuracy**: Maintains competitive segmentation quality

### Training

- Loss functions: Cross-entropy, Dice loss, IoU loss
- Data augmentation: Flips, rotations, color changes
- Evaluation metrics: Mean IoU, Dice coefficient, pixel accuracy

### Applications

- Autonomous vehicles (road scene understanding)
- Mobile applications (portrait mode, object removal)
- Robotics and drones

### Challenges

- Balancing speed and accuracy
- Handling class imbalance in segmentation datasets
- Optimizing for edge devices

## Files

- `code/`: Implementation with Fast-SCNN model
- Dataset: CamVid or similar segmentation dataset

## Results

Fast-SCNN achieves real-time segmentation with competitive accuracy, demonstrating efficient deep learning for pixel-level tasks.
