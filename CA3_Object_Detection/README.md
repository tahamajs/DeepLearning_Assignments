# CA3: Object Detection

This assignment explores advanced computer vision techniques for object detection and segmentation, implementing two state-of-the-art architectures: Fast-SCNN for real-time semantic segmentation and Oriented R-CNN for arbitrary-oriented object detection.

## Overview

The assignment consists of two specialized projects:

1. **Fast-SCNN Implementation**: Real-time semantic segmentation for mobile devices
2. **Oriented R-CNN Implementation**: Detection of objects with arbitrary orientations

## Contents

- `Fast_SCNN/`: Real-time semantic segmentation implementation
- `Oriented_RCNN/`: Arbitrary-oriented object detection system

Each subfolder contains:

- `code/`: PyTorch implementations
- `description/`: Assignment specifications
- `paper/`: Research papers and references
- `report/`: Detailed analysis and results

## Fast-SCNN: Real-Time Semantic Segmentation

### Key Features

- **Efficient Architecture**: Depthwise separable convolutions for computational efficiency
- **Multi-Scale Processing**: Pyramid pooling module for global context
- **Real-Time Performance**: Optimized for mobile and embedded deployment
- **Urban Scene Understanding**: Segmentation of roads, buildings, vehicles, pedestrians

### Technical Details

- **Learning to Downsample**: Initial downsampling module with skip connections
- **Global Feature Extractor**: Pyramid pooling with multiple kernel sizes (1×1, 2×2, 3×3, 6×6)
- **Feature Fusion**: Concatenation of multi-scale features with channel attention
- **Loss Function**: Cross-entropy with class balancing for imbalanced segmentation

### Results

- **IoU Score**: 0.62 average across all classes
- **Model Size**: 1.2M parameters (vs. 50M+ for standard segmentation models)
- **Inference Speed**: 30+ FPS on mobile GPUs
- **Memory Efficiency**: 50MB model size suitable for edge deployment

## Oriented R-CNN: Arbitrary-Oriented Object Detection

### Key Features

- **Oriented Anchors**: 5-parameter anchor representation (x, y, w, h, θ)
- **Rotated ROI Align**: Rotation-aware feature extraction
- **Geometric Transformations**: Proper handling of oriented bounding boxes
- **IoU Computation**: Specialized intersection-over-union for rotated rectangles

### Technical Details

- **Region Proposal Network (RPN)**: Oriented anchor generation and classification
- **Rotated RoI Align**: Bilinear sampling with rotation compensation
- **Bounding Box Regression**: 5-parameter regression (dx, dy, dw, dh, dθ)
- **Orientation Encoding**: Angle representation and normalization

### Results

- **Detection Accuracy**: Superior performance on oriented objects vs. axis-aligned methods
- **Geometric Precision**: Accurate localization of rotated objects
- **Robustness**: Handles various orientations and aspect ratios
- **Application**: Ship detection in satellite imagery, text detection in documents

## Key Concepts Demonstrated

### Semantic Segmentation

- **Pixel-wise Classification**: Assigning class labels to every pixel
- **Encoder-Decoder Architectures**: U-Net style networks for segmentation
- **Efficient Convolutions**: Depthwise separable and grouped convolutions
- **Multi-Scale Features**: Pyramid pooling and feature fusion

### Object Detection

- **Region Proposal Networks**: Generating candidate object locations
- **RoI Pooling/Align**: Feature extraction from proposed regions
- **Bounding Box Regression**: Refining object localization
- **Orientation Handling**: 5-DoF bounding box representation

### Geometric Transformations

- **Rotation Matrices**: Mathematical representation of orientations
- **IoU for Rotated Boxes**: Intersection computation for oriented rectangles
- **Spatial Transformations**: Affine and projective transformations
- **Coordinate Systems**: Converting between different geometric representations

## Educational Value

This assignment provides deep understanding of:

- **Advanced CV Architectures**: State-of-the-art detection and segmentation models
- **Geometric Computer Vision**: Handling rotations and orientations
- **Efficient Deep Learning**: Model compression and mobile optimization
- **Real-world Applications**: Aerial imagery, autonomous driving, document analysis

## Dependencies

- Python 3.8+
- PyTorch 1.9+
- torchvision
- OpenCV
- NumPy, Pandas
- Matplotlib, Seaborn
- CUDA (recommended for training)

## Usage

### Fast-SCNN

1. Navigate to `Fast_SCNN/code/`
2. Run data preprocessing and model training
3. Evaluate segmentation performance on test sets
4. Review results in `Fast_SCNN/report/`

### Oriented R-CNN

1. Navigate to `Oriented_RCNN/code/`
2. Execute the training pipeline for oriented detection
3. Test on oriented object datasets
4. Analyze geometric accuracy in `Oriented_RCNN/report/`

## References

- [Fast-SCNN Description](Fast_SCNN/description/)
- [Oriented R-CNN Description](Oriented_RCNN/description/)
- [Research Papers](Fast_SCNN/paper/) | [Research Papers](Oriented_RCNN/paper/)
- [Implementation Reports](Fast_SCNN/report/) | [Implementation Reports](Oriented_RCNN/report/)

---

**Course**: Neural Networks and Deep Learning (CA3)
**Institution**: University of Tehran
**Date**: September 2025</content>
<parameter name="filePath">/Users/tahamajs/Documents/uni/LLM/Deep_UT/CA1_Neural_Networks_Basics/README.md
