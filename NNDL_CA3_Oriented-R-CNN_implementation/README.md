# NNDL_CA3_Oriented-R-CNN_implementation

This folder contains the implementation of Oriented R-CNN for oriented object detection, specifically for ship detection with rotated bounding boxes. Part of Neural Networks and Deep Learning course assignment 3.

## Concepts Covered

### Oriented Object Detection

Traditional object detection uses axis-aligned bounding boxes (AABBs), but many objects (ships, vehicles) are oriented. Oriented detection uses rotated rectangles defined by center (x,y), width, height, and angle θ.

### Oriented R-CNN Architecture

Extends Faster R-CNN for oriented bounding boxes.

Key components:

- **Backbone**: ResNet-50 with FPN (Feature Pyramid Network) for multi-scale features
- **Region Proposal Network (RPN)**: Generates oriented proposals using oriented anchors
- **ROI Align**: Rotated ROI Align for extracting features from oriented regions
- **RCNN Head**: Predicts oriented box refinements and classifications

### Oriented Bounding Box Representation

- **5-parameter**: (x, y, w, h, θ) - center, size, rotation angle
- **8-parameter**: Four corner coordinates (x1,y1,x2,y2,x3,y3,x4,y4)

### Loss Functions

- **RPN Classification Loss**: Binary cross-entropy for object/background
- **RPN Regression Loss**: Smooth L1 for box refinement (oriented)
- **RCNN Classification Loss**: Cross-entropy for class prediction
- **RCNN Regression Loss**: Smooth L1 for oriented box regression

### Training Details

- **Dataset**: HRSC2016 (High-Resolution Ship Collection 2016)
  - 1061 images with ship annotations
  - Oriented bounding boxes for ships
  - Classes: ship (single class)
- **Data Loading**: Custom HRSCDataset class parsing XML annotations
- **Preprocessing**: Resize to 1024x1024, normalization
- **Augmentation**: None in code, but can add rotations/flips
- **Optimizer**: SGD with momentum 0.9, weight decay 0.0001
- **Learning Rate**: 0.001 with CosineAnnealingLR
- **Batch Size**: 2
- **Epochs**: 1 (demonstration; typically 12+ for convergence)

### Model Implementation

- **OrientedRCNN Class**: Inherits from nn.Module
- **Oriented Anchors**: Generated with different scales, ratios, angles
- **NMS**: Non-Maximum Suppression for oriented boxes
- **Evaluation**: IoU computation for oriented boxes

### Results

- **Training Loss Components**:
  - RPN Classification: Decreases as model learns proposals
  - RPN Regression: Refines oriented box predictions
  - RCNN Classification: Improves ship detection accuracy
  - RCNN Regression: Fine-tunes oriented boxes
- **Validation Loss**: Tracks training loss, indicates generalization
- **Checkpointing**: Saves best model and per-epoch checkpoints

### Challenges

- **Oriented IoU Computation**: More complex than axis-aligned
- **Anchor Generation**: Need oriented anchors at multiple angles
- **Training Stability**: Oriented regression can be unstable
- **Evaluation Metrics**: Standard COCO metrics adapted for oriented boxes

### Applications

- **Aerial/Satellite Imagery**: Ship detection in ports, oceans
- **Autonomous Vehicles**: Oriented vehicle detection
- **Industrial Inspection**: Oriented defect detection
- **Medical Imaging**: Oriented organ/tumor detection

## Files

- `code/`: PyTorch implementation of Oriented R-CNN
- `report/`: Detailed analysis and results
- `paper/`: Original Oriented R-CNN paper
- `description/`: Assignment description

## Conclusion

Oriented R-CNN extends traditional object detection to handle rotated objects, crucial for applications where orientation matters. The implementation demonstrates the architecture and training pipeline for oriented ship detection.
