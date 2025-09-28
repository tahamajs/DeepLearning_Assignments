# NNDL_CA3_Oriented-R-CNN_implementation

This folder contains the implementation of Oriented R-CNN for oriented object detection, specifically for ship detection with rotated bounding boxes. Part of Neural Networks and Deep Learning course assignment 3.

## Concepts Covered

### Oriented Object Detection

Traditional object detection uses axis-aligned bounding boxes (AABBs), but many objects (ships, vehicles, airplanes) have arbitrary orientations. Oriented detection uses rotated rectangles defined by center coordinates, dimensions, and rotation angle.

#### Mathematical Representation

- **5-parameter representation**: (x_c, y_c, w, h, θ)

  - (x_c, y_c): Center coordinates
  - w, h: Width and height
  - θ: Rotation angle in radians (counter-clockwise from horizontal)

- **8-parameter representation**: (x1, y1, x2, y2, x3, y3, x4, y4)
  - Four corner coordinates of the rotated rectangle

#### Conversion Between Representations

To convert from 5-parameter to corners:

```
# Define rotation matrix
R = [[cosθ, -sinθ],
     [sinθ,  cosθ]]

# Corner offsets from center (before rotation)
corners = [[-w/2, -h/2], [w/2, -h/2], [w/2, h/2], [-w/2, h/2]]

# Apply rotation and translation
rotated_corners = [(x_c + R @ corner) for corner in corners]
```

### Oriented R-CNN Architecture

Oriented R-CNN extends Faster R-CNN to handle rotated bounding boxes throughout the pipeline.

#### Key Components:

1. **Backbone Network**:

   - **ResNet-50**: Deep residual network for feature extraction
   - **Feature Pyramid Network (FPN)**: Multi-scale feature maps
     - Combines features from different levels (P2-P6)
     - Lateral connections and top-down pathway
     - Enables detection at multiple scales

2. **Region Proposal Network (RPN)**:

   - **Oriented Anchors**: Pre-defined oriented boxes at multiple positions
     - Scales: [32, 64, 128, 256, 512]
     - Aspect ratios: [0.5, 1, 2]
     - Angles: [-90°, -45°, 0°, 45°, 90°] (5 orientations)
   - **Anchor Generation**: Places oriented anchors at each spatial position
   - **Classification Head**: Binary classification (object vs. background)
   - **Regression Head**: 5-parameter refinement (Δx, Δy, Δw, Δh, Δθ)

3. **Rotated ROI Align**:

   - **Standard ROI Align**: Bilinear sampling for axis-aligned regions
   - **Rotated Version**: Handles rotated regions using spatial transformer
   - **Feature Extraction**: Pools features from oriented proposal regions
   - **Output Size**: 7×7×256 features for each proposal

4. **RCNN Head**:
   - **Fully Connected Layers**: 1024 → 1024 neurons
   - **Classification Branch**: Predicts object classes
   - **Regression Branch**: Refines oriented bounding boxes
   - **Output**: Class probabilities and 5-parameter box refinements

#### Oriented Bounding Box Regression

Regression targets are computed relative to anchor boxes:

```
Δx = (x_gt - x_anchor) / w_anchor
Δy = (y_gt - y_anchor) / h_anchor
Δw = log(w_gt / w_anchor)
Δh = log(h_gt / h_anchor)
Δθ = θ_gt - θ_anchor  (normalized to [-π/2, π/2])
```

### Loss Functions

#### RPN Losses

- **Classification Loss**: Binary cross-entropy
  ```
  L_cls = -∑ (y log ŷ + (1-y) log(1-ŷ))
  ```
- **Regression Loss**: Smooth L1 loss for oriented boxes
  ```
  L_reg = ∑ smooth_L1(Δ - Δ̂)
  where smooth_L1(x) = 0.5x² if |x| < 1 else |x| - 0.5
  ```

#### RCNN Losses

- **Classification Loss**: Cross-entropy over classes
- **Regression Loss**: Smooth L1 for oriented box refinement

#### Total Loss

```
L_total = L_rpn_cls + λ1 L_rpn_reg + L_rcnn_cls + λ2 L_rcnn_reg
```

Typically λ1 = λ2 = 1

### Oriented IoU Computation

Computing IoU for rotated rectangles is more complex than axis-aligned boxes.

#### Algorithm:

1. **Polygon Intersection**: Convert both boxes to polygons
2. **Compute Intersection Area**: Using polygon clipping algorithms
3. **Compute Union Area**: Area1 + Area2 - Intersection
4. **IoU = Intersection / Union**

#### Implementation Details:

- Uses libraries like Shapely or custom polygon operations
- Handles edge cases (no intersection, containment)
- Numerically stable for small angles

### Training Details

#### Dataset: HRSC2016

- **High-Resolution Ship Collection 2016**
- **Statistics**: 1,061 images, ~2,000 ship instances
- **Resolution**: High-resolution aerial/satellite imagery
- **Annotations**: Oriented bounding boxes in XML format
- **Single Class**: All ships treated as one class

#### Data Pipeline

- **Custom Dataset Class**: Parses XML annotations
- **Preprocessing**: Resize images to 1024×1024
- **Normalization**: Mean subtraction, std division
- **Data Loading**: PyTorch DataLoader with batch size 2

#### Training Configuration

- **Optimizer**: SGD with momentum 0.9, weight decay 0.0001
- **Learning Rate**: Initial 0.001, CosineAnnealingLR schedule
- **Batch Size**: 2 (limited by GPU memory)
- **Epochs**: 12+ for full convergence (demo shows 1 epoch)
- **Gradient Clipping**: Prevents exploding gradients

#### Augmentation Strategies

- **Geometric**: Random horizontal/vertical flips
- **Photometric**: Brightness, contrast, saturation jitter
- **Rotational**: Small angle rotations (±15°)
- **Scaling**: Random scaling factors

### Model Implementation

#### OrientedRCNN Class

```python
class OrientedRCNN(nn.Module):
    def __init__(self):
        self.backbone = ResNet50_FPN()
        self.rpn = OrientedRPN()
        self.roi_align = RotatedROIAlign()
        self.head = RCNNHead()
```

#### Oriented Anchors

- Generated on feature maps at different levels
- Each anchor defined by (x, y, w, h, θ)
- Total anchors: ~100K per image

#### Non-Maximum Suppression (NMS)

- **Oriented NMS**: Suppresses overlapping oriented boxes
- **IoU Threshold**: 0.7 for ship detection
- **Score Threshold**: 0.05 minimum confidence

### Results and Evaluation

#### Training Dynamics

- **RPN Classification Loss**: Decreases rapidly as proposals improve
- **RPN Regression Loss**: Converges as oriented box predictions stabilize
- **RCNN Classification Loss**: Improves ship detection accuracy
- **RCNN Regression Loss**: Refines oriented box precision

#### Performance Metrics

- **Precision**: TP / (TP + FP)
- **Recall**: TP / (TP + FN)
- **AP (Average Precision)**: Area under precision-recall curve
- **AR (Average Recall)**: Recall at different IoU thresholds

#### Qualitative Results

- **Accurate Orientation**: Boxes align with ship headings
- **Scale Invariance**: Detects ships of various sizes
- **Context Awareness**: Distinguishes ships from land/water boundaries

### Challenges and Solutions

1. **Complex IoU Calculation**: Efficient polygon-based algorithms
2. **Anchor Design**: Multi-orientation anchors for coverage
3. **Training Instability**: Careful loss weighting and gradient clipping
4. **Evaluation Complexity**: Adapted COCO metrics for oriented boxes
5. **Data Scarcity**: Limited oriented datasets require augmentation

### Applications

- **Maritime Surveillance**: Ship detection and tracking in ports
- **Aerial Imagery Analysis**: Vehicle detection in overhead images
- **Industrial Inspection**: Oriented defect detection on rotated parts
- **Medical Imaging**: Oriented organ segmentation in CT/MRI
- **Autonomous Navigation**: Oriented obstacle detection for drones

## Files

- `code/NNDL_CA2_2.ipynb`: Complete PyTorch implementation
- `report/`: Performance analysis and ablation studies
- `paper/`: Original Oriented R-CNN paper (Yang et al.)
- `description/`: Assignment specifications and dataset details

## Key Learnings

1. Oriented detection requires modifications throughout the pipeline
2. Rotated ROI operations are computationally intensive
3. Anchor design significantly impacts performance
4. IoU computation for rotated boxes needs careful implementation
5. Multi-scale features are crucial for robust detection

## Conclusion

Oriented R-CNN extends traditional object detection to handle arbitrarily oriented objects, enabling accurate detection of ships, vehicles, and other oriented targets. The implementation demonstrates the complete pipeline from oriented anchors to rotated ROI operations, showcasing the complexities and solutions for oriented object detection in computer vision.
