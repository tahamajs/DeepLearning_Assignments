# NNDL_CA3_Oriented-R-CNN_implementation

This folder contains the implementation of Oriented R-CNN for detecting oriented objects in images. Part of Neural Networks and Deep Learning course assignment 3.

## Concepts Covered

### Oriented Object Detection

Unlike axis-aligned bounding boxes, oriented detection handles rotated objects with arbitrary orientations.

### Oriented R-CNN Architecture

Extends Faster R-CNN for oriented bounding boxes.

Key components:

- **Backbone**: Feature extraction (e.g., ResNet with FPN)
- **Region Proposal Network (RPN)**: Generates oriented proposals
- **Rotated RoI Align**: Extracts features from oriented regions
- **Oriented RCNN Head**: Predicts oriented bounding boxes and classes

### Oriented Bounding Boxes

Represented by 6 parameters: center (x,y), width, height, and orientation offsets (alpha, beta).

### Training

- Loss functions: Classification loss, regression loss for oriented boxes
- Data: Datasets with oriented annotations (e.g., HRSC2016 for ships)
- Evaluation: Mean Average Precision (mAP) for oriented detection

### Challenges

- More complex than axis-aligned detection
- Requires specialized RoI operations
- Handling rotation invariance

### Applications

- Aerial imagery analysis (ships, vehicles)
- Scene text detection
- Industrial inspection

## Files

- `code/`: Oriented R-CNN implementation
- Dataset: HRSC2016 or similar

## Results

The model successfully detects oriented objects, improving upon traditional axis-aligned methods for rotated objects.
