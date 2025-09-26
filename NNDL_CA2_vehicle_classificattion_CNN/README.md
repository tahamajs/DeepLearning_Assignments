# NNDL_CA2_vehicle_classificattion_CNN

This folder contains the implementation of a Convolutional Neural Network (CNN) for vehicle classification. The project is part of the Neural Networks and Deep Learning course assignment 2.

## Concepts Covered

### Convolutional Neural Networks (CNNs)

CNNs are powerful for image classification tasks. They learn hierarchical features from images through convolutional operations.

Key elements:

- **Feature Extraction**: Convolutional layers detect local patterns
- **Spatial Invariance**: Pooling layers make features translation-invariant
- **Deep Architecture**: Stacking layers for complex feature learning

### Vehicle Classification

The model classifies images of vehicles into categories such as:

- Cars
- Trucks
- Buses
- Motorcycles
- etc.

### Data Handling

- Image preprocessing: resizing, normalization
- Augmentation techniques: random crops, color jittering
- Handling class imbalance if present

### Model Design

- Input: RGB images
- Convolutional blocks with increasing depth
- Global average pooling before classification
- Output: Probability distribution over vehicle classes

### Training Process

- Loss: Cross-entropy for multi-class classification
- Optimization: Adaptive optimizers like Adam
- Regularization: Dropout, weight decay
- Learning rate scheduling

### Evaluation Metrics

- Top-1 and Top-5 accuracy
- Confusion matrix
- Per-class precision and recall

### Applications

- Autonomous driving systems
- Traffic monitoring
- Vehicle inventory management

## Files

- `code/`: Implementation scripts/notebooks
- Dataset and model files

## Results

The CNN achieves high accuracy in classifying various vehicle types, showcasing CNNs' capability in fine-grained image classification.
