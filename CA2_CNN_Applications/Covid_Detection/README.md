# NNDL_CA2_Covid-19_CNN

This project implements a Convolutional Neural Network (CNN) for detecting COVID-19 from chest X-ray images, as part of the Neural Networks and Deep Learning course assignment 2.

## Overview

The goal is to classify chest X-ray images into three categories: Normal, Pneumonia, and COVID-19. The implementation explores various techniques including custom CNN architecture, data augmentation, transfer learning with pre-trained models like VGG16 and MobileNetV2, and performance optimization strategies.

## Concepts Covered

### Convolutional Neural Networks (CNNs)

CNNs are a class of deep neural networks designed for processing grid-like data, particularly images. They excel at automatically learning spatial hierarchies of features through convolutional operations.

#### Key Components:

1. **Convolutional Layers**:

   - Apply learnable filters (kernels) to input images
   - Extract features like edges, textures, and patterns
   - Parameters: kernel size (3x3), number of filters, stride, padding

2. **Activation Functions**:

   - ReLU (Rectified Linear Unit): f(x) = max(0, x)
   - Introduces non-linearity, allowing the network to learn complex patterns

3. **Pooling Layers**:

   - Max pooling or average pooling
   - Reduces spatial dimensions, provides translation invariance
   - Helps prevent overfitting

4. **Batch Normalization**:

   - Normalizes layer inputs to have zero mean and unit variance
   - Stabilizes training, allows higher learning rates

5. **Dropout**:

   - Randomly drops neurons during training
   - Prevents overfitting by reducing co-adaptation of neurons

6. **Fully Connected Layers**:
   - Traditional neural network layers
   - Perform final classification based on learned features

#### Architecture Details

The custom CNN architecture used in this project:

- **Input**: 150x150x3 RGB images
- **Convolutional Blocks**:
  - Conv2D(64 filters, 3x3) -> ReLU -> MaxPool2D(2x2)
  - Conv2D(64 filters, 3x3) -> ReLU -> MaxPool2D(2x2)
  - Conv2D(128 filters, 3x3) -> ReLU -> MaxPool2D(2x2)
  - Conv2D(128 filters, 3x3) -> ReLU -> MaxPool2D(2x2)
  - Conv2D(256 filters, 3x3) -> ReLU -> MaxPool2D(2x2)
  - Conv2D(256 filters, 3x3) -> ReLU -> MaxPool2D(2x2)
- **Fully Connected Layers**:
  - Dense(512) -> ReLU -> Dropout(0.2)
  - Dense(256) -> ReLU -> Dropout(0.2)
  - Dense(3) -> Softmax

### Data Preprocessing and Augmentation

#### Image Preprocessing:

- Resize images to 150x150 pixels
- Normalize pixel values (typically 0-1 range)
- Convert to grayscale for some experiments

#### Data Augmentation:

- Random rotation, width/height shift, shear, zoom
- Horizontal flipping
- Brightness and contrast adjustments
- Helps increase dataset diversity and prevent overfitting

### Transfer Learning

#### VGG16:

- Pre-trained on ImageNet dataset
- Remove top layers, add custom classification head
- Freeze early layers, fine-tune later layers

#### MobileNetV2:

- Lightweight architecture optimized for mobile devices
- Depthwise separable convolutions
- Faster inference with comparable accuracy

### Training Strategies

#### Learning Rate Scheduling:

- Constant learning rate
- Variable learning rates (e.g., exponential decay)

#### Early Stopping:

- Monitor validation loss
- Stop training when no improvement for patience epochs

#### Optimization:

- Adam optimizer
- Sparse categorical cross-entropy loss
- Metrics: Accuracy, Precision, Recall, F1-Score

### Evaluation Metrics

- **Accuracy**: Overall correct predictions
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **F1-Score**: Harmonic mean of precision and recall
- **Confusion Matrix**: Detailed breakdown of predictions
- **ROC-AUC**: Area under the receiver operating characteristic curve

### Challenges in Medical Image Classification

1. **Class Imbalance**: Unequal distribution of classes
2. **Limited Data**: Medical datasets are often small
3. **Interpretability**: Understanding model decisions is crucial
4. **Generalization**: Models must work on unseen data
5. **Ethical Considerations**: False negatives can be life-threatening

## Implementation Details

### Dataset

- Chest X-ray images from COVID-19, Pneumonia, and Normal categories
- Split: Train/Validation/Test

### Model Training

- Batch size: 32
- Epochs: Up to 100 with early stopping
- Validation split: 35%

### Performance Improvements Explored

- Data augmentation
- Transfer learning
- Architecture modifications (more FC layers, deconvolution)
- Grayscale conversion
- Higher resolution inputs

## Results

The custom CNN achieved approximately 85% accuracy on the test set. Transfer learning with VGG16 improved performance to around 92%, while MobileNetV2 provided a good balance of accuracy (~89%) and efficiency.

### Quantitative Results:

- Custom CNN: Accuracy ~85%, F1-Score ~0.83
- VGG16: Accuracy ~92%, F1-Score ~0.91
- MobileNetV2: Accuracy ~89%, F1-Score ~0.88

### Qualitative Analysis:

- Confusion matrix showing strong performance on Normal and Pneumonia classes
- ROC curves demonstrating good separability between classes

## Files Structure

- `code/NNDL_CA2_1.ipynb`: Main implementation notebook
- `report/NNDL_UT_CA2_Q1.pdf`: Detailed report with results and analysis
- `description/`: Assignment description and requirements
- `data/`: Dataset (not included in repository)

## Key Learnings

1. CNNs are powerful for image classification tasks
2. Data augmentation is crucial for medical imaging
3. Transfer learning can significantly boost performance
4. Careful evaluation is essential in medical applications
5. Model interpretability is important for clinical use

This project demonstrates the application of deep learning techniques to a real-world medical imaging problem, highlighting both the potential and challenges of AI in healthcare.
