# NNDL_CA2_vehicle_classificattion_CNN

This project implements Convolutional Neural Networks (CNNs) for vehicle classification, as part of the Neural Networks and Deep Learning course assignment 2. The implementation explores custom CNN architectures, transfer learning with pre-trained models (VGG16, AlexNet), and traditional machine learning approaches using SVM on extracted features.

## Overview

The goal is to classify images of vehicles into specific categories (e.g., different Toyota models like Corolla, Camry, Rav4, etc.). The project compares deep learning approaches with classical methods, evaluates the impact of data augmentation, and demonstrates feature extraction techniques.

## Concepts Covered

### Convolutional Neural Networks (CNNs)

CNNs are specialized for image processing tasks, learning hierarchical feature representations through convolutional operations.

#### Key Components:

1. **Convolutional Layers**:

   - Learn spatial features using filters/kernels
   - Parameters: kernel size, number of filters, stride, padding
   - Output: feature maps highlighting different patterns

2. **Activation Functions**:

   - ReLU: Introduces non-linearity, helps with vanishing gradients
   - Formula: f(x) = max(0, x)

3. **Pooling Layers**:

   - Max pooling: Retains maximum values in regions
   - Average pooling: Computes average values
   - Reduces spatial dimensions, provides translation invariance

4. **Fully Connected Layers**:

   - Traditional neural network layers for classification
   - Connect every neuron to every neuron in the previous layer

5. **Batch Normalization**:
   - Normalizes layer inputs for stable training
   - Allows higher learning rates and acts as regularization

### Transfer Learning

#### Pre-trained Models:

- **VGG16**: 16-layer network trained on ImageNet

  - Deep architecture with small 3x3 convolutions
  - Strong feature extraction capabilities

- **AlexNet**: 8-layer network, winner of ImageNet 2012
  - Introduced ReLU, dropout, and data augmentation
  - Uses larger filters in early layers

#### Fine-tuning Strategy:

- Freeze early layers (preserve general features)
- Fine-tune later layers for specific task
- Replace final classification layer

### Feature Extraction for Classical ML

#### CNN as Feature Extractor:

- Use pre-trained CNN to extract features from images
- Remove classification head
- Feed features into traditional classifiers like SVM

#### Support Vector Machines (SVM):

- Finds optimal hyperplane for classification
- Kernel trick for non-linear separability
- Different kernels: Linear, RBF, Polynomial

### Data Augmentation

#### Techniques:

- **Geometric transformations**: Rotation, translation, scaling, flipping
- **Color transformations**: Brightness, contrast, saturation adjustments
- **Noise injection**: Adding random noise to images

#### Benefits:

- Increases dataset diversity
- Prevents overfitting
- Improves generalization

### Custom CNN Architecture

The project implements a custom CNN (ToyotaModelCNN) with:

- Multiple convolutional blocks
- Increasing filter sizes (32 → 64 → 128 → 256)
- Max pooling after each block
- Dropout for regularization
- Fully connected layers for classification

### Training and Optimization

#### Loss Functions:

- Cross-entropy loss for multi-class classification
- Formula: L = -∑ y_i log(ŷ_i)

#### Optimizers:

- Adam: Adaptive moment estimation
- Combines benefits of AdaGrad and RMSProp

#### Regularization:

- Dropout: Randomly drops neurons during training
- Weight decay (L2 regularization)

### Evaluation Metrics

#### Classification Metrics:

- **Accuracy**: (TP + TN) / (TP + TN + FP + FN)
- **Precision**: TP / (TP + FP)
- **Recall**: TP / (TP + FN)
- **F1-Score**: 2 × (Precision × Recall) / (Precision + Recall)

#### Confusion Matrix:

- Shows prediction distribution across classes
- Helps identify misclassification patterns

### Challenges in Fine-grained Classification

1. **Intra-class Variation**: High similarity between classes
2. **Inter-class Similarity**: Different classes may look similar
3. **Limited Data**: Fine-grained datasets are often small
4. **Class Imbalance**: Unequal samples per class

## Implementation Details

### Dataset

- Toyota vehicle images (10 classes: Corolla, Camry, Rav4, Tacoma, etc.)
- Image preprocessing: Resize to 224x224, normalization
- Balanced sampling to handle class imbalance

### Model Architectures

#### Custom CNN:

```python
class ToyotaModelCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv4 = nn.Conv2d(128, 256, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(256 * 14 * 14, 512)
        self.fc2 = nn.Linear(512, 10)
        self.dropout = nn.Dropout(0.5)
```

#### Transfer Learning Models:

- VGG16 with custom classifier
- AlexNet with custom classifier

#### SVM Classifiers:

- Linear SVM on extracted features
- RBF kernel SVM

### Training Configuration

- Batch size: 32
- Learning rate: 0.001 (Adam)
- Epochs: 50-100 with early stopping
- Data augmentation: Random horizontal flip, rotation

### Feature Extraction Pipeline

1. Load pre-trained CNN (VGG16/AlexNet)
2. Remove classification layers
3. Forward pass images to get feature vectors
4. Train SVM on extracted features

## Results

### Quantitative Results:

- **Custom CNN**: Accuracy ~85%, F1-Score ~0.84
- **VGG16 Fine-tuned**: Accuracy ~92%, F1-Score ~0.91
- **AlexNet Fine-tuned**: Accuracy ~89%, F1-Score ~0.88
- **VGG16 + SVM**: Accuracy ~87%, F1-Score ~0.86
- **AlexNet + SVM**: Accuracy ~85%, F1-Score ~0.84

### Qualitative Analysis:

- Confusion matrices show best performance on distinct models (Camry, Tacoma)
- Some confusion between similar-looking vehicles (Corolla vs. Camry)
- Data augmentation improved validation accuracy by ~5%

### Model Comparison:

- Transfer learning outperforms custom CNN
- Fine-tuning slightly better than feature extraction + SVM
- VGG16 generally performs better than AlexNet

## Key Learnings

1. Transfer learning significantly boosts performance on limited datasets
2. Fine-tuning pre-trained models is effective for image classification
3. CNN feature extraction + SVM is a viable alternative to end-to-end training
4. Data augmentation is crucial for preventing overfitting
5. Fine-grained classification requires careful model design and data handling

## Files Structure

- `code/NNDL_CA2_2.ipynb`: Main implementation with CNN models and SVM
- `code/NNDL_CA2_2_normalized.ipynb`: Alternative implementation
- `report/NNDL_UT_CA2_Q2.pdf`: Detailed analysis and results
- `description/`: Assignment requirements

This project demonstrates the power of deep learning for image classification while comparing modern and traditional approaches, highlighting the importance of transfer learning in computer vision tasks.
