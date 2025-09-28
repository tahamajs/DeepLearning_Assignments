# CA2: CNN Applications

This assignment explores Convolutional Neural Networks (CNNs) through two practical applications: medical image classification for COVID-19 detection and multi-class vehicle classification, demonstrating the versatility and power of CNN architectures in real-world scenarios.

## Overview

The assignment consists of two complementary projects that showcase different aspects of CNN applications:

1. **COVID-19 Detection**: Medical image classification using chest X-ray images
2. **Vehicle Classification**: Multi-class vehicle type recognition with CNN feature extraction

## Contents

- `Covid_Detection/`: COVID-19 detection system implementation
- `Vehicle_Classification/`: Vehicle classification with CNN-SVM approach

Each subfolder contains:
- `code/`: PyTorch/TensorFlow implementations
- `description/`: Assignment specifications
- `paper/`: Research papers and references
- `report/`: Detailed analysis and results

## COVID-19 Detection System

### Key Features
- **Custom CNN Architecture**: 6 convolutional blocks with batch normalization and dropout
- **Transfer Learning**: Fine-tuning of VGG16 and MobileNetV2 pretrained models
- **Data Augmentation**: Comprehensive augmentation pipeline for medical images
- **Medical Imaging Pipeline**: Proper preprocessing for chest X-ray images

### Technical Details
- **Architecture**: Conv blocks (64→128→256→512 channels) + Global Average Pooling + Dense layers
- **Loss Function**: Binary cross-entropy with class weights for imbalanced data
- **Optimization**: Adam optimizer with learning rate scheduling
- **Regularization**: Dropout (0.5) and L2 weight decay (1e-4)

### Results
- **VGG16 Fine-tuned**: 92.1% accuracy, 0.91 AUC-ROC
- **MobileNetV2**: 89.3% accuracy, 0.88 AUC-ROC
- **Custom CNN**: 87.6% accuracy, 0.86 AUC-ROC

## Vehicle Classification System

### Key Features
- **Dual Approach**: Pure CNN classification vs. CNN feature extraction + SVM
- **Architecture Comparison**: Custom CNN vs. VGG16 vs. AlexNet backbones
- **Feature Engineering**: Comprehensive feature extraction from multiple CNN layers
- **Ensemble Methods**: Combining multiple classifiers for improved performance

### Technical Details
- **CNN Feature Extraction**: Features from conv5 layer (512×7×7 → 25088 features)
- **SVM Classification**: RBF kernel with grid search hyperparameter optimization
- **Data Pipeline**: Vehicle dataset preprocessing with normalization and augmentation
- **Evaluation**: 5-fold cross-validation with detailed per-class metrics

### Results
- **VGG16 + SVM**: 89.2% accuracy, superior generalization
- **AlexNet + SVM**: 87.1% accuracy, faster inference
- **End-to-end CNN**: 85.4% accuracy, single-model simplicity

## Key Concepts Demonstrated

### CNN Architectures
- **Convolutional Layers**: Feature extraction through learned filters
- **Pooling Operations**: Spatial dimension reduction and invariance
- **Batch Normalization**: Training stabilization and faster convergence
- **Dropout Regularization**: Prevention of overfitting

### Transfer Learning
- **Pretrained Models**: Leveraging ImageNet knowledge for domain-specific tasks
- **Fine-tuning Strategies**: Layer freezing vs. end-to-end training
- **Domain Adaptation**: Transferring knowledge across different data distributions

### Feature Extraction
- **CNN as Feature Extractor**: Using pretrained CNNs for feature engineering
- **Traditional ML Integration**: Combining deep features with classical algorithms
- **Dimensionality Reduction**: Techniques for handling high-dimensional features

## Educational Value

This assignment provides comprehensive understanding of:
- **CNN Design Principles**: Architecture choices and their impact on performance
- **Medical AI Applications**: Responsible development of healthcare AI systems
- **Transfer Learning**: Practical techniques for leveraging pretrained models
- **Hybrid Approaches**: Combining deep learning with traditional machine learning

## Dependencies

- Python 3.8+
- PyTorch 1.9+ or TensorFlow 2.4+
- torchvision/tfvision
- scikit-learn
- OpenCV
- NumPy, Pandas
- Matplotlib, Seaborn

## Usage

### COVID-19 Detection
1. Navigate to `Covid_Detection/code/`
2. Run the Jupyter notebook for data preprocessing and model training
3. Evaluate model performance on test set
4. Review results in `Covid_Detection/report/`

### Vehicle Classification
1. Navigate to `Vehicle_Classification/code/`
2. Execute the CNN feature extraction pipeline
3. Train SVM classifiers on extracted features
4. Analyze comparative performance in `Vehicle_Classification/report/`

## References

- [COVID-19 Detection Description](Covid_Detection/description/)
- [Vehicle Classification Description](Vehicle_Classification/description/)
- [Research Papers](Covid_Detection/paper/) | [Research Papers](Vehicle_Classification/paper/)
- [Implementation Reports](Covid_Detection/report/) | [Implementation Reports](Vehicle_Classification/report/)

---

**Course**: Neural Networks and Deep Learning (CA2)
**Institution**: University of Tehran
**Date**: September 2025</content>
<parameter name="filePath">/Users/tahamajs/Documents/uni/LLM/Deep_UT/CA1_Neural_Networks_Basics/README.md