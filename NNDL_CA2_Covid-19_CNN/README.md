# NNDL_CA2_Covid-19_CNN

This folder contains the implementation of a Convolutional Neural Network (CNN) for detecting COVID-19 from chest X-ray images. The project is part of the Neural Networks and Deep Learning course assignment 2.

## Concepts Covered

### Convolutional Neural Networks (CNNs)

CNNs are specialized neural networks designed for processing grid-like data, such as images. They use convolutional layers to automatically learn spatial hierarchies of features from input images.

Key components:

- **Convolutional Layers**: Apply filters to extract features like edges, textures, and patterns.
- **Pooling Layers**: Reduce spatial dimensions and computational complexity while retaining important features.
- **Fully Connected Layers**: Perform classification based on learned features.
- **Activation Functions**: Introduce non-linearity (e.g., ReLU).

### COVID-19 Detection

The model is trained to classify chest X-ray images into categories such as:

- Normal
- COVID-19 positive
- Other pneumonia cases

### Data Preprocessing

- Image resizing and normalization
- Data augmentation (rotation, flipping, scaling) to increase dataset diversity
- Train-validation-test split

### Model Architecture

Typically includes:

- Multiple convolutional blocks with increasing filter sizes
- Batch normalization for stable training
- Dropout for regularization
- Softmax output for multi-class classification

### Training and Evaluation

- Loss function: Categorical Cross-Entropy
- Optimizer: Adam or SGD
- Metrics: Accuracy, Precision, Recall, F1-Score
- Confusion matrix analysis

### Challenges and Considerations

- Class imbalance in medical datasets
- Overfitting due to limited data
- Interpretability of CNN decisions
- Generalization to new data

## Files

- `code/`: Contains the main implementation notebook or scripts
- Other files: Dataset, models, results

## Results

The model achieves [insert accuracy] on the test set, demonstrating the effectiveness of CNNs in medical image classification.
