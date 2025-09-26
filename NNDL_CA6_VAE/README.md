# NNDL_CA6_VAE

This folder contains the implementation of Variational Autoencoder (VAE). Part of Neural Networks and Deep Learning course assignment 6.

## Concepts Covered

### Autoencoders

Neural networks that learn efficient data representations by encoding and decoding.

### Variational Autoencoder (VAE)

Probabilistic version of autoencoder for generative modeling.

Key components:

- **Encoder**: Maps input to latent distribution parameters (mean, variance)
- **Latent Space**: Continuous, probabilistic representation
- **Decoder**: Reconstructs input from latent samples
- **Reparameterization Trick**: Enables gradient flow through sampling

### Loss Function

- **Reconstruction Loss**: Measures fidelity (MSE, BCE)
- **KL Divergence**: Regularizes latent space to follow standard normal

### Generative Capabilities

- Sample from latent space to generate new data
- Interpolation in latent space
- Anomaly detection via reconstruction error

### Applications

- Image generation
- Dimensionality reduction
- Data denoising
- Representation learning

## Implementation Details

### Dataset

- **Kvasir Dataset**: Gastrointestinal tract images
- **Normal Classes**: normal-z-line, normal-pylorus, normal-cecum
- **Anomaly Class**: polyps
- **Preprocessing**:
  - Center crop with random scale
  - Resize to 96x96
  - Random horizontal/vertical flips
  - Normalization to [0,1]

### VAE Architecture

#### Encoder

- **Input**: 96x96x3 images
- **Conv Layers**: 32→64→128→256 channels
- **Latent Space**: μ and σ vectors (dimension 128)
- **Reparameterization**: z = μ + σ \* ε, ε ~ N(0,1)

#### Decoder

- **Input**: Latent vector z
- **ConvTranspose Layers**: 256→128→64→32→3 channels
- **Output**: Reconstructed 96x96x3 image
- **Activation**: Sigmoid for [0,1] range

#### Loss Function

- **Reconstruction Loss**: MSE or BCE between input and output
- **KL Divergence**: Regularizes latent space to N(0,1)
- **Total Loss**: Recon + β \* KL (β=1.0)

### Training Parameters

- **Latent Dimension**: 128
- **Batch Size**: 128
- **Learning Rate**: 0.001 (Adam)
- **Epochs**: 100
- **β (KL Weight)**: 1.0

### Anomaly Detection Pipeline

1. Train VAE on normal images only
2. Compute reconstruction error for test images
3. Threshold on error for anomaly classification
4. Evaluate with accuracy and AUC-ROC

### Evaluation Metrics

- **Reconstruction Quality**: PSNR, SSIM on normal images
- **Anomaly Detection**: Accuracy, AUC-ROC, F1-score
- **Latent Space**: Distribution analysis

## Results

### VAE Training

- **MSE VAE**: Better pixel-level reconstruction
- **BCE VAE**: Better for binary anomaly detection
- **Latent Space**: Well-regularized to N(0,1)

### Anomaly Detection Performance

- **MSE VAE Classifier**: Accuracy ~0.85, AUC ~0.88
- **BCE VAE Classifier**: Accuracy ~0.87, AUC ~0.90
- **Threshold Selection**: ROC curve analysis for optimal threshold

### Qualitative Results

- **Normal Images**: Low reconstruction error, high PSNR
- **Polyp Images**: High reconstruction error, low PSNR
- **Generated Samples**: Realistic gastrointestinal images

### Ablation Studies

- **Latent Dimension**: Higher dim improves reconstruction but may overfit
- **Loss Weight β**: Too high β hurts reconstruction, too low hurts regularization
- **Data Augmentation**: Improves generalization to test variations

### Challenges Addressed

- **Class Imbalance**: Train on normal only, test on balanced set
- **Medical Variability**: Augmentation handles different angles/lighting
- **Evaluation**: Unsupervised metrics for anomaly detection
- **Interpretability**: Reconstruction error as anomaly score
