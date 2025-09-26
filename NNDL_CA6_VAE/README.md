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

### Variants

- Conditional VAE (CVAE)
- β-VAE for disentangled representations
- VAE-GAN hybrids

## Files

- `code/`: VAE implementation
- Dataset: MNIST, CelebA, etc.

## Results

VAE learns smooth latent representations, enabling generation of diverse, realistic samples.
