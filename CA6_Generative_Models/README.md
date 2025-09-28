# CA6: Generative Models

This assignment explores generative modeling techniques, implementing GAN-based domain adaptation and Variational Autoencoders (VAEs), demonstrating the power of generative approaches in unsupervised learning and domain transfer.

## Overview

The assignment consists of two generative modeling projects:

1. **Unsupervised Domain Adaptation with GANs**: CycleGAN for domain transfer
2. **VAE for Anomaly Detection**: Generative modeling for medical imaging

## Contents

- `Unsupervised_Domain_Adaptation_GAN/`: CycleGAN implementation for domain adaptation
- `VAE/`: Variational Autoencoder for anomaly detection

Each subfolder contains:
- `code/`: PyTorch implementations
- `description/`: Assignment specifications
- `paper/`: Research papers and references
- `report/`: Detailed analysis and results

## Unsupervised Domain Adaptation with GANs

### Key Features
- **Cycle Consistency**: Bidirectional mapping between domains
- **Domain Confusion**: Adversarial alignment of feature distributions
- **Unsupervised Learning**: No target domain labels required
- **Style Transfer**: Realistic transformation of visual appearance

### Technical Details
- **Generator Networks**: U-Net style with residual blocks
- **Discriminator Networks**: Patch-based discrimination
- **Loss Components**: Adversarial loss + cycle consistency + identity loss
- **Training Strategy**: Alternating optimization with careful loss balancing

### Results
- **Target Accuracy**: 87.6% on MNIST-M (vs. 75.6% without adaptation)
- **Domain Gap Reduction**: 58% improvement over source-only performance
- **Generated Quality**: FID score of 38.7 indicates realistic samples
- **Feature Alignment**: t-SNE visualization shows domain-invariant representations

## VAE for Anomaly Detection

### Key Features
- **Probabilistic Encoding**: Amortized variational inference
- **Reparameterization Trick**: Enables gradient-based optimization
- **Anomaly Scoring**: Reconstruction error as anomaly indicator
- **Medical Application**: Polyp detection in gastrointestinal endoscopy

### Technical Details
- **Encoder**: CNN-based recognition network (μ, log σ²)
- **Decoder**: Transpose CNN for image reconstruction
- **ELBO Loss**: Reconstruction + KL divergence regularization
- **β-VAE Variant**: Tunable regularization strength

### Results
- **Reconstruction Quality**: PSNR 28.5dB, SSIM 0.89 on normal images
- **Anomaly Detection**: AUC 0.90, superior to reconstruction-based methods
- **Latent Space**: Well-structured manifold for interpolation
- **Medical Utility**: Reliable polyp detection with low false positive rate

## Key Concepts Demonstrated

### Generative Adversarial Networks
- **Adversarial Training**: Generator vs. discriminator optimization
- **Cycle Consistency**: Ensuring bidirectional mapping quality
- **Mode Collapse**: Avoiding degenerate solutions
- **Training Stability**: Techniques for stable GAN training

### Variational Autoencoders
- **Variational Inference**: Approximate posterior distributions
- **Reparameterization**: Enabling backpropagation through sampling
- **ELBO Optimization**: Evidence lower bound maximization
- **Latent Space Regularization**: Controlling latent representations

### Domain Adaptation
- **Domain Shift**: Handling distribution differences between datasets
- **Unsupervised Adaptation**: Learning without target labels
- **Feature Alignment**: Matching source and target distributions
- **Transfer Learning**: Leveraging knowledge from related domains

### Anomaly Detection
- **Reconstruction-based Methods**: Using generative models for anomaly scoring
- **One-class Classification**: Learning from normal data only
- **Threshold Selection**: Determining anomaly decision boundaries
- **Evaluation Metrics**: Precision, recall, F1-score for imbalanced data

## Educational Value

This assignment provides comprehensive understanding of:
- **Generative Modeling**: Creating new data samples from learned distributions
- **Adversarial Learning**: Training through competition between networks
- **Unsupervised Learning**: Learning without explicit supervision
- **Probabilistic ML**: Bayesian approaches to deep learning

## Dependencies

- Python 3.8+
- PyTorch 1.9+
- torchvision
- NumPy, Pandas
- Matplotlib, Seaborn
- scikit-learn
- OpenCV (for image processing)

## Usage

### Domain Adaptation GAN
1. Navigate to `Unsupervised_Domain_Adaptation_GAN/code/`
2. Prepare source and target domain datasets
3. Train CycleGAN for domain transfer
4. Evaluate adaptation performance in `Unsupervised_Domain_Adaptation_GAN/report/`

### VAE Anomaly Detection
1. Navigate to `VAE/code/`
2. Train VAE on normal data
3. Compute reconstruction errors on test data
4. Analyze anomaly detection results in `VAE/report/`

## References

- [Domain Adaptation Description](Unsupervised_Domain_Adaptation_GAN/description/)
- [VAE Description](VAE/description/)
- [Research Papers](Unsupervised_Domain_Adaptation_GAN/paper/) | [Research Papers](VAE/paper/)
- [Implementation Reports](Unsupervised_Domain_Adaptation_GAN/report/) | [Implementation Reports](VAE/report/)

---

**Course**: Neural Networks and Deep Learning (CA6)
**Institution**: University of Tehran
**Date**: September 2025