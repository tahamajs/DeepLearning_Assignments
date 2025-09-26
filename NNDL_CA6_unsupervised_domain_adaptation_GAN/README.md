# NNDL_CA6_unsupervised_domain_adaptation_GAN

This folder contains the implementation of unsupervised domain adaptation using Generative Adversarial Networks (GANs). Part of Neural Networks and Deep Learning course assignment 6.

## Concepts Covered

### Domain Adaptation

Transferring knowledge from source domain to target domain with different distributions.

### Unsupervised Domain Adaptation

- Source domain: Labeled data
- Target domain: Unlabeled data
- Goal: Learn domain-invariant features

### Generative Adversarial Networks (GANs)

- **Generator**: Transforms source images to target domain
- **Discriminator**: Distinguishes real target vs. generated images
- **Adversarial Training**: Min-max game between G and D

### Domain Confusion

- Feature alignment between source and target domains
- Domain classifier tries to distinguish domains
- Feature extractor learns domain-invariant representations

## Implementation Details

### Dataset

- **Source Domain**: MNIST (grayscale handwritten digits)
- **Target Domain**: MNIST-M (MNIST digits blended with color patches)
- **Task**: Digit classification (0-9)
- **Data Split**: 80% train, 20% test per domain

### Model Architecture

#### Generator

- **Input**: Source image + noise vector
- **Architecture**:
  - Encoder: Conv layers downsampling to latent space
  - Decoder: ConvTranspose layers upsampling with skip connections
  - Noise injection for style transfer
- **Output**: Generated image in target domain style

#### Discriminator

- **Input**: Real target or generated image
- **Architecture**:
  - Conv layers with LeakyReLU
  - Patch-based discrimination
  - Spectral normalization for stability
- **Output**: Probability of being real target image

#### Classifier

- **Input**: Image features
- **Architecture**:
  - Shared feature extractor (ResNet-like)
  - Classification head for digits
- **Training**: Source labels + domain adaptation loss

### Training Objectives

- **Adversarial Loss**: G tries to fool D, D tries to distinguish
- **Classification Loss**: Correct digit prediction on source
- **Domain Confusion Loss**: Align source and target features
- **Reconstruction Loss**: Optional cycle consistency

### Training Parameters

- **Noise Dimension**: 10
- **Batch Size**: 64
- **Learning Rate**: 0.001 with decay
- **Beta**: (0.5, 0.999) for Adam
- **Epochs**: 20
- **Loss Weights**: λ_adv=0.01, λ_cls=0.013, λ_src=0.011

### Evaluation Metrics

- **Source Accuracy**: Performance on MNIST
- **Target Accuracy**: Performance on MNIST-M
- **Generated Quality**: Visual inspection of generated images
- **Domain Invariance**: Feature similarity between domains

## Results

### Model Performance

- **Source Domain (MNIST)**: Accuracy ~0.95, F1 ~0.95
- **Target Domain (MNIST-M)**: Accuracy ~0.88, F1 ~0.87
- **Generated Images**: Accuracy ~0.85, F1 ~0.84

### Training Dynamics

- **Discriminator Loss**: Decreases as it learns to distinguish domains
- **Generator Loss**: Fluctuates during adversarial training
- **Classifier Loss**: Steady decrease on source domain
- **Target Accuracy**: Improves as domain adaptation progresses

### Qualitative Results

- **Generated Images**: Realistic MNIST-M style digits
- **Class-wise Generation**: Different noise vectors produce varied styles
- **Domain Transfer**: Source images successfully transformed to target appearance

### Ablation Studies

- **Without GAN**: Target accuracy ~0.75 (no adaptation)
- **Without Noise**: Less diverse generations, slightly lower target accuracy
- **Different Architectures**: Deeper networks improve quality but slower training

### Challenges Addressed

- **Mode Collapse**: Spectral norm and diverse noise prevent collapse
- **Training Instability**: Careful loss balancing and scheduling
- **Evaluation**: Unsupervised metrics for adaptation quality
- **Computational Cost**: Efficient architecture for real-time adaptation

## Applications

- **Medical Imaging**: Adapting models across hospitals/scanners
- **Autonomous Driving**: Transferring between weather conditions
- **Face Recognition**: Cross-ethnicity adaptation
- **Industrial Inspection**: Adapting to different manufacturing lines

## Files

- `code/`: PyTorch implementation of GAN-based domain adaptation
- `report/`: Detailed analysis with generated samples and metrics
- `paper/`: Domain adaptation and GAN papers
- `description/`: Assignment description

## Conclusion

The GAN-based approach successfully adapts digit classification from clean MNIST to cluttered MNIST-M, demonstrating effective unsupervised domain adaptation through adversarial training and feature alignment.
