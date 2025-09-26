# NNDL_CA6_unsupervised_domain_adaptation_GAN

This folder contains the implementation of unsupervised domain adaptation using Generative Adversarial Networks (GANs). Part of Neural Networks and Deep Learning course assignment 6.

## Concepts Covered

### Domain Adaptation

Adapting a model trained on source domain to target domain without target labels.

### Unsupervised Domain Adaptation

No labeled data in target domain.

### GANs in Domain Adaptation

- **Generator**: Transforms source to target domain
- **Discriminator**: Distinguishes real target from generated
- **Feature Extractor**: Learns domain-invariant features

### Methods

- **Domain-Adversarial Training**: Adversarial loss for domain confusion
- **CycleGAN**: Cycle consistency for unpaired data
- **Pixel-level adaptation**: Image-to-image translation

### Training

- Adversarial loss: Generator vs. Discriminator
- Task loss: Classification on source
- Consistency losses: Cycle, identity

### Evaluation

- Target domain accuracy
- Domain confusion metrics
- Qualitative: Transformed images

### Challenges

- Mode collapse in GANs
- Balancing adaptation and task performance
- Generalization to unseen domains

### Applications

- Medical imaging (different scanners)
- Autonomous driving (weather conditions)
- Sentiment analysis (different text domains)

## Files

- `code/`: GAN-based adaptation
- Source and target datasets

## Results

The model adapts features across domains, improving performance on unlabeled target data.
