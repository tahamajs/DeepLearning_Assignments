# CA7: Advanced Topics

This assignment covers advanced topics in deep learning, including comparative adversarial analysis of CNNs vs. ViTs and multilingual image captioning in Persian, exploring cutting-edge research areas and practical extensions.

## Overview

The assignment consists of two advanced projects:

1. **CNN vs. ViT Adversarial Attacks**: Comparative robustness analysis
2. **Multilingual Image Captioning**: Persian language captioning system

## Contents

- `CNN_VIT_Adversarial_Attack/`: Comparative adversarial analysis
- `Image_Captioning/`: Persian image captioning implementation

Each subfolder contains:

- `code/`: PyTorch implementations
- `description/`: Assignment specifications
- `paper/`: Research papers and references
- `report/`: Detailed analysis and results

## Comparative Adversarial Analysis: CNNs vs. ViTs

### Key Features

- **Architecture Comparison**: ResNet-50 vs. ViT-Base side-by-side analysis
- **Attack Suite**: FGSM, PGD, CW attacks with multiple strengths
- **Defense Evaluation**: Adversarial training and input preprocessing
- **Robustness Metrics**: Detailed analysis of clean vs. robust performance

### Technical Details

- **CNN Model**: ResNet-50 with 25M parameters
- **ViT Model**: 12-layer transformer with 86M parameters
- **Attack Implementation**: Torchattacks library with custom modifications
- **Defense Methods**: Adversarial training with PGD-based augmentation

### Results

- **Clean Performance**: ViT 84.7% vs. ResNet 76.2% accuracy
- **Adversarial Robustness**: ViT 57.4% vs. ResNet 52.1% under strong attacks
- **Attack Transferability**: High transfer rate between architectures
- **Computational Trade-offs**: ViT requires more compute but offers better robustness

## Multilingual Image Captioning in Persian

### Key Features

- **Persian NLP Pipeline**: Hazm library for tokenization and normalization
- **Multilingual Attention**: Multi-head attention for cross-modal alignment
- **RTL Text Handling**: Proper bidirectional text processing
- **Cultural Adaptation**: Persian-specific caption generation

### Technical Details

- **Text Processing**: Persian normalization, word tokenization, vocabulary building
- **Model Architecture**: Transformer-based encoder-decoder with Persian embeddings
- **Beam Search**: Multilingual beam search with Persian language model
- **Evaluation**: BLEU scores adapted for Persian morphological complexity

### Results

- **BLEU-4 Score**: 0.195 (competitive for low-resource language)
- **Persian Fluency**: Natural Persian sentence generation
- **Cultural Relevance**: Captions reflect Persian linguistic and cultural context
- **Multilingual Capability**: Framework extensible to other RTL languages

## Key Concepts Demonstrated

### Adversarial Robustness

- **Architecture-dependent Vulnerabilities**: How model architecture affects robustness
- **Attack Transferability**: Cross-architecture attack effectiveness
- **Defense Strategies**: Architecture-specific defense mechanisms
- **Robustness-Accuracy Trade-offs**: Balancing clean performance and adversarial defense

### Multilingual Deep Learning

- **Low-resource Languages**: Challenges in Persian NLP
- **Cross-lingual Transfer**: Leveraging English resources for Persian
- **Cultural Adaptation**: Language-specific model adjustments
- **RTL Processing**: Handling right-to-left text in neural models

### Comparative Architecture Analysis

- **CNN vs. Transformer**: Strengths and weaknesses of different architectures
- **Scalability Analysis**: How performance scales with model/data size
- **Computational Efficiency**: Training and inference cost comparisons
- **Task-specific Suitability**: When to use CNNs vs. Transformers

### Advanced Evaluation

- **Multilingual Metrics**: Adapting evaluation metrics for different languages
- **Cultural Bias Assessment**: Ensuring fair and culturally appropriate outputs
- **Robustness Benchmarks**: Standardized evaluation of adversarial defenses
- **Cross-domain Generalization**: Performance across different data distributions

## Educational Value

This assignment provides advanced understanding of:

- **Model Security**: Adversarial attacks and defenses in deep learning
- **Multilingual AI**: Building AI systems for non-English languages
- **Architecture Design**: Choosing appropriate models for specific tasks
- **Research Methodologies**: Comparative analysis and benchmarking

## Dependencies

- Python 3.8+
- PyTorch 1.9+
- torchvision
- transformers
- Hazm (Persian NLP)
- torchattacks
- NumPy, Pandas
- Matplotlib, Seaborn

## Usage

### Adversarial Analysis

1. Navigate to `CNN_VIT_Adversarial_Attack/code/`
2. Train baseline CNN and ViT models
3. Generate adversarial examples with various attacks
4. Compare robustness and analyze results in `CNN_VIT_Adversarial_Attack/report/`

### Persian Captioning

1. Navigate to `Image_Captioning/code/`
2. Process Persian text data with Hazm
3. Train multilingual captioning model
4. Generate and evaluate Persian captions in `Image_Captioning/report/`

## References

- [Adversarial Analysis Description](CNN_VIT_Adversarial_Attack/description/)
- [Persian Captioning Description](Image_Captioning/description/)
- [Research Papers](CNN_VIT_Adversarial_Attack/paper/) | [Research Papers](Image_Captioning/paper/)
- [Implementation Reports](CNN_VIT_Adversarial_Attack/report/) | [Implementation Reports](Image_Captioning/report/)

---

**Course**: Neural Networks and Deep Learning (CA7)
**Institution**: University of Tehran
**Date**: September 2025
