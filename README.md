# Deep Learning Assignments Repository

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-orange.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

This repository contains comprehensive implementations of advanced deep learning concepts and models as part of the Neural Networks and Deep Learning course assignments. Each assignment demonstrates practical applications of cutting-edge deep learning techniques with detailed mathematical formulations, architectural designs, and performance evaluations.

**📊 Total Notebooks**: 14 | **🖼️ Total Images Extracted**: 338 | **📁 Assignments**: 7

---
## 🧭 Table of Contents
1. [Repository Structure](#-repository-structure)
2. [Assignments Overview](#-assignments-overview)
3. [Key Technologies](#️-key-technologies-and-frameworks)
4. [Core Concepts](#-core-concepts-demonstrated)
5. [Getting Started](#-getting-started)
6. [Image Collection Summary](#-image-collection-summary)
7. [Datasets](#-datasets)
8. [Reproducibility](#-reproducibility)
9. [Experiment Management](#-experiment-management)
10. [Usage](#-usage)
11. [Troubleshooting](#-troubleshooting)
12. [FAQ](#-faq)
13. [Roadmap](#-roadmap)
14. [Citation](#-citation)
15. [License](#-license)
16. [Acknowledgments](#-acknowledgments)

> All image paths are relative and verified. If browsing on GitHub, images should render automatically. If any image fails to load locally, run `python extract_all_images.py` to regenerate notebook image outputs.

## 📋 Repository Structure
This section helps you quickly orient yourself in the project. Each top-level folder corresponds to a course assignment (CA1–CA7) or shared resources. Inside an assignment folder you will typically find:

- `code/` – Jupyter notebooks and Python modules implementing the models.
- `code/notebook_images/` – Auto-extracted images from executed notebook cells (useful for reports or browsing results without re-running).
- `description/` – The official assignment brief, constraints, and goals.
- `paper/` – Reference publications that informed design choices.
- `report/` – Your analytical write-up, metrics, and discussion.
- `README.md` – A mini guide focused on that assignment only.

Use this structure to reproduce experiments or to dive into a concept area (e.g., generative modeling in CA6) without reading unrelated material.

The repository is organized into 7 main assignment folders with descriptive names, each containing:

- `code/` or subfolders with code implementations and Jupyter notebooks
- `code/notebook_images/` - Extracted visualizations and results from notebooks
- `description/` - Assignment specifications and requirements
- `paper/` - Research papers and references
- `report/` - Detailed analysis and results
- `README.md` - Comprehensive technical documentation

### Additional Resources

- `NNDL_Slides/` - Course slides covering theoretical foundations and advanced topics
- `python_files/` - Standalone Python implementations of key assignments for easy execution
- `LICENSE` - MIT License file
- `extract_all_images.py` - Script to extract images from all notebooks

---

## 📚 Assignments Overview
Below, each assignment block provides: (1) conceptual focus, (2) implementation highlights, (3) quantitative results, and (4) representative visualizations. Treat these summaries as an index; open the linked notebook for executable code and deeper commentary.

### CA1: Neural Networks Basics

**Fundamental Neural Network Concepts**

This assignment covers essential neural network principles including architecture, forward/backward propagation, activation functions, and optimization algorithms.

**Contents:**

- Custom neural network implementation from scratch
- Comparison with deep learning frameworks
- Hyperparameter analysis and convergence studies
- Implementation of backpropagation algorithm

**Key Results:**

- Demonstrated understanding of gradient descent and backpropagation
- Analysis of activation functions and their impact on learning
- Performance evaluation on benchmark datasets

![Neural Network Training](CA1_Neural_Networks_Basics/code/notebook_images/image_cell027_output001.png)

_Training dynamics and loss curves demonstrating backpropagation learning_

![Activation Functions](CA1_Neural_Networks_Basics/code/notebook_images/image_cell027_output003.png)

_Comparison of different activation functions and their impact on learning_

**📊 Notebook Images**: 4 visualizations extracted

- Loss curves and training dynamics
- Activation function comparisons
- Convergence analysis

**📁 Location**: `CA1_Neural_Networks_Basics/code/NNDL_CA1_Q1.ipynb`

---

### CA2: CNN Applications
Two complementary real-world classification problems demonstrate how architectural choices (custom vs. pretrained) and feature pipelines affect performance and robustness.

#### 🦠 Covid_Detection

**Medical Image Classification with Deep CNNs**

This project implements a comprehensive COVID-19 detection system using chest X-ray images. The implementation explores multiple CNN architectures and transfer learning approaches to address the critical challenge of automated COVID-19 diagnosis.

**Key Features:**

- **Custom CNN Architecture**: 6 convolutional blocks with batch normalization and dropout
- **Transfer Learning**: Fine-tuning of VGG16 and MobileNetV2 pretrained on ImageNet
- **Data Augmentation**: Extensive augmentation pipeline (rotation, flipping, scaling, brightness)
- **Medical Imaging Pipeline**: Proper preprocessing for chest X-ray images

**Technical Details:**

- **Architecture**: Conv blocks (64→128→256→512 channels) + Global Average Pooling + Dense layers
- **Loss Function**: Binary cross-entropy with class weights for imbalanced data
- **Optimization**: Adam optimizer with learning rate scheduling
- **Regularization**: Dropout (0.5) and L2 weight decay (1e-4)

**Results & Analysis:**

- **VGG16 Fine-tuned**: 92.1% accuracy, 0.91 AUC-ROC
- **MobileNetV2**: 89.3% accuracy, 0.88 AUC-ROC
- **Custom CNN**: 87.6% accuracy, 0.86 AUC-ROC

![COVID-19 Detection](CA2_CNN_Applications/Covid_Detection/code/notebook_images/image_cell035_output000.png)

_X-ray image classification results showing model predictions_

![Training Progress](CA2_CNN_Applications/Covid_Detection/code/notebook_images/image_cell056_output001.png)

_Training and validation accuracy/loss curves for COVID detection models_

**📊 Notebook Images**: 59 visualizations extracted

- Training/validation curves
- Confusion matrices
- ROC curves
- Sample predictions on X-ray images

**📁 Location**: `CA2_CNN_Applications/Covid_Detection/code/NNDL_CA2_1.ipynb`

#### 🚗 Vehicle_Classification

**Multi-Class Vehicle Classification System**

This assignment implements a robust vehicle classification system exploring both end-to-end CNN training and traditional machine learning approaches on CNN-extracted features.

**Key Features:**

- **Dual Approach**: Pure CNN classification vs. CNN feature extraction + SVM
- **Architecture Comparison**: Custom CNN vs. VGG16 vs. AlexNet backbones
- **Feature Engineering**: Comprehensive feature extraction from multiple CNN layers
- **Ensemble Methods**: Combining multiple classifiers for improved performance

**Technical Details:**

- **CNN Feature Extraction**: Features from conv5 layer (512×7×7 → 25088 features)
- **SVM Classification**: RBF kernel with grid search hyperparameter optimization
- **Data Pipeline**: Vehicle dataset preprocessing with normalization and augmentation
- **Evaluation**: 5-fold cross-validation with detailed per-class metrics

**Results & Analysis:**

- **VGG16 + SVM**: 89.2% accuracy, superior generalization
- **AlexNet + SVM**: 87.1% accuracy, faster inference
- **End-to-end CNN**: 85.4% accuracy, single-model simplicity

![Vehicle Classification](CA2_CNN_Applications/Vehicle_Classification/code/notebook_images/image_cell029_output001.png)

_Vehicle classification results demonstrating multi-class recognition_

![Feature Visualization](CA2_CNN_Applications/Vehicle_Classification/code/notebook_images/image_cell052_output001.png)

_CNN feature maps showing learned representations for vehicle recognition_

**📊 Notebook Images**: 21 visualizations extracted

- Feature visualization
- Classification results
- Confusion matrices
- Sample vehicle images with predictions

**📁 Location**: `CA2_CNN_Applications/Vehicle_Classification/code/NNDL_CA2_2.ipynb`

---

### CA3: Object Detection
Focus shifts from per-image classification to spatial understanding. Fast-SCNN addresses dense pixel labeling (semantic segmentation) efficiency, while Oriented R-CNN tackles rotation-aware detection required in aerial or document imagery.

#### 🏙️ Fast_SCNN

**Real-Time Semantic Segmentation with Efficient CNNs**

This project implements Fast-SCNN, a lightweight CNN architecture designed for real-time semantic segmentation on mobile and embedded devices.

**Key Features:**

- **Efficient Architecture**: Depthwise separable convolutions for computational efficiency
- **Multi-Scale Processing**: Pyramid pooling module for global context
- **Real-Time Performance**: Optimized for mobile deployment
- **Urban Scene Understanding**: Segmentation of roads, buildings, vehicles, pedestrians

**Technical Details:**

- **Learning to Downsample**: Initial downsampling module with skip connections
- **Global Feature Extractor**: Pyramid pooling with multiple kernel sizes (1×1, 2×2, 3×3, 6×6)
- **Feature Fusion**: Concatenation of multi-scale features with channel attention
- **Loss Function**: Cross-entropy with class balancing for imbalanced segmentation

**Results & Analysis:**

- **IoU Score**: 0.62 average across all classes
- **Model Size**: 1.2M parameters (vs. 50M+ for standard segmentation models)
- **Inference Speed**: 30+ FPS on mobile GPUs
- **Memory Efficiency**: 50MB model size suitable for edge deployment

![Semantic Segmentation](CA3_Object_Detection/Fast_SCNN/code/notebook_images/image_cell027_output000.png)

_Real-time semantic segmentation results on urban scenes_

![Segmentation Examples](CA3_Object_Detection/Fast_SCNN/code/notebook_images/image_cell051_output000.png)

_Additional segmentation examples showing multi-class pixel-level classification_

**📊 Notebook Images**: 28 visualizations extracted

- Segmentation masks
- Training curves
- Real-time inference examples
- Multi-scale feature maps

**📁 Location**: `CA3_Object_Detection/Fast_SCNN/code/NNDL_CA3_1.ipynb`

#### 🔄 Oriented_RCNN

**Arbitrary-Oriented Object Detection**

This assignment implements Oriented R-CNN for detecting objects with arbitrary orientations, crucial for applications like aerial imagery analysis and document layout detection.

**Key Features:**

- **Oriented Anchors**: 5-parameter anchor representation (x, y, w, h, θ)
- **Rotated ROI Align**: Rotation-aware feature extraction
- **Geometric Transformations**: Proper handling of oriented bounding boxes
- **IoU Computation**: Specialized intersection-over-union for rotated rectangles

**Technical Details:**

- **Region Proposal Network (RPN)**: Oriented anchor generation and classification
- **Rotated RoI Align**: Bilinear sampling with rotation compensation
- **Bounding Box Regression**: 5-parameter regression (dx, dy, dw, dh, dθ)
- **Orientation Encoding**: Angle representation and normalization

**Results & Analysis:**

- **Detection Accuracy**: Superior performance on oriented objects vs. axis-aligned methods
- **Geometric Precision**: Accurate localization of rotated objects
- **Robustness**: Handles various orientations and aspect ratios

![Oriented Object Detection](CA3_Object_Detection/Oriented_RCNN/code/notebook_images/image_cell021_output000.png)

_Oriented bounding boxes demonstrating rotation-aware object detection_

![Oriented Detection Example](CA3_Object_Detection/Oriented_RCNN/code/notebook_images/image_cell026_output000.png)

_Additional example showing precise localization of rotated objects_

**📊 Notebook Images**: 2 visualizations extracted

- Oriented bounding box visualizations
- Detection examples

**📁 Location**: `CA3_Object_Detection/Oriented_RCNN/code/NNDL_CA3_2.ipynb`

---

### CA4: Sequence Modeling
Sequence-focused tasks: translating visual embeddings to language (captioning) and modeling temporal dependencies & uncertainty (time series). This highlights encoder-decoder mechanics versus recurrent probabilistic forecasting.

#### 📝 Image_Captioning

**Hybrid LSTM-GRU Image Captioning with ResNet50**

This project implements an encoder-decoder architecture using ResNet50 for feature extraction and a hybrid LSTM-GRU decoder for generating natural language descriptions from images.

**Key Features:**

- **Visual Encoder**: ResNet50 pre-trained on ImageNet for robust feature extraction
- **Hybrid Decoder**: Combination of LSTM and GRU for sequential text generation
- **Sequence Generation**: Autoregressive text generation with beam search
- **Training Strategies**: Teacher forcing, dropout, different embedding dimensions

**Technical Details:**

- **Encoder**: ResNet50 → Global Average Pooling → 2048-dim features
- **Decoder**: LSTM → GRU → Linear → Softmax for vocabulary prediction
- **Embedding Dimensions**: Evaluated 50, 150, and 300 dimensions
- **Training**: Teacher forcing with dropout regularization (0.5)

**Results & Analysis:**

- **BLEU-1 Score**: ~0.72 (unigram overlap)
- **BLEU-4 Score**: ~0.18 (4-gram overlap)
- **Optimal Configuration**: Embedding size 150-300 provides best trade-off
- **Beam Search**: Consistently outperforms greedy decoding

![Image Captioning Results](CA4_Sequence_Modeling/Image_Captioning/code/notebook_images/image_cell029_output001.png)

_Example generated captions using encoder-decoder architecture_

![Training Curves](CA4_Sequence_Modeling/Image_Captioning/code/notebook_images/image_cell063_output000.png)

_Training and validation loss curves showing model convergence_

**📊 Notebook Images**: 18 visualizations extracted

- Sample caption generations
- Training/validation loss curves
- Performance comparisons across configurations
- Generated captions with different methods (greedy vs. beam search)

**📁 Location**: `CA4_Sequence_Modeling/Image_Captioning/code/nndl-ca4-1.ipynb`

#### ⏰ Time_Series_Prediction

**Uncertainty-Aware Time Series Forecasting**

This assignment implements RNN-based models for time series prediction with uncertainty quantification using Monte Carlo dropout.

**Key Features:**

- **Bidirectional RNNs**: LSTM and GRU variants for sequence modeling
- **Uncertainty Estimation**: Monte Carlo dropout for prediction confidence
- **Temporal Dependencies**: Capturing long-range patterns in sequential data
- **Robust Forecasting**: Handling noisy and irregular time series

**Technical Details:**

- **Architecture**: Bidirectional LSTM/GRU with multiple layers
- **Uncertainty Quantification**: MC Dropout with 50 forward passes
- **Loss Function**: Maximum likelihood estimation with Gaussian likelihood
- **Regularization**: Dropout, recurrent dropout, and L2 regularization

**Results & Analysis:**

- **R² Score**: 0.85 on test data
- **Uncertainty Calibration**: Well-calibrated prediction intervals
- **Robustness**: Handles missing data and outliers effectively

![Time Series Prediction](CA4_Sequence_Modeling/Time_Series_Prediction/code/notebook_images/image_cell036_output000.png)

_Time series forecasting with uncertainty quantification using Monte Carlo dropout_

![Uncertainty Visualization](CA4_Sequence_Modeling/Time_Series_Prediction/code/notebook_images/image_cell071_output000.png)

_Prediction intervals demonstrating calibrated uncertainty estimates_

**📊 Notebook Images**: 20 visualizations extracted

- Time series predictions with confidence intervals
- Training dynamics
- Uncertainty visualization
- Comparison of different architectures

**📁 Location**: `CA4_Sequence_Modeling/Time_Series_Prediction/code/NNDL_CA4_2_1.ipynb`

---

### CA5: Vision Transformers
Explores attention-centric architectures (ViT, CLIP) and contrasts them with CNN inductive biases. Includes robustness assessment against adversarial perturbations.

#### 🔍 VIT_Classification

**Vision Transformer for Image Classification**

This project implements Vision Transformer (ViT) from scratch and compares its performance with traditional CNNs on image classification tasks.

**Key Features:**

- **Patch Embedding**: Image divided into fixed-size patches (16×16)
- **Self-Attention**: Multi-head attention for global context modeling
- **Position Encoding**: Learnable positional embeddings
- **Class Token**: Special token for classification

**Technical Details:**

- **Patch Size**: 16×16 pixels → 768-dim embeddings
- **Transformer Blocks**: 12 layers, 12 attention heads, 768-dim model
- **Pre-training**: Optional initialization with ImageNet-pretrained weights
- **Fine-tuning**: End-to-end training on target datasets

**Results & Analysis:**

- **Accuracy**: 88.2% on CIFAR-10 (comparable to ResNet-50)
- **Computational Cost**: Higher training cost but better scaling
- **Attention Patterns**: Global receptive field captures long-range dependencies
- **Data Efficiency**: Benefits from larger datasets more than CNNs

![Vision Transformer](CA5_Vision_Transformers/VIT_Classification/code/notebook_images/image_cell016_output000.png)

_Attention heatmaps showing how ViT focuses on different image regions_

![ViT Training](CA5_Vision_Transformers/VIT_Classification/code/notebook_images/image_cell024_output000.png)

_Training curves comparing ViT performance with CNN baselines_

**📊 Notebook Images**: 13 visualizations extracted

- Attention heatmaps
- Training curves
- Classification results
- Patch visualization

**📁 Location**: `CA5_Vision_Transformers/VIT_Classification/code/NNDL_CA5_1.ipynb`

#### 🛡️ CLIP_Adversarial_Attack

**Adversarial Attacks on Multimodal Models**

This assignment explores adversarial vulnerabilities in CLIP (Contrastive Language-Image Pretraining) and implements various defense mechanisms.

**Key Features:**

- **Multimodal Attacks**: Perturbing images while preserving semantic meaning
- **Defense Strategies**: LoRA fine-tuning, TeCoA loss, Visual Prompt Tuning
- **Robust Evaluation**: Comprehensive clean vs. adversarial performance analysis
- **Parameter Efficiency**: Low-rank adaptation for practical deployment

**Technical Details:**

- **CLIP Architecture**: Vision Transformer + Text Transformer
- **Attack Methods**: FGSM, PGD with ε-constraints
- **Defense Techniques**: Test-time classifier alignment, prompt tuning
- **Evaluation**: Robustness metrics across multiple attack strengths

**Results & Analysis:**

- **Clean Accuracy**: 65.2% zero-shot performance
- **Adversarial Drop**: 20.1% accuracy loss under attack
- **Defense Improvement**: TeCoA achieves 62.1% robust accuracy
- **Parameter Efficiency**: LoRA uses only 0.8M trainable parameters

![CLIP Adversarial Attack](CA5_Vision_Transformers/CLIP_Adversarial_Attack/code/notebook_images/image_cell035_output001.png)

_Adversarial examples and defense mechanisms for multimodal CLIP model_

![Attack Robustness](CA5_Vision_Transformers/CLIP_Adversarial_Attack/code/notebook_images/image_cell032_output001.png)

_Attack success rates and robustness metrics across different defense strategies_

**📊 Notebook Images**: 14 visualizations extracted

- Adversarial examples
- Attack success rates
- Defense effectiveness
- Robustness curves

**📁 Location**: `CA5_Vision_Transformers/CLIP_Adversarial_Attack/code/NNDL_CA5_2.ipynb`

---

### CA6: Generative Models
Investigates representation learning via generation: domain translation (CycleGAN) for adaptation and probabilistic latent modeling (VAE) for anomaly detection.

#### 🔄 Unsupervised_Domain_Adaptation_GAN

**GAN-Based Unsupervised Domain Adaptation**

This project implements CycleGAN for domain adaptation, enabling models trained on one domain to perform well on related but different domains.

**Key Features:**

- **Cycle Consistency**: Bidirectional mapping between domains
- **Domain Confusion**: Adversarial alignment of feature distributions
- **Unsupervised Learning**: No target domain labels required
- **Style Transfer**: Realistic transformation of visual appearance

**Technical Details:**

- **Generator Networks**: U-Net style with residual blocks
- **Discriminator Networks**: Patch-based discrimination
- **Loss Components**: Adversarial loss + cycle consistency + identity loss
- **Training Strategy**: Alternating optimization with careful loss balancing

**Results & Analysis:**

- **Target Accuracy**: 87.6% on MNIST-M (vs. 75.6% without adaptation)
- **Domain Gap Reduction**: 58% improvement over source-only performance
- **Generated Quality**: FID score of 38.7 indicates realistic samples
- **Feature Alignment**: t-SNE visualization shows domain-invariant representations

![Domain Adaptation GAN](CA6_Generative_Models/Unsupervised_Domain_Adaptation_GAN/code/notebook_images/image_cell040_output000.png)

_CycleGAN domain transfer results showing style translation between domains_

![Generated Samples](CA6_Generative_Models/Unsupervised_Domain_Adaptation_GAN/code/notebook_images/image_cell057_output000.png)

_Generated samples demonstrating realistic domain transfer quality_

**📊 Notebook Images**: 28 visualizations extracted

- Domain transfer examples
- Generated samples
- Training dynamics
- Feature space visualizations

**📁 Location**: `CA6_Generative_Models/Unsupervised_Domain_Adaptation_GAN/code/NNDL_CA6_1.ipynb`

#### 🎭 VAE

**Variational Autoencoder for Anomaly Detection**

This assignment implements VAE for generative modeling and demonstrates its application in unsupervised anomaly detection for medical imaging.

**Key Features:**

- **Probabilistic Encoding**: Amortized variational inference
- **Reparameterization Trick**: Enables gradient-based optimization
- **Anomaly Scoring**: Reconstruction error as anomaly indicator
- **Medical Application**: Polyp detection in gastrointestinal endoscopy

**Technical Details:**

- **Encoder**: CNN-based recognition network (μ, log σ²)
- **Decoder**: Transpose CNN for image reconstruction
- **ELBO Loss**: Reconstruction + KL divergence regularization
- **β-VAE Variant**: Tunable regularization strength

**Results & Analysis:**

- **Reconstruction Quality**: PSNR 28.5dB, SSIM 0.89 on normal images
- **Anomaly Detection**: AUC 0.90, superior to reconstruction-based methods
- **Latent Space**: Well-structured manifold for interpolation
- **Medical Utility**: Reliable polyp detection with low false positive rate

![VAE Reconstruction](CA6_Generative_Models/VAE/code/notebook_images/image_cell019_output000.png)

_Variational Autoencoder reconstruction and latent space visualization_

![VAE Latent Space](CA6_Generative_Models/VAE/code/notebook_images/image_cell034_output001.png)

_Latent space interpolation showing smooth transitions between generated samples_

**📊 Notebook Images**: 8 visualizations extracted

- Reconstruction examples
- Latent space visualizations
- Anomaly detection results
- Medical image analysis

**📁 Location**: `CA6_Generative_Models/VAE/code/NNDL_CA6_2.ipynb`

---

### CA7: Advanced Topics
Integrates prior themes—adversarial robustness comparison (CNN vs. ViT) and multilingual captioning challenges (Persian RTL text). Emphasis on cross-disciplinary adaptation and security.

#### 🔀 CNN_VIT_Adversarial_Attack

**Comparative Adversarial Analysis: CNNs vs. ViTs**

This extra assignment provides a comprehensive comparison of adversarial vulnerabilities between convolutional and transformer-based vision models.

**Key Features:**

- **Architecture Comparison**: ResNet-50 vs. ViT-Base side-by-side analysis
- **Attack Suite**: FGSM, PGD, CW attacks with multiple strengths
- **Defense Evaluation**: Adversarial training and input preprocessing
- **Robustness Metrics**: Detailed analysis of clean vs. robust performance

**Technical Details:**

- **CNN Model**: ResNet-50 with 25M parameters
- **ViT Model**: 12-layer transformer with 86M parameters
- **Attack Implementation**: Torchattacks library with custom modifications
- **Defense Methods**: Adversarial training with PGD-based augmentation

**Results & Analysis:**

- **Clean Performance**: ViT 84.7% vs. ResNet 76.2% accuracy
- **Adversarial Robustness**: ViT 57.4% vs. ResNet 52.1% under strong attacks
- **Attack Transferability**: High transfer rate between architectures
- **Computational Trade-offs**: ViT requires more compute but offers better robustness

![CNN vs ViT Adversarial](CA7_Advanced_Topics/CNN_VIT_Adversarial_Attack/code/notebook_images/image_cell020_output000.png)

_Comparative analysis of adversarial robustness between CNNs and Vision Transformers_

![Adversarial Examples](CA7_Advanced_Topics/CNN_VIT_Adversarial_Attack/code/notebook_images/image_cell040_output000.png)

_Adversarial examples and attack success rates comparison_

**📊 Notebook Images**: 34 visualizations extracted

- Adversarial examples comparison
- Attack success rates
- Robustness metrics
- Performance comparisons

**📁 Location**: `CA7_Advanced_Topics/CNN_VIT_Adversarial_Attack/code/NNDL_CAe_1.ipynb`

#### 🌐 Image_Captioning (Persian)

**Multilingual Image Captioning in Persian**

This advanced project extends image captioning to Persian language, addressing the challenges of right-to-left script and low-resource language processing.

**Key Features:**

- **Persian NLP Pipeline**: Hazm library for tokenization and normalization
- **Multilingual Attention**: Multi-head attention for cross-modal alignment
- **RTL Text Handling**: Proper bidirectional text processing
- **Cultural Adaptation**: Persian-specific caption generation

**Technical Details:**

- **Text Processing**: Persian normalization, word tokenization, vocabulary building
- **Model Architecture**: Transformer-based encoder-decoder with Persian embeddings
- **Beam Search**: Multilingual beam search with Persian language model
- **Evaluation**: BLEU scores adapted for Persian morphological complexity

**Results & Analysis:**

- **BLEU-4 Score**: 0.195 (competitive for low-resource language)
- **Persian Fluency**: Natural Persian sentence generation
- **Cultural Relevance**: Captions reflect Persian linguistic and cultural context

![Persian Image Captioning](CA7_Advanced_Topics/Image_Captioning/images/image_cell19_output0.png)

_Sample Persian captions generated with attention mechanisms_

![Dataset Analysis](CA7_Advanced_Topics/Image_Captioning/images/image_cell20_output0.png)

_Caption length distribution in the Persian dataset_

![Persian Caption Attention](CA7_Advanced_Topics/Image_Captioning/images/image_cell54_output0.png)

_Attention visualization showing focus regions for Persian caption generation_

![Training Progress](CA7_Advanced_Topics/Image_Captioning/images/image_cell47_output0.png)

_Training and validation BLEU scores showing model improvement over epochs_

**📊 Notebook Images**: 89 visualizations extracted (largest collection!)

- Sample Persian captions
- Training curves
- Attention visualizations
- Comparison with English baseline

**📁 Location**: `CA7_Advanced_Topics/Image_Captioning/code/NNDL_CAe_2.ipynb`

---

## 🛠️ Key Technologies and Frameworks
This catalog lists core libraries. Use it to verify environment completeness or to replace any component (e.g., swap visualization library) without breaking conceptual flows.

- **Deep Learning Frameworks**: PyTorch, TensorFlow/Keras
- **Computer Vision**: OpenCV, PIL, torchvision
- **Natural Language Processing**: Hazm (Persian), NLTK
- **Data Science**: NumPy, Pandas, Scikit-learn
- **Visualization**: Matplotlib, Seaborn
- **Experiment Tracking**: Weights & Biases, TensorBoard

## 🎓 Core Concepts Demonstrated
Each bullet here maps to explicit implementations somewhere in the notebooks. For quick study, choose a concept (e.g., "attention mechanisms") then search across the repo or open the matching assignment.

### Neural Network Architectures

- **Convolutional Networks**: CNNs, ResNets, EfficientNets
- **Recurrent Networks**: LSTMs, GRUs, attention mechanisms
- **Transformers**: Self-attention, multi-head attention, position encoding
- **Generative Models**: GANs, VAEs, flow-based models

### Learning Paradigms

- **Supervised Learning**: Classification, regression, object detection
- **Unsupervised Learning**: Autoencoders, generative modeling
- **Self-Supervised Learning**: Contrastive learning (CLIP)
- **Adversarial Learning**: Attacks, defenses, robust training

### Advanced Techniques

- **Transfer Learning**: Pretrained models, fine-tuning
- **Regularization**: Dropout, batch normalization, weight decay
- **Optimization**: Adam, SGD, learning rate scheduling
- **Data Augmentation**: Geometric transforms, color jittering
- **Ensemble Methods**: Model averaging, bagging

### Evaluation Metrics

- **Classification**: Accuracy, precision, recall, F1-score, AUC-ROC
- **Detection/Segmentation**: IoU, mAP, precision-recall curves
- **Generation**: BLEU, ROUGE, METEOR, FID, IS
- **Time Series**: MAE, RMSE, R², uncertainty metrics

---

## 🚀 Getting Started
Your step-by-step path: clone → create environment → install deps → open a notebook → execute cells → extract images (optional) → consult reports for interpretation.

### Prerequisites

- Python 3.8+
- PyTorch 1.9+
- CUDA-compatible GPU (recommended)
- Jupyter Notebook or JupyterLab

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd Deep_UT

# Install dependencies (if requirements.txt exists)
pip install -r requirements.txt

# Or install core packages
pip install torch torchvision numpy pandas matplotlib jupyter
```

### Recommended Environment Setup

```bash
# Create isolated environment (venv example)
python -m venv .venv
source .venv/bin/activate  # macOS/Linux

# OR using conda
conda create -n nndl python=3.10 -y
conda activate nndl

# Install core dependencies
pip install -r requirements.txt  # if present

# Verify GPU (optional)
python - <<'PY'
import torch; print('CUDA available:', torch.cuda.is_available())
PY
```

### Quick Start (Example: CA5 ViT Classification)
```bash
cd CA5_Vision_Transformers/VIT_Classification/code
jupyter notebook  # open NNDL_CA5_1.ipynb
```

### Seeding for Determinism
Add at top of notebooks/scripts:
```python
import random, numpy as np, torch
def seed_all(seed=42):
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False
seed_all()
```

### Navigation

Each assignment folder is self-contained:

- `CA1_Neural_Networks_Basics/`
- `CA2_CNN_Applications/`
- `CA3_Object_Detection/`
- `CA4_Sequence_Modeling/`
- `CA5_Vision_Transformers/`
- `CA6_Generative_Models/`
- `CA7_Advanced_Topics/`

### Execution

1. **Run Jupyter Notebooks**: Navigate to `code/` directories and open notebooks
2. **Run Python Scripts**: Execute standalone Python files in `python_files/`
3. **Extract Images**: Run `python extract_all_images.py` to extract all notebook images

### Viewing Results

All extracted images from notebooks are available in:

- `[Assignment]/code/notebook_images/` directories
- Organized by cell number and output index

---

## 📊 Image Collection Summary
Use this table to identify which notebooks demonstrate the richest visual diagnostics. High image count generally correlates with exploratory depth (e.g., Persian captioning).

| Assignment | Notebook                 | Images Extracted |
| ---------- | ------------------------ | ---------------- |
| CA1        | Neural Networks Basics   | 4                |
| CA2        | Covid Detection          | 59               |
| CA2        | Vehicle Classification   | 21               |
| CA3        | Fast-SCNN                | 28               |
| CA3        | Oriented R-CNN           | 2                |
| CA4        | Image Captioning         | 18               |
| CA4        | Time Series Prediction   | 20               |
| CA5        | ViT Classification       | 13               |
| CA5        | CLIP Adversarial Attack  | 14               |
| CA6        | Domain Adaptation GAN    | 28               |
| CA6        | VAE                      | 8                |
| CA7        | CNN vs ViT Attack        | 34               |
| CA7        | Persian Image Captioning | 89               |
| **Total**  | **14 Notebooks**         | **338 Images**   |

---

## 📖 Documentation
Reports and referenced papers augment code with rationale. Prefer reading a report after first skim of the notebook outputs; it will connect visuals to theory.

Each assignment contains:

- Detailed README.md files with technical explanations
- PDF reports with comprehensive analysis
- Research papers referenced in each project
- Assignment descriptions and requirements

## 🎓 Educational Value
This section clarifies intended audiences and how each group can leverage the repository (learning curves for students, reproducible baselines for researchers, etc.).

This repository serves as a comprehensive resource for:

- **Students**: Practical implementations of deep learning concepts
- **Researchers**: Benchmarking and extending state-of-the-art methods
- **Practitioners**: Production-ready code for real-world applications
- **Educators**: Teaching materials with detailed explanations

Each implementation includes:

- Mathematical derivations
- Architectural decisions
- Hyperparameter tuning strategies
- Performance analysis
- Visualizations and results

---

## 📝 Usage
Guidelines ensure respectful, traceable reuse—especially in academic contexts. Always keep derivative work transparent by citing and linking back.

All code in this repository is provided for educational purposes. When using any part of this code:

1. **Attribute**: Provide appropriate credit to the authors
2. **Understand**: Read the code comments and documentation
3. **Experiment**: Modify parameters and architectures to learn
4. **Cite**: If used in academic work, cite the relevant assignments

---

## 📄 License
MIT: you can use, modify, distribute—provided you retain the license notice. Ideal for educational and prototype purposes.

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👤 Author Information
Author and course metadata—helpful for provenance and citation entries.

**Course**: Neural Networks and Deep Learning  
**Institution**: University of Tehran, Faculty of Electrical and Computer Engineering  
**Student**: Taha Majlesi (810101504)  
**Date**: 2025

---

## 🙏 Acknowledgments
Credits foundational contributors (data providers, open-source maintainers). Expanding this when integrating new datasets maintains academic integrity.

- University of Tehran for the comprehensive course curriculum
- All researchers whose papers were referenced in these assignments
- Open-source community for providing excellent deep learning frameworks
- Dataset providers for making their data publicly available

---

**Note**: This repository represents a comprehensive collection of deep learning implementations covering fundamental concepts to advanced topics. All notebooks include detailed markdown explanations, mathematical formulations, and visual results extracted from the execution outputs.

---
## 📦 Datasets
| Assignment | Dataset(s) | Source / Link |
|-----------|------------|---------------|
| CA1 Basics | Synthetic & MNIST | http://yann.lecun.com/exdb/mnist/ |
| CA2 Covid Detection | COVID-19 Chest X-Ray, Normal/Pneumonia | Kaggle / Cohen et al. curated sets |
| CA2 Vehicle Classification | Vehicle images (custom curated) | Internal course dataset |
| CA3 Fast-SCNN | Cityscapes subset | https://www.cityscapes-dataset.com/ |
| CA3 Oriented R-CNN | DOTA / aerial samples subset | https://captain-whu.github.io/DOTA/ |
| CA4 Image Captioning | Flickr8k/Flickr30k (English) | https://github.com/jbrownlee/Datasets |
| CA4 Time Series | Synthetic + real sensor logs | Course-provided |
| CA5 ViT/CLIP | CIFAR-10, COCO (subsample), custom text prompts | https://www.cs.toronto.edu/~kriz/cifar.html |
| CA6 Domain Adaptation | MNIST → MNIST-M | https://github.com/facebookresearch/domainbed |
| CA6 VAE | Medical polyp imagery | Kvasir dataset |
| CA7 Persian Captioning | Custom Persian image–caption corpus | Course-provided |

> Some datasets require manual download due to license constraints. See each assignment folder `description/` for precise acquisition steps.

## 🔁 Reproducibility
Follow every listed step for comparable metrics. Deviations (different CUDA versions, mixed precision tweaks) can introduce silent performance shifts.
1. Use the seeding snippet in [Getting Started](#-getting-started).
2. Maintain versions using `pip freeze > requirements.lock.txt`.
3. Run notebooks in order; avoid mixing checkpoint states across runs.
4. For multi-GPU variance, set `CUDA_LAUNCH_BLOCKING=1` when debugging.
5. Log metrics (BLEU, IoU, Accuracy) every epoch; append JSON to `report/metrics.json` (pattern followed in some assignments).

## 📈 Experiment Management
| Aspect | Recommendation |
|--------|---------------|
| Logging | Use Weights & Biases or TensorBoard for consistent charts. |
| Checkpoints | Save per epoch; keep best metric & last epoch. |
| Hyperparameters | Store in `config.yaml` (add if missing) for each experiment. |
| Versioning | Tag Git commits with `exp/<assignment>-<date>` after stable results. |

## 🧪 Benchmark Summary (Selected)
Snapshot of representative metrics—useful to sanity-check reruns. If your numbers diverge widely, inspect seed setting, dataset integrity, and library versions.
| Task | Model | Key Metric | Score |
|------|-------|-----------|-------|
| Covid Detection | VGG16 (fine-tuned) | AUC-ROC | 0.91 |
| Vehicle Classification | VGG16 + SVM | Accuracy | 89.2% |
| Fast-SCNN | Fast-SCNN | Mean IoU | 0.62 |
| Oriented R-CNN | Oriented R-CNN | Detection mAP* | (qualitative) |
| Image Captioning (EN) | ResNet50 + Hybrid Decoder | BLEU-4 | 0.18 |
| Time Series | BiLSTM | R² | 0.85 |
| ViT Classification | ViT-Base | CIFAR-10 Acc | 88.2% |
| CLIP Robustness | CLIP + TeCoA | Robust Acc | 62.1% |
| Domain Adaptation | CycleGAN + Classifier | Target Acc | 87.6% |
| VAE Anomaly | β-VAE | AUC | 0.90 |
| Persian Captioning | Transformer | BLEU-4 | 0.195 |
*mAP value varies by subset; detailed numbers in assignment report.

## 🛠️ Troubleshooting
| Issue | Possible Cause | Fix |
|-------|----------------|-----|
| CUDA unavailable | Driver / toolkit mismatch | Reinstall CUDA or use CPU fallback (`device='cpu'`). |
| Notebook image missing | Not executed / extraction script not run | Execute cell & run `extract_all_images.py`. |
| Divergent training loss | Learning rate too high | Reduce LR (e.g., `1e-3 → 5e-4`). |
| Low BLEU score | Tokenization inconsistency | Rebuild tokenizer & ensure consistent preprocessing. |
| Memory OOM | Batch size too large | Lower batch size or enable gradient accumulation. |

## ❓ FAQ
**Q: Why BLEU-4 is relatively low?**  Image captioning BLEU-4 is sensitive to vocabulary size and dataset scale; small datasets produce modest multi-gram overlap.

**Q: Can I substitute datasets?**  Yes—update paths and ensure identical preprocessing scripts.

**Q: How to run on CPU only?**  Set environment variable `CUDA_VISIBLE_DEVICES=""` or modify device selection logic.

**Q: Where are model checkpoints stored?**  Typically inside each assignment's `code/` or a generated `checkpoints/` folder (create if missing).

## 🗺️ Roadmap
- [ ] Add `config.yaml` templates per assignment.
- [ ] Introduce unit tests for data loaders.
- [ ] Add ONNX export for key models.
- [ ] Provide Dockerfile for containerized reproducibility.
- [ ] Expand Persian dataset & add morphological analysis module.

## 🤝 Contributing
Set expectations for external improvements. Even if private now, a clear process eases future collaboration or public release.
Contributions (issues, PRs) are welcome. Please:
1. Fork and create a feature branch: `git checkout -b feat/<short-name>`
2. Follow existing code style and add docstrings.
3. Include before/after metrics for model changes.
4. Update README sections if user-facing behavior changes.

## 📑 Citation
If this work contributes to academic or professional output, cite it as:
```
@misc{majlesi2025nndl,
	title={Neural Networks and Deep Learning Course Assignments},
	author={Majlesi, Taha},
	institution={University of Tehran},
	year={2025},
	note={GitHub repository}
}
```

## 🧵 Style & Conventions
Consistency improves readability and reduces onboarding friction. Adhering to naming and output placement norms keeps automation scripts (like image extraction) reliable.
- Python: PEP8 naming, snake_case functions, UpperCamelCase classes.
- Reproducibility: Always set seeds and record versions.
- File Outputs: Place generated artifacts under `report/` or `code/notebook_images/`.

---
