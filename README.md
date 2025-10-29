# Deep Learning Assignments Repository

This repository contains comprehensive implementations of advanced deep learning concepts and models as part of the Neural Networks and Deep Learning course assignments. Each assignment demonstrates practical applications of cutting-edge deep learning techniques with detailed mathematical formulations, architectural designs, and performance evaluations.

## Repository Structure

The repository is organized into 7 main assignment folders with descriptive names, each containing:

- `code/` or subfolders with code implementations
- `description/`: Assignment specifications
- `paper/`: Research papers and references
- `report/`: Detailed analysis and results
- `README.md`: Comprehensive technical documentation

### Additional Resources

- `NNDL_Slides/`: Course slides covering theoretical foundations and advanced topics
- `python_files/`: Standalone Python implementations of key assignments for easy execution (mirroring the notebook structure)
- `LICENSE`: MIT License file
- `README.md`: This main documentation file

## Assignments Overview

### CA1: Neural Networks Basics

**Fundamental Neural Network Concepts**

This assignment covers essential neural network principles including architecture, forward/backward propagation, activation functions, and optimization algorithms. The implementation provides hands-on experience with mathematical foundations of neural networks, from basic building blocks to complete training pipelines.

**Key Features:**

- **Neural Network Architecture**: Understanding neurons, layers, and network topology
- **Activation Functions**: Implementation and comparison of Sigmoid, ReLU, Tanh
- **Loss Functions**: Mean Squared Error (MSE) and Cross-Entropy implementations
- **Optimization Algorithms**: Gradient Descent, Stochastic Gradient Descent (SGD)
- **Weight Initialization**: Proper initialization techniques for stable training
- **Learning Rate Scheduling**: Adaptive learning rate methods

**Technical Implementation:**

- **Custom Implementation**: Neural network from scratch using NumPy
- **Forward Propagation**: Computing network outputs through layers
- **Backpropagation**: Computing gradients for parameter updates
- **Framework Comparison**: Validation against PyTorch/TensorFlow implementations

**Results and Analysis:**

- **Convergence Analysis**: Learning curves and optimization behavior visualization
- **Hyperparameter Impact**: Effect of learning rate, batch size, and network architecture
- **Generalization**: Training vs. validation performance analysis
- **Computational Efficiency**: Time and memory complexity evaluation

**Educational Value:**

- Mathematical foundations of neural networks
- Implementation challenges and best practices
- Debugging and troubleshooting techniques
- Performance optimization strategies

See [detailed README](CA1_Neural_Networks_Basics/README.md) for complete documentation.

### CA2: CNN Applications

#### Covid_Detection

**Medical Image Classification with Deep CNNs**

This project implements a comprehensive COVID-19 detection system using chest X-ray images. The implementation explores multiple CNN architectures and transfer learning approaches to address the critical challenge of automated COVID-19 diagnosis. The system classifies chest X-ray images into three categories: Normal, Pneumonia, and COVID-19.

**Key Features:**

- **Custom CNN Architecture**:
  - 6 convolutional blocks with increasing filters (64→128→256→512)
  - Batch normalization for stable training
  - Dropout (0.2-0.5) for regularization
  - Input: 150×150×3 RGB images
  - Architecture: Conv blocks → Global Average Pooling → Dense(512) → Dense(256) → Dense(3) with Softmax
- **Transfer Learning**:
  - **VGG16**: Pre-trained on ImageNet, fine-tuned on medical data
  - **MobileNetV2**: Lightweight architecture optimized for mobile deployment
  - Fine-tuning strategy: Freeze early layers, train later layers
- **Data Augmentation**:
  - Geometric transformations: Rotation, width/height shift, shear, zoom
  - Color adjustments: Brightness and contrast modifications
  - Horizontal flipping for augmentation
- **Medical Imaging Pipeline**:
  - Proper preprocessing for chest X-ray images
  - Intensity normalization (0-1 range)
  - Resize to 150×150 pixels

**Technical Details:**

- **Loss Function**: Categorical cross-entropy with class weights for handling imbalanced data
- **Optimization**:
  - Adam optimizer (β1=0.9, β2=0.999)
  - Learning rate scheduling (constant and exponential decay)
- **Regularization**:
  - Dropout: 0.2-0.5 depending on layer
  - L2 weight decay: 1e-4
- **Training Strategy**:
  - Early stopping to prevent overfitting
  - Model checkpointing for best performance
  - Validation monitoring

**Results & Analysis:**

- **VGG16 Fine-tuned**: 92.1% accuracy, 0.91 AUC-ROC, 0.89 F1-score
- **MobileNetV2**: 89.3% accuracy, 0.88 AUC-ROC, 0.86 F1-score
- **Custom CNN**: 87.6% accuracy, 0.86 AUC-ROC, 0.84 F1-score
- **Clinical Relevance**:
  - Demonstrates practical applicability in medical diagnosis
  - High sensitivity for COVID-19 detection
  - Robust performance on diverse chest X-ray images
- **Key Insights**:
  - Transfer learning significantly improves performance
  - VGG16 provides best accuracy but MobileNetV2 offers better efficiency
  - Custom CNN achieves competitive results with architectural simplicity

See [detailed README](CA2_CNN_Applications/Covid_Detection/README.md) for complete results and visualizations.

#### Vehicle_Classification

**Multi-Class Vehicle Classification System**

This assignment implements a robust vehicle classification system exploring both end-to-end CNN training and traditional machine learning approaches on CNN-extracted features. The project classifies vehicle images into specific categories (e.g., different Toyota models like Corolla, Camry, Rav4, etc.) and compares deep learning approaches with classical machine learning methods.

**Key Features:**

- **Dual Approach**:
  - Pure CNN classification (end-to-end learning)
  - CNN feature extraction + SVM (hybrid approach)
- **Architecture Comparison**:
  - **Custom CNN (ToyotaModelCNN)**: Multiple convolutional blocks with increasing filters (32→64→128→256)
  - **VGG16**: 16-layer network with strong feature extraction
  - **AlexNet**: 8-layer network, winner of ImageNet 2012
- **Feature Engineering**:
  - Feature extraction from conv5 layer (512×7×7 → 25,088 features)
  - Multiple CNN layers analyzed for optimal feature representation
- **Data Augmentation**:
  - Geometric transformations: Rotation, translation, scaling, flipping
  - Color transformations: Brightness, contrast, saturation adjustments
  - Noise injection for robustness

**Technical Details:**

- **CNN Architecture**:
  - Convolutional blocks with batch normalization
  - Max pooling after each block
  - Dropout for regularization
  - Fully connected layers for classification
- **SVM Classification**:
  - RBF kernel with grid search hyperparameter optimization
  - Support for linear, RBF, and polynomial kernels
  - Hyperparameter tuning: C, gamma parameters
- **Fine-tuning Strategy**:
  - Freeze early layers (preserve general features)
  - Fine-tune later layers for vehicle-specific features
  - Replace final classification layer
- **Evaluation**:
  - 5-fold cross-validation
  - Detailed per-class metrics (precision, recall, F1-score)
  - Confusion matrix analysis

**Results & Analysis:**

- **VGG16 + SVM**: 89.2% accuracy, superior generalization, best overall performance
- **AlexNet + SVM**: 87.1% accuracy, faster inference, good balance
- **End-to-end CNN**: 85.4% accuracy, single-model simplicity
- **Custom CNN**: Competitive performance with architectural flexibility
- **Key Insights**:
  - Feature extraction approach provides better generalization than end-to-end training
  - SVM on CNN features often outperforms pure CNN classifiers
  - Transfer learning significantly boosts performance
  - VGG16 features are most discriminative for vehicle classification

See [detailed README](CA2_CNN_Applications/Vehicle_Classification/README.md) for complete results and visualizations.

### CA3: Object Detection

#### Fast_SCNN

**Real-Time Semantic Segmentation with Efficient CNNs**

This project implements Fast-SCNN (Fast Semantic Segmentation Convolutional Neural Network), a lightweight CNN architecture designed for real-time semantic segmentation on mobile and embedded devices. Fast-SCNN balances speed and accuracy for applications requiring pixel-level scene understanding.

**Key Features:**

- **Efficient Architecture**:
  - Depthwise separable convolutions reduce parameters by ~9×
  - Total parameters: ~1.2M (vs. ~50M for DeepLabV3+)
  - Optimized for mobile and embedded devices
- **Multi-Scale Processing**:
  - **Learning to Downsample Module**: Initial downsampling with standard and depthwise separable convolutions
  - **Pyramid Pooling Module (PPM)**: Multi-scale context at 1×1, 2×2, 3×3, 6×6 scales
  - **Global Feature Extractor**: Bottleneck blocks inspired by MobileNetV2
- **Real-Time Performance**:
  - ~120 FPS inference speed on mobile GPUs
  - 50MB model size suitable for edge deployment
- **Urban Scene Understanding**:
  - Segmentation of 11 semantic classes: Sky, Building, Pole, Road, Pavement, Tree, SignSymbol, Fence, Car, Pedestrian, Bicyclist
  - CamVid dataset: 367 training, 101 validation, 233 test images (360×480 resolution)

**Technical Details:**

- **Architecture Components**:
  1. **Learning to Downsample**: Standard conv + DSConv for initial feature extraction
  2. **Global Feature Extractor**:
     - Expansion (1×1 conv): 32→96 channels
     - Depthwise (3×3): Efficient convolution
     - Projection (1×1 conv): 96→16 channels
     - Residual connections when input/output channels match
  3. **Feature Fusion Module**: Combines high-res (1/8) and low-res (1/32) features with dilated depthwise conv
  4. **Classifier**: DSConv layers + final Conv2D(num_classes) with Dropout(0.3) and Softmax
- **Loss Functions**:
  - Cross-entropy loss: `L_CE = -∑_c y_c log(ŷ_c)`
  - IoU loss: `L_IoU = 1 - IoU_score`
  - Dice loss: `L_Dice = 1 - (2∑ y_i ŷ_i + ε) / (∑ y_i + ∑ ŷ_i + ε)`
- **Optimization**:
  - Adam optimizer (β1=0.9, β2=0.999)
  - Polynomial learning rate decay: `lr = initial_lr × (1 - step/max_steps)^power`
  - Batch size: 16
  - Learning rate: 0.045 (decays to 0)

**Results & Analysis:**

- **Mean IoU (mIoU)**: 0.62 average across all classes
- **Pixel Accuracy**: 91.3% overall pixel classification accuracy
- **Model Efficiency**:
  - 1.2M parameters (91% reduction vs. standard models)
  - 50MB model size
  - ~120 FPS on mobile GPUs
- **Per-Class Performance**:
  - High performance on large objects (Sky, Building, Road)
  - Challenges with small objects (Pole, Bicyclist)
- **Key Insights**:
  - Depthwise separable convolutions enable real-time performance
  - Multi-scale features crucial for accurate segmentation
  - Balance between accuracy and efficiency achieved

See [detailed README](CA3_Object_Detection/Fast_SCNN/README.md) for complete results and visualizations.

#### Oriented_RCNN

**Arbitrary-Oriented Object Detection**

This assignment implements Oriented R-CNN for detecting objects with arbitrary orientations, crucial for applications like aerial imagery analysis and document layout detection. The system extends Faster R-CNN to handle rotated bounding boxes throughout the entire detection pipeline.

**Key Features:**

- **Oriented Bounding Box Representation**:
  - **5-parameter format**: (x_c, y_c, w, h, θ) where θ is rotation angle
  - **8-parameter format**: Four corner coordinates (x1,y1, x2,y2, x3,y3, x4,y4)
  - Conversion between representations using rotation matrices
- **Oriented Anchors**:
  - Pre-defined oriented boxes at multiple positions
  - Scales: [32, 64, 128, 256, 512]
  - Aspect ratios: [0.5, 1, 2]
  - Angles: [-90°, -45°, 0°, 45°, 90°] (5 orientations)
- **Rotated ROI Align**:
  - Handles rotated regions using spatial transformer
  - Bilinear sampling with rotation compensation
  - Output: 7×7×256 features for each proposal
- **Geometric Transformations**:
  - Proper handling of oriented bounding boxes
  - Rotation-aware feature extraction
  - Specialized IoU computation for rotated rectangles

**Technical Details:**

- **Backbone Network**:
  - **ResNet-50**: Deep residual network for feature extraction
  - **Feature Pyramid Network (FPN)**: Multi-scale feature maps (P2-P6)
  - Lateral connections and top-down pathway for multi-scale detection
- **Region Proposal Network (RPN)**:
  - **Oriented Anchor Generation**: Places anchors at each spatial position
  - **Classification Head**: Binary classification (object vs. background)
  - **Regression Head**: 5-parameter refinement (Δx, Δy, Δw, Δh, Δθ)
  - Regression targets: Normalized relative to anchor boxes
- **RCNN Head**:
  - Fully connected layers: 1024 → 1024 neurons
  - Classification branch: Object class probabilities
  - Regression branch: Oriented bounding box refinements
- **Loss Functions**:
  - **RPN Classification**: Binary cross-entropy `L_cls = -∑ (y log ŷ + (1-y) log(1-ŷ))`
  - **RPN Regression**: Smooth L1 loss for oriented boxes
  - **RCNN Loss**: Combined classification and regression losses

**Results & Analysis:**

- **Detection Accuracy**: Superior performance on oriented objects vs. axis-aligned methods
- **Geometric Precision**: Accurate localization of rotated objects
- **Robustness**: Handles various orientations and aspect ratios
- **Application Areas**:
  - Ship detection in satellite imagery
  - Text detection in documents
  - Vehicle detection in aerial photography
  - Medical image analysis
- **Key Insights**:
  - 5-parameter representation more efficient than 8-parameter
  - Oriented anchors crucial for good initialization
  - Rotated ROI Align essential for accurate feature extraction
  - Significant improvement over axis-aligned detection for rotated objects

See [detailed README](CA3_Object_Detection/Oriented_RCNN/README.md) for complete results and visualizations.

### CA4: Sequence Modeling

#### Image_Captioning

**Attention-Based Image Captioning with LSTM/GRU**

This project implements an encoder-decoder architecture with attention mechanisms for generating natural language descriptions from images. The system bridges computer vision and natural language processing to create coherent captions that describe image content.

**Key Features:**

- **Visual Encoder**:
  - ResNet-50 or VGG16 pre-trained CNN for image feature extraction
  - Global average pooling of final convolutional layer
  - Output: Feature vector v ∈ ℝ^d (d=2048 for ResNet-50)
  - Captures semantic and spatial information from images
- **Attention Decoder**:
  - **LSTM**: Long Short-Term Memory with gating mechanisms
    - Forget gate: `f_t = σ(W_f · [h_{t-1}, x_t] + b_f)`
    - Input gate: `i_t = σ(W_i · [h_{t-1}, x_t] + b_i)`
    - Cell state: `C_t = f_t * C_{t-1} + i_t * C̃_t`
  - **GRU**: Gated Recurrent Unit with reset and update gates
    - Reset gate: `r_t = σ(W_r · [h_{t-1}, x_t])`
    - Update gate: `z_t = σ(W_z · [h_{t-1}, x_t])`
- **Attention Mechanisms**:
  - **Bahdanau Attention (Additive)**: `e_{t,i} = v_a^T tanh(W_a h_{t-1} + U_a v_i)`
  - **Luong Attention (Multiplicative)**: `e_{t,i} = h_t^T W_a v_i`
  - Context vector: `c_t = ∑_i α_{t,i} v_i` where `α_{t,i} = softmax(e_{t,i})`
- **Sequence Generation**:
  - Autoregressive text generation
  - Beam search for diverse caption generation
  - Vocabulary management and tokenization

**Technical Details:**

- **Encoder**:
  - ResNet-50 → Adaptive pooling → 2048-dim features
  - Features extracted from multiple spatial locations
- **Attention Mechanism**:
  - Bahdanau attention with MLP scoring
  - Attention weights visualize focus regions
- **Decoder**:
  - 512-dim LSTM/GRU with attention context concatenation
  - Word embeddings + context vector as input
  - Output: Probability distribution over vocabulary
- **Training Strategy**:
  - **Teacher Forcing**: Use ground truth tokens during training
  - **Scheduled Sampling**: Gradually decrease teacher forcing probability
  - Exposure bias mitigation
  - Cross-entropy loss on predicted tokens

**Results & Analysis:**

- **BLEU Scores**:
  - **BLEU-1**: 0.72 (unigram overlap)
  - **BLEU-4**: 0.18 (4-gram overlap)
- **Attention Visualization**:
  - Clear focus on relevant image regions
  - Attention maps highlight objects mentioned in captions
- **Semantic Quality**:
  - Generated captions capture main objects and actions
  - Proper object relationships described
  - Coherent sentence structure
- **Model Comparison**:
  - LSTM vs. GRU performance analysis
  - Attention vs. non-attention comparison
  - Impact of different attention mechanisms

See [detailed README](CA4_Sequence_Modeling/Image_Captioning/README.md) for complete results and visualizations.

#### Time_Series_Prediction

**Uncertainty-Aware Time Series Forecasting with RNNs**

This assignment implements RNN-based models for time series prediction with uncertainty quantification using Maximum Likelihood Estimation (MLE). The system predicts future values based on historical observations while providing confidence intervals for forecasts.

**Key Features:**

- **Bidirectional RNNs**:
  - LSTM and GRU variants for sequence modeling
  - Process sequences in both forward and backward directions
  - Richer temporal context for predictions
- **Uncertainty Estimation**:
  - **Maximum Likelihood Estimation (MLE)**: Learn both mean (μ) and variance (σ²)
  - Gaussian assumption: `y ~ N(μ, σ²)`
  - Dual-head architecture: Separate outputs for mean and log-variance
  - Negative log-likelihood loss: `L = ∑ [logσ² + (y-μ)²/σ²]`
- **Temporal Dependencies**:
  - Capturing long-range patterns in sequential data
  - Handling non-stationary time series
  - Multiple forecast horizons (short-term and long-term)
- **Robust Forecasting**:
  - Handling noisy and irregular time series
  - Missing data handling
  - Outlier robustness

**Technical Details:**

- **Architecture**:
  - Bidirectional LSTM/GRU with multiple layers
  - Encoder RNN processes input sequence: `h_T = RNN(x1, ..., xT)`
  - Dual prediction heads:
    - Mean head: `μ = W_μ h_T + b_μ`
    - Variance head: `logσ² = W_σ h_T + b_σ` (log for numerical stability)
- **Uncertainty Quantification**:
  - Probabilistic forecasts with confidence intervals
  - Well-calibrated prediction intervals
  - Uncertainty reflects model confidence
- **Loss Function**:
  - Negative log-likelihood (MLE objective)
  - Balances accuracy and uncertainty calibration
- **Regularization**:
  - Dropout for preventing overfitting
  - Recurrent dropout for RNN layers
  - L2 regularization on weights

**Results & Analysis:**

- **Performance Metrics**:
  - **R² Score**: 0.85 on test data
  - Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE)
  - Prediction interval coverage
- **Uncertainty Calibration**:
  - Well-calibrated prediction intervals
  - Uncertainty increases for extrapolation regions
  - Captures temporal uncertainty patterns
- **Robustness**:
  - Handles missing data effectively
  - Outlier detection and handling
  - Stable predictions on noisy data
- **Model Comparison**:
  - LSTM vs. GRU performance
  - Bidirectional vs. unidirectional
  - Impact of uncertainty modeling

See [detailed README](CA4_Sequence_Modeling/Time_Series_Prediction/README.md) for complete results and visualizations.

### CA5: Vision Transformers

#### VIT_Classification

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

#### CLIP_Adversarial_Attack

**Adversarial Attacks on Multimodal Models**

This comprehensive study investigates adversarial attacks on CLIP (Contrastive Language-Image Pre-training) vision-language models, focusing on transfer attacks from traditional CNN architectures. The project implements and evaluates multiple adversarial training strategies including standard adversarial training, text-guided contrastive adversarial training (TeCoA), and visual prompt tuning (VPT).

**Key Features:**

- **Transfer Attacks**: Generating adversarial examples using ResNet-20 and testing their effectiveness on CLIP
- **Defense Strategies**: Multiple defense mechanisms including:
  - Standard Adversarial Training (Adv.) with Cross-Entropy Loss
  - Text-guided Contrastive Adversarial Training (TeCoA) with temperature parameter analysis
  - Visual Prompt Tuning (VPT) for parameter-efficient adaptation
- **Multimodal Robustness**: Analyzing the inherent robustness properties of CLIP's multimodal architecture
- **Comprehensive Evaluation**: Detailed performance analysis on clean images, adversarial examples, and transfer attacks

**Technical Details:**

- **CLIP Architecture**:
  - Vision Encoder: ViT-Base-Patch32 (12 layers, 768-dim embeddings, 32×32 patches)
  - Text Encoder: Transformer (12 layers, 512-dim embeddings, 77 token context length)
  - Shared Embedding Space: 512-dimensional
- **Attack Methods**:
  - PGD (Projected Gradient Descent) with ε=8/255, α=2/255, 7 iterations
  - Transfer attacks from ResNet-20 source model
- **Defense Techniques**:
  - LoRA (Low-Rank Adaptation): Rank=8, α=32, 0.32% trainable parameters
  - TeCoA: Contrastive loss with temperature parameters (0.01 vs 0.1)
  - VPT: Learnable visual prompts with minimal parameters (~5K)
- **Evaluation Framework**:
  - Classification metrics: Accuracy, Precision, Recall, F1-Score
  - Adversarial robustness: Attack Success Rate (ASR), Robustness Gap
  - Statistical significance testing

**Results & Analysis:**

- **Transfer Attack Vulnerability**: 78.4% attack success rate demonstrates CLIP's vulnerability to transfer attacks
- **Baseline Performance**:
  - Clean Accuracy: 65-75% (zero-shot CLIP)
  - Adversarial Accuracy: 45-55% (20% robustness gap)
- **Defense Effectiveness**:
  - **LoRA + Cross-Entropy**: Clean Acc ~72-82%, Adv Acc ~58-62% (-14% gap)
  - **LoRA + TeCoA (τ=0.01)**: Clean Acc ~75-92%, Adv Acc ~62-66% (-13% gap) ✅ **Best Performance**
  - **LoRA + TeCoA (τ=0.1)**: Clean Acc ~73-91%, Adv Acc ~61-65% (-13% gap)
  - **VPT + TeCoA**: Clean Acc ~74%, Adv Acc ~61% (-13% gap, only ~5K parameters)
- **Key Insights**:
  - Temperature parameter significantly affects robustness (lower τ=0.01 superior)
  - Parameter-efficient methods (LoRA, VPT) achieve competitive performance
  - Multimodal nature provides inherent defense mechanisms
  - Text embeddings show higher resistance to image-based attacks

**Visualizations**: Comprehensive visualizations including confusion matrices, training curves, clean vs. adversarial comparisons, and performance comparisons across all defense strategies. See [detailed README](CA5_Vision_Transformers/CLIP_Adversarial_Attack/README.md) for complete results.

### CA6: Generative Models

#### Unsupervised_Domain_Adaptation_GAN

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

#### VAE

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

### CA7: Advanced Topics

#### CNN_VIT_Adversarial_Attack

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

#### Image_Captioning

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
- **Multilingual Capability**: Framework extensible to other RTL languages

## Key Technologies and Frameworks

- **Deep Learning Frameworks**: PyTorch, TensorFlow/Keras
- **Computer Vision**: OpenCV, PIL, torchvision
- **Natural Language Processing**: Hazm (Persian), NLTK
- **Data Science**: NumPy, Pandas, Scikit-learn
- **Visualization**: Matplotlib, Seaborn
- **Experiment Tracking**: Weights & Biases, TensorBoard

## Core Concepts Demonstrated

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

## Getting Started

1. **Prerequisites**: Python 3.8+, PyTorch 1.9+, CUDA-compatible GPU
2. **Installation**: `pip install -r requirements.txt` (if available)
3. **Navigation**: Each assignment folder is self-contained with descriptive names:
   - `CA1_Neural_Networks_Basics/`
   - `CA2_CNN_Applications/`
   - `CA3_Object_Detection/`
   - `CA4_Sequence_Modeling/`
   - `CA5_Vision_Transformers/`
   - `CA6_Generative_Models/`
   - `CA7_Advanced_Topics/`
4. **Execution**:
   - Run Jupyter notebooks in `code/` directories or subfolders
   - Alternatively, execute Python scripts in `python_files/` for standalone implementations
5. **Documentation**: Refer to individual README.md files for detailed guides

## How to Run

برای اجرای اسکریپت `main.py`، دستور زیر را در ترمینال اجرا کنید:

```bash
python main.py
```

یا اگر از Python 3 استفاده می‌کنید:

```bash
python3 main.py
```

## How to Test

برای اجرای تست‌های این پروژه، دستور زیر را در ترمینال اجرا کنید:

```bash
pytest test_main.py
```

یا اگر `pytest` به صورت پیش‌فرض نصب نیست، می‌توانید از `python -m pytest` استفاده کنید:

```bash
python -m pytest test_main.py
```

## Educational Value

This repository serves as a comprehensive resource for:

- **Students**: Practical implementations of deep learning concepts
- **Researchers**: Benchmarking and extending state-of-the-art methods
- **Practitioners**: Production-ready code for real-world applications
- **Educators**: Teaching materials with detailed explanations

Each implementation includes mathematical derivations, architectural decisions, hyperparameter tuning, and performance analysis, providing a complete learning experience from theory to practice.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use this repository for academic purposes, please cite the relevant assignments and provide appropriate attribution to the original authors and datasets used.

---

**Course**: Neural Networks and Deep Learning
**Institution**: University of Tehran
**Date**: September 2025
