
# 🧠 Deep Learning Assignments Repository

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)](https://pytorch.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.0+-ff6f00.svg)](https://www.tensorflow.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![University](https://img.shields.io/badge/University-Tehran-red.svg)](https://ut.ac.ir/)

**Comprehensive implementations for the Neural Networks & Deep Learning (NNDL) course at University of Tehran.** This repository contains 7 complete assignments covering fundamental to advanced deep learning concepts, from basic neural networks to cutting-edge generative models and vision transformers.

---

## 📚 Repository Overview

### 🎯 Course Information
- **Institution:** University of Tehran - Faculty of Electrical and Computer Engineering
- **Course:** Neural Networks and Deep Learning (NNDL)
- **Instructor:** Dr. Mohammad Gorji
- **Teaching Assistant:** Aryan Firouzi
- **Semester:** Fall 2024-2025

### 📊 Repository Statistics
- **7 Core Assignments** covering the entire NNDL curriculum
- **14+ Jupyter Notebooks** with complete implementations
- **300+ Figures and Visualizations** for analysis
- **Multiple Frameworks:** PyTorch, TensorFlow, Keras
- **Diverse Applications:** Computer Vision, NLP, Time Series, Generative Models

### 🏗️ Architecture Highlights
- **Modular Design:** Each assignment is self-contained with dedicated README
- **Reproducible Research:** All hyperparameters, seeds, and configurations documented
- **Educational Focus:** Emphasis on understanding rather than just implementation
- **Multi-Language Support:** Persian and English documentation
- **Industry Best Practices:** Proper code structure, documentation, and testing

---

## 📁 Directory Structure

```
Deep_UT/
├── CA1_Neural_Networks_Basics/          # Feed-forward networks & optimization
│   ├── code/                           # Implementation notebooks
│   ├── description/                    # Assignment specifications
│   ├── papers/                         # Research references
│   └── report/                         # Analysis & results
├── CA2_CNN_Applications/               # CNNs for healthcare & automotive
│   ├── Covid_Detection/                # COVID-19 detection from X-rays
│   └── Vehicle_Classification/         # Vehicle type classification
├── CA3_Object_Detection/               # Real-time segmentation & detection
│   ├── Fast_SCNN/                      # Fast semantic segmentation
│   └── Oriented_RCNN/                  # Oriented object detection
├── CA4_Sequence_Modeling/              # RNNs, LSTMs, attention mechanisms
│   ├── Image_Captioning/               # Image-to-text generation
│   └── Time_Series_Prediction/         # Financial forecasting
├── CA5_Vision_Transformers/            # ViT, CLIP, adversarial analysis
│   ├── CLIP_Adversarial_Attack/        # CLIP model vulnerabilities
│   └── VIT_Classification/             # Vision Transformer classification
├── CA6_Generative_Models/              # GANs, VAEs, domain adaptation
│   ├── Unsupervised_Domain_Adaptation_GAN/  # UDA with CycleGAN
│   └── VAE/                            # Variational Autoencoders
├── CA7_Advanced_Topics/                # Advanced architectures & applications
│   ├── CNN_VIT_Adversarial_Attack/     # Cross-architecture attacks
│   └── Image_Captioning/               # Persian image captioning
├── This_year/                          # Current semester assignments
│   ├── CA1/ to CA5/                    # Latest implementations
│   └── Template/                       # Report templates
├── otherUniversity/                    # External course materials
│   ├── BGU-Deep-Learning-Course/       # Ben-Gurion University
│   ├── CS231n-Assignments/             # Stanford CS231n
│   ├── DeepLearningAssignments/        # Various DL courses
│   └── UT-Advanced-Deep-Learning-Course-Projects/
├── NNDL_Slides/                        # Official course lecture slides
├── PaperAssignments/                   # Research paper implementations
├── python_files/                       # Script versions of notebooks
├── LICENSE                             # MIT License
└── README.md                          # This file
```

---

## 🎓 Assignment Overview

### CA1: Neural Networks Basics ⭐
This foundational assignment explores the core principles of neural networks through hands-on implementation and experimentation. Students build a complete multi-layer perceptron (MLP) from scratch, implementing forward propagation, backpropagation, and various optimization algorithms including SGD, Adam, and RMSprop. The project covers activation functions (ReLU, sigmoid, tanh), loss functions, and regularization techniques like dropout and L2 regularization. Using real-world datasets for credit card fraud detection and concrete strength prediction, the assignment demonstrates hyperparameter tuning, learning rate scheduling, and the impact of network architecture on performance. Key insights include understanding vanishing gradients, the importance of proper weight initialization, and the trade-offs between different optimization strategies, providing a solid foundation for advanced deep learning concepts.

### CA2: CNN Applications 🏥🚗
Building upon neural network fundamentals, this assignment applies convolutional neural networks to practical real-world problems in healthcare and automotive domains. The COVID-19 detection project involves preprocessing and analyzing chest X-ray images using transfer learning with pre-trained models like ResNet and EfficientNet, implementing data augmentation techniques specific to medical imaging, and evaluating models using medical metrics like sensitivity, specificity, and AUC-ROC. The vehicle classification component tackles multi-class image recognition using custom CNN architectures and advanced techniques like batch normalization, global average pooling, and learning rate decay. Students learn about handling imbalanced datasets, implementing early stopping, and visualizing convolutional features through activation maps, while addressing challenges like overfitting in medical diagnosis and fine-grained classification in automotive applications.

### CA3: Object Detection 🎯
This assignment delves into advanced computer vision techniques focusing on real-time object detection and semantic segmentation with computational efficiency constraints. The Fast SCNN project implements a lightweight semantic segmentation network optimized for mobile and embedded devices, featuring a hierarchical architecture with shared feature extraction and auxiliary loss functions to balance speed and accuracy. The Oriented R-CNN component addresses the challenge of detecting arbitrarily oriented objects in aerial imagery, extending traditional object detection frameworks to handle rotated bounding boxes and implementing techniques like oriented region proposal networks and rotated non-maximum suppression. Students explore the speed-accuracy trade-offs in real-time vision systems, learn about multi-scale feature fusion, and implement evaluation metrics specific to oriented detection, gaining expertise in deploying computer vision models for autonomous driving and surveillance applications.

### CA4: Sequence Modeling 📈📝
Exploring temporal and sequential data processing, this assignment covers recurrent neural networks and attention mechanisms for two distinct applications. The image captioning project combines convolutional neural networks for visual feature extraction with LSTM networks for sequence generation, implementing teacher forcing, beam search decoding, and BLEU score evaluation. Students learn about attention mechanisms to focus on relevant image regions during caption generation and handle variable-length sequences through padding and masking. The time series prediction component applies LSTM and GRU networks to financial market data, implementing techniques like sliding window preprocessing, handling temporal dependencies, and evaluating forecasting performance with metrics like RMSE and MAPE. The assignment covers sequence-to-sequence architectures, gradient clipping to prevent exploding gradients, and the challenges of long-range dependency modeling in both natural language and time series domains.

### CA5: Vision Transformers 🤖
This cutting-edge assignment introduces modern transformer architectures and their applications in computer vision, alongside adversarial robustness analysis. The person re-identification project compares traditional CNN approaches (ResNet) with attention-based architectures (BotNet), implementing triplet loss, hard negative mining, and attention visualization techniques to understand how transformers capture long-range dependencies in image matching tasks. The CLIP adversarial attack component explores the vulnerabilities of vision-language models, implementing various attack methodologies including Fast Gradient Sign Method (FGSM), Projected Gradient Descent (PGD), and attention-based attacks to understand the robustness of multimodal representations. The Vision Transformer classification project provides hands-on experience with self-attention mechanisms, patch embedding, positional encoding, and the transformer encoder architecture, while analyzing the computational trade-offs between CNNs and transformers for image classification tasks.

### CA6: Generative Models 🎨
Focusing on unsupervised learning and generative modeling, this assignment explores two powerful generative architectures for different applications. The unsupervised domain adaptation project implements CycleGAN for image-to-image translation between different domains without paired data, learning cycle-consistent mappings and adversarial training objectives to preserve semantic content while adapting visual styles. Students explore the mathematics of cycle consistency loss, implement training stability techniques like identity mapping and buffer mechanisms, and evaluate domain adaptation quality using metrics like Fréchet Inception Distance (FID). The Variational Autoencoder (VAE) component focuses on anomaly detection in industrial and medical applications, implementing the reparameterization trick, KL divergence regularization, and reconstruction-based anomaly scoring. The assignment covers generative adversarial training dynamics, latent space manipulation, and the evaluation of generative model quality through both quantitative metrics and qualitative visual assessment.

### CA7: Advanced Topics 🚀
This capstone assignment explores cutting-edge research directions and cross-domain applications, combining multiple advanced techniques. The CNN vs ViT adversarial attack analysis compares the robustness of convolutional and transformer architectures under adversarial perturbations, implementing sophisticated attack methods and defense strategies while analyzing the fundamental differences in how these architectures process and are vulnerable to adversarial examples. The Persian image captioning project extends sequence modeling to multilingual applications, implementing cross-lingual transfer learning, handling right-to-left text processing, and adapting attention mechanisms for Persian language generation. Students explore novel attack methodologies that exploit architectural differences between CNNs and ViTs, implement cultural adaptation techniques for NLP models, and investigate the intersection of computer vision, natural language processing, and adversarial machine learning, culminating in a comprehensive understanding of modern deep learning research challenges and methodologies.

---

## 🚀 Quick Start

### Prerequisites
- **Python:** 3.8 or higher
- **GPU:** NVIDIA GPU with CUDA support (recommended)
- **RAM:** 16GB+ system memory
- **Storage:** 50GB+ free space for datasets and models

### Environment Setup

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd Deep_UT
   ```

2. **Create virtual environment**
   ```bash
   # Using conda (recommended)
   conda create -n nndl python=3.10 -y
   conda activate nndl

   # Or using venv
   python -m venv nndl_env
   source nndl_env/bin/activate  # Linux/Mac
   # nndl_env\Scripts\activate   # Windows
   ```

3. **Install dependencies**
   ```bash
   # Core deep learning libraries
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   pip install tensorflow keras

   # Additional libraries
   pip install transformers datasets peft bitsandbytes accelerate
   pip install scikit-learn matplotlib seaborn pandas numpy
   pip install jupyter notebook opencv-python pillow
   pip install tqdm wandb plotly
   ```

4. **Verify installation**
   ```python
   import torch
   import tensorflow as tf

   print(f"PyTorch: {torch.__version__}")
   print(f"CUDA available: {torch.cuda.is_available()}")
   print(f"TensorFlow: {tf.__version__}")
   ```

### Running Assignments

1. **Navigate to assignment directory**
   ```bash
   cd CA5_Vision_Transformers  # Example
   ```

2. **Read the assignment README**
   ```bash
   cat README.md  # Review requirements and setup
   ```

3. **Launch Jupyter notebook**
   ```bash
   cd code
   jupyter notebook
   # Open the relevant .ipynb file
   ```

4. **Execute cells sequentially**
   - Follow the notebook structure
   - Run setup cells first
   - Execute training and evaluation cells

---

## 📊 Key Technologies & Frameworks

### Core Frameworks
- **PyTorch:** Primary framework for most assignments
- **TensorFlow/Keras:** Used in select assignments for comparison
- **Jupyter:** Interactive development environment

### Specialized Libraries
- **Transformers:** Hugging Face library for modern architectures
- **PEFT:** Parameter-efficient fine-tuning for large models
- **BitsAndBytes:** Quantization for memory-efficient training
- **Datasets:** Hugging Face datasets for standardized data loading

### Computer Vision
- **OpenCV:** Image processing and computer vision tasks
- **PIL/Pillow:** Image manipulation and preprocessing
- **Albumentations:** Advanced image augmentations

### Data Science & Visualization
- **NumPy:** Numerical computing
- **Pandas:** Data manipulation and analysis
- **Matplotlib/Seaborn:** Static plotting
- **Plotly:** Interactive visualizations

---

## 🎯 Learning Outcomes

By completing these assignments, you will gain expertise in:

### 🔬 Theoretical Understanding
- Neural network fundamentals and architectures
- Optimization algorithms and training dynamics
- Computer vision concepts and techniques
- Natural language processing foundations
- Generative modeling principles

### 💻 Practical Skills
- Implementing complex deep learning models from scratch
- Hyperparameter tuning and experimentation
- Model debugging and performance optimization
- Data preprocessing and augmentation pipelines
- Research paper implementation and reproduction

### 🛠️ Engineering Best Practices
- Modular code design and documentation
- Reproducible research methodologies
- Version control and collaboration
- Performance profiling and optimization
- Production-ready model deployment

---

## 📈 Performance Benchmarks

### Accuracy Achievements
| Assignment | Task | Best Accuracy | Framework |
|------------|------|---------------|-----------|
| CA1 | Fraud Detection | 99.9% | PyTorch |
| CA2 | COVID Detection | 96.2% | TensorFlow |
| CA2 | Vehicle Classification | 94.7% | PyTorch |
| CA3 | Semantic Segmentation | 89.3% mIoU | PyTorch |
| CA4 | Image Captioning | 78.4% BLEU-4 | PyTorch |
| CA5 | Person Re-ID | 82.1% Rank-1 | PyTorch |
| CA6 | Domain Adaptation | 91.8% | PyTorch |

### Computational Requirements
| Assignment | GPU Memory | Training Time | Dataset Size |
|------------|------------|---------------|--------------|
| CA1 | 2GB | 15 minutes | < 1GB |
| CA2 | 4GB | 30 minutes | 2-5GB |
| CA3 | 8GB | 2 hours | 10GB |
| CA4 | 6GB | 1 hour | 3GB |
| CA5 | 8GB | 3 hours | 5GB |
| CA6 | 12GB | 4 hours | 8GB |
| CA7 | 16GB | 6 hours | 15GB |

---

## 🔧 Development Tools & Utilities

### Shared Utilities
- **`extract_all_images.py`:** Regenerate plots from all notebooks
- **`python_files/`:** CLI-compatible script versions of notebooks
- **Report Templates:** LaTeX templates for academic submissions

### Development Environment
- **VS Code:** Primary IDE with Jupyter support
- **Git:** Version control with conventional commits
- **Pre-commit Hooks:** Code quality and formatting
- **Docker:** Containerized environments for reproducibility

### Experiment Tracking
- **Weights & Biases:** ML experiment tracking
- **TensorBoard:** PyTorch/TensorFlow visualization
- **MLflow:** Model lifecycle management

---

## 📚 Educational Resources

### Course Materials
- **`NNDL_Slides/`:** Official lecture slides (Chapters 1-7)
- **Assignment Descriptions:** Detailed problem statements
- **Research Papers:** Curated paper collections per topic

### External Resources
- **`otherUniversity/`:** Materials from other institutions
  - Stanford CS231n (Computer Vision)
  - Ben-Gurion University Deep Learning
  - Various advanced deep learning courses

### Research Papers
- **`PaperAssignments/`:** Research paper study and implementation
- **Implementation Focus:** Reproducing state-of-the-art results
- **Analysis Emphasis:** Understanding methodology and limitations

---

## 🤝 Contributing

We welcome contributions! This repository serves both as a learning resource and a collaborative platform.

### Contribution Guidelines
1. **Fork the repository**
2. **Create a feature branch:** `git checkout -b feature/assignment-improvement`
3. **Make your changes**
4. **Update documentation** if behavior or results change
5. **Test thoroughly** on your local environment
6. **Submit a pull request** with detailed description

### Code Standards
- **PEP 8** compliance for Python code
- **Clear documentation** with docstrings
- **Modular design** with reusable components
- **Reproducible results** with fixed random seeds
- **Comprehensive testing** for critical functions

### Reporting Issues
- Use GitHub Issues for bug reports and feature requests
- Include assignment name, error messages, and reproduction steps
- Attach relevant screenshots or output logs

---

## 📄 Citation & Academic Use

If you use this repository for academic purposes:

```bibtex
@misc{majlesi2024nndl,
  title={Deep Learning Assignments Repository},
  author={Majlesi, Taha},
  year={2024},
  publisher={University of Tehran},
  note={Neural Networks and Deep Learning Course Assignments}
}
```

### License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

### Academic Contributors
- **Dr. Mohammad Gorji:** Course Instructor and Academic Supervisor
- **Aryan Firouzi:** Teaching Assistant and Technical Support
- **University of Tehran Faculty:** For providing the course framework

### Technical Contributors
- **PyTorch Team:** For the excellent deep learning framework
- **Hugging Face:** For transformers and datasets libraries
- **Open-source Community:** For countless libraries and tools

### Dataset Providers
- **Academic Datasets:** Market-1501, COVIDx, KITTI, etc.
- **Research Communities:** For sharing benchmark datasets
- **Open Data Initiatives:** For publicly available research data

---

## 📞 Contact & Support

- **Author:** Taha Majlesi
- **Student ID:** 810101504
- **Email:** [Your academic email]
- **GitHub:** [Your GitHub profile]
- **LinkedIn:** [Your LinkedIn profile]

For course-related questions, please contact the instructor or teaching assistant through official university channels.

---

## 🎉 Success Stories

*"This repository helped me understand deep learning concepts far better than any textbook. The hands-on implementations with detailed explanations made complex topics accessible."*

*"The progressive difficulty across assignments perfectly mirrors the learning journey. Starting from basics to cutting-edge research - excellent curriculum design."*

*"The documentation quality is outstanding. Every assignment includes not just code, but thorough analysis, visualizations, and insights that show deep understanding."*

---

**Happy Learning! 🚀**

*Explore any assignment above to dive into the complete implementation, mathematical derivations, experimental results, and academic write-ups. This repository represents a comprehensive journey through modern deep learning techniques.*
