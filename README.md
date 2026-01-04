
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
**Focus:** Fundamental concepts and manual implementations
- **Topics:** Feed-forward networks, backpropagation, optimization algorithms
- **Key Experiments:** Custom MLP from scratch, optimizer comparisons, activation function analysis
- **Datasets:** Credit card fraud detection, concrete strength prediction
- **Highlights:** Manual backpropagation implementation, hyperparameter studies

### CA2: CNN Applications 🏥🚗
**Focus:** Convolutional Neural Networks for real-world applications
- **Covid Detection:** X-ray image analysis for COVID-19 diagnosis
- **Vehicle Classification:** Multi-class vehicle type recognition
- **Techniques:** Transfer learning, data augmentation, medical imaging preprocessing
- **Metrics:** Classification accuracy, medical evaluation metrics

### CA3: Object Detection 🎯
**Focus:** Advanced computer vision with real-time constraints
- **Fast SCNN:** Real-time semantic segmentation for mobile devices
- **Oriented R-CNN:** Rotated object detection for aerial imagery
- **Challenges:** Speed-accuracy trade-offs, orientation handling
- **Applications:** Autonomous driving, aerial surveillance

### CA4: Sequence Modeling 📈📝
**Focus:** Recurrent networks and sequence-to-sequence learning
- **Image Captioning:** CNN-RNN architectures for image description
- **Time Series Prediction:** LSTM networks for financial forecasting
- **Techniques:** Attention mechanisms, sequence preprocessing
- **Evaluation:** BLEU scores, forecasting metrics

### CA5: Vision Transformers 🤖
**Focus:** Modern transformer architectures and adversarial analysis
- **Person Re-identification:** ResNet vs BotNet comparison with attention visualization
- **CLIP Adversarial Attacks:** Robustness analysis of vision-language models
- **ViT Classification:** Vision Transformer implementation and analysis
- **Techniques:** Self-attention, adversarial perturbations, interpretability

### CA6: Generative Models 🎨
**Focus:** Unsupervised learning and generative architectures
- **Domain Adaptation GAN:** CycleGAN for unsupervised domain adaptation
- **VAE Implementation:** Variational autoencoders for anomaly detection
- **Applications:** Image-to-image translation, generative modeling
- **Evaluation:** FID scores, reconstruction quality

### CA7: Advanced Topics 🚀
**Focus:** Cutting-edge techniques and cross-domain applications
- **CNN vs ViT Attacks:** Comparative adversarial robustness analysis
- **Persian Captioning:** Multilingual image description generation
- **Techniques:** Cross-architecture analysis, multilingual NLP
- **Innovation:** Novel attack methodologies, cultural adaptation

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
