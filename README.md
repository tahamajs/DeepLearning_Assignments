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
| Folder | Description | Typical Contents |
|--------|-------------|------------------|
| `CA1_Neural_Networks_Basics/` | Foundational feed-forward networks and optimization studies. | `code/`, `description/`, `paper/`, `report/` |
| `CA2_CNN_Applications/` | CNN-based classification projects (medical & automotive). | `Covid_Detection/`, `Vehicle_Classification/` |
| `CA3_Object_Detection/` | Detection and segmentation pipelines with orientation handling. | `Fast_SCNN/`, `Oriented_RCNN/` |
| `CA4_Sequence_Modeling/` | Captioning and time-series forecasting assignments. | `Image_Captioning/`, `Time_Series_Prediction/` |
| `CA5_Vision_Transformers/` | Transformer-based vision experiments and robustness. | `VIT_Classification/`, `CLIP_Adversarial_Attack/` |
| `CA6_Generative_Models/` | GANs and VAEs for domain adaptation and anomaly detection. | `Unsupervised_Domain_Adaptation_GAN/`, `VAE/` |
| `CA7_Advanced_Topics/` | Capstone projects on adversarial analysis and multilingual captioning. | `CNN_VIT_Adversarial_Attack/`, `Image_Captioning/` |
| `python_files/` | Script equivalents of notebooks for CLI execution. | Python modules grouped by assignment |
| `visualization/` | Shared plotting utilities and exported figures. | Scripts, static assets |
| `NNDL_Slides/` | Course lecture slides. | PDF slide decks |
| `LICENSE`, `README.md` | Repository metadata. | License text, this guide |

### Additional Resources

- `NNDL_Slides/` - Course slides covering theoretical foundations and advanced topics
- `python_files/` - Standalone Python implementations of key assignments for easy execution
- `LICENSE` - MIT License file
- `extract_all_images.py` - Script to extract images from all notebooks

---

## 📚 Assignments Overview
Below, each assignment block provides: (1) conceptual focus, (2) implementation highlights, (3) quantitative results, and (4) representative visualizations. Treat these summaries as an index; open the linked notebook for executable code and deeper commentary.

### CA1: Neural Networks Basics
**Overview:** Introduces feed-forward neural networks from first principles, covering architecture design, forward/backward propagation math, activation selection, and optimization heuristics.

**Highlights**
- Custom network built from scratch with explicit backpropagation routines for educational transparency.
- Comparative experiments against PyTorch implementations to validate gradients and speed.
- Hyperparameter sweeps (learning rate, activation choice) visualized to explain convergence behavior.

**Key Results**
- Confirmed gradient descent correctness through loss curves on benchmark datasets.
- Demonstrated activation function impact on training stability and accuracy.
- Documented optimization trade-offs (SGD vs. Adam) for shallow vs. deeper stacks.

**Representative Visuals**
- ![Neural Network Training](CA1_Neural_Networks_Basics/code/notebook_images/image_cell027_output001.png)
	_Training dynamics validating manual backprop implementation._
- ![Activation Functions](CA1_Neural_Networks_Basics/code/notebook_images/image_cell027_output003.png)
	_Activation comparison (ReLU vs. Sigmoid vs. Tanh) and their learning curves._

**Primary Notebook:** `CA1_Neural_Networks_Basics/code/NNDL_CA1_Q1.ipynb`

---

### CA2: CNN Applications
**Overview:** Two real-world classification pipelines show how architecture choice (bespoke CNN vs. transfer learning) and feature engineering affect performance and deployment trade-offs.

# Deep Learning Assignments Repository

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-orange.svg)](https://pytorch.org/)
[![License](https://img.shields.io-badge/License-MIT-green.svg)](LICENSE)

Hands-on implementations for the Neural Networks & Deep Learning (NNDL) course. Each assignment lives in its own folder with a dedicated README that explains datasets, hyperparameters, metrics, and visualizations. This root file stays lightweight and simply points you to the right place.

**📊 Snapshot:** 7 assignments · 14 primary notebooks · 338 extracted figures

---

## Directory Map & READMEs

| Folder | Focus | Documentation |
|--------|-------|---------------|
| `CA1_Neural_Networks_Basics/` | Feed-forward networks, backprop, optimization studies | [Assignment README](CA1_Neural_Networks_Basics/README.md) |
| `CA2_CNN_Applications/` | Applied CNNs for healthcare & automotive datasets | [Overview](CA2_CNN_Applications/README.md) · [Covid Detection](CA2_CNN_Applications/Covid_Detection/README.md) · [Vehicle Classification](CA2_CNN_Applications/Vehicle_Classification/README.md) |
| `CA3_Object_Detection/` | Real-time segmentation & oriented detection | [Overview](CA3_Object_Detection/README.md) · [Fast-SCNN](CA3_Object_Detection/Fast_SCNN/README.md) · [Oriented R-CNN](CA3_Object_Detection/Oriented_RCNN/README.md) |
| `CA4_Sequence_Modeling/` | Image captioning + time-series forecasting | [Overview](CA4_Sequence_Modeling/README.md) · [Image Captioning](CA4_Sequence_Modeling/Image_Captioning/README.md) · [Time Series](CA4_Sequence_Modeling/Time_Series_Prediction/README.md) |
| `CA5_Vision_Transformers/` | ViT classifiers and CLIP adversarial analysis | [Overview](CA5_Vision_Transformers/README.md) · [ViT Classification](CA5_Vision_Transformers/VIT_Classification/README.md) · [CLIP Attack](CA5_Vision_Transformers/CLIP_Adversarial_Attack/README.md) |
| `CA6_Generative_Models/` | CycleGAN domain adaptation + VAE anomaly detection | [Overview](CA6_Generative_Models/README.md) · [UDA GAN](CA6_Generative_Models/Unsupervised_Domain_Adaptation_GAN/README.md) · [VAE](CA6_Generative_Models/VAE/README.md) |
| `CA7_Advanced_Topics/` | CNN vs. ViT adversarial study & Persian captioning | [Overview](CA7_Advanced_Topics/README.md) · [CNN↔ViT Attack](CA7_Advanced_Topics/CNN_VIT_Adversarial_Attack/README.md) · [Persian Captioning](CA7_Advanced_Topics/Image_Captioning/README.md) |
| `python_files/` | Script equivalents of notebooks (CLI-friendly) | Use alongside assignment READMEs |
| `NNDL_Slides/` | Lecture notes (PDF) | Reference while studying theory |
| `visualization/` | Shared plotting utilities | Scripts + exported assets |

Each linked README details prerequisites, dataset download steps, training commands, and report highlights for that specific module.

---

## Getting Started (Global)

1. **Clone & enter repo**  
	 ```bash
	 git clone <repository-url>
	 cd Deep_UT
	 ```
2. **Create an environment** (pick one)  
	 ```bash
	 python -m venv .venv && source .venv/bin/activate
	 # or
	 conda create -n nndl python=3.10 -y && conda activate nndl
	 ```
3. **Install dependencies**  
	 ```bash
	 pip install -r requirements.txt  # when available
	 ```
4. **Open the target assignment README** for dataset/location specifics, then run the noted notebook(s) inside the corresponding `code/` directory.

> 🔁 Need plots without re-running notebooks? Execute `python extract_all_images.py` (root-level) to refresh every `code/notebook_images/` folder.

---

## Workflow Cheatsheet

- Read the assignment README first (datasets, configs, metrics are recorded there).
- Launch notebooks from the relevant `code/` folder (`jupyter notebook NNDL_*.ipynb`).
- Store any new checkpoints/figures inside that assignment’s directory to keep the repo tidy.
- For scripted runs, use the mirrored modules inside `python_files/` (paths parallel the notebook layout).

---

## Shared Utilities & Resources

- `extract_all_images.py` – regenerates PNGs from executed notebooks for reports.
- `python_files/` – ready-to-run Python scripts mirroring each notebook.
- `NNDL_Slides/` – official course slides to complement the assignments.

---

## Contribution & Support

Contributions are welcome. When opening a PR:
1. Base your changes on the relevant assignment folder.
2. Update that folder’s README if behavior or metrics change.
3. Include before/after metrics or screenshots where possible.

Questions or tweaks? Open an issue and reference the specific assignment README so discussions stay scoped.

---

## License & Credits

- **License:** [MIT](LICENSE)
- **Course:** Neural Networks and Deep Learning (University of Tehran)
- **Author:** Taha Majlesi (810101504)
- **Acknowledgments:** University faculty, referenced researchers, open-source maintainers, and dataset providers who made these experiments possible.

---

Happy experimenting! Dive into any assignment above to explore the full write-up, equations, and results without overloading this main landing page.
**Overview:** Covers translating visual features into language and modeling temporal signals with quantified uncertainty—contrasting encoder-decoder generation with probabilistic forecasting.
