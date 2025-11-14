
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
