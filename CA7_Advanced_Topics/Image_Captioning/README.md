# Advanced Image Captioning with Attention Mechanisms

**Course**: Neural Networks and Deep Learning  
**Assignment**: CAe - Question 2  
**Student**: Taha Majlesi (810101504)  
**Institution**: University of Tehran, Faculty of Electrical and Computer Engineering

---

## 📋 Table of Contents

1. [Abstract](#abstract)
2. [Introduction](#introduction)
3. [Theoretical Framework](#theoretical-framework)
4. [Dataset Preparation](#dataset-preparation)
5. [Model Architecture](#model-architecture)
6. [Training Process](#training-process)
7. [Advanced Techniques](#advanced-techniques)
8. [Results and Evaluation](#results-and-evaluation)
9. [Conclusion](#conclusion)
10. [Files and Resources](#files-and-resources)

---

## Abstract

This project presents a comprehensive implementation of an advanced image captioning system using deep learning techniques. The system employs an encoder-decoder architecture with attention mechanisms to generate natural language descriptions for images in Persian (Farsi). We explore and compare three key attention mechanisms:

1. **Traditional Attention** (Additive Attention)
2. **Scheduled Sampling** with Teacher Forcing decay
3. **Scaled Dot-Product Attention**

Each method's performance is evaluated and analyzed to understand their strengths and limitations. The model successfully generates fluent Persian descriptions with proper attention to relevant image regions.

---

## Introduction

Image captioning is a fundamental task in computer vision and natural language processing that aims to automatically generate natural language descriptions of images. This task bridges the gap between visual understanding and linguistic expression, requiring models to not only recognize objects and scenes in images but also understand their relationships and express them in coherent natural language.

### Problem Definition

Given an input image **I**, the goal is to generate a sequence of words **S = (s₁, s₂, ..., sₙ)** that accurately describes the visual content of the image:

$$\hat{S} = \arg\max_S P(S|I)$$

### Key Challenges

- **Semantic Gap**: Bridging visual features to linguistic concepts
- **Compositionality**: Understanding spatial and temporal relationships
- **Diversity**: Generating varied and creative descriptions
- **Accuracy**: Ensuring factual correctness of generated captions
- **Multilingual Support**: Handling different languages and cultural contexts

---

## Theoretical Framework

### Encoder-Decoder Architecture

The encoder-decoder framework is the foundation of modern image captioning systems:

**Encoder**: Converts the input image into a rich visual representation
$$V = \text{CNN}(I)$$

**Decoder**: Generates captions conditioned on the visual features
$$P(S|I) = \prod_{t=1}^{T} P(s_t | s_{<t}, V)$$

### Attention Mechanism

Attention mechanisms address the limitation of fixed-length representations by allowing the decoder to dynamically focus on different parts of the image:

$$\alpha_t = \text{softmax}(e_t)$$
$$c_t = \sum_{i=1}^{L} \alpha_{t,i} v_i$$

Where:

- $e_t$ are attention energies
- $\alpha_t$ are attention weights
- $c_t$ is the context vector
- $v_i$ are visual features

### Mathematical Formulation

1. **Visual Encoding**: $V = \{v_1, v_2, ..., v_L\}$ where $L$ is the number of spatial locations
2. **Attention Computation**: $e_{t,i} = f_{att}(h_{t-1}, v_i)$
3. **Context Vector**: $c_t = \sum_{i=1}^{L} \alpha_{t,i} v_i$
4. **Hidden State Update**: $h_t = f_{lstm}(h_{t-1}, [s_{t-1}; c_t])$
5. **Word Prediction**: $P(s_t|s_{<t}, I) = \text{softmax}(W_o h_t + b_o)$

---

## Dataset Preparation

### Dataset: COCO-Flickr-FA-40k

- **Total Images**: 40,000 images from COCO dataset
- **Captions**: One Persian caption per image
- **Image Dimensions**: Range from 72×51 to 673×664 pixels
- **Train/Val/Test Split**: 10,000 / 500 / 500 samples

### Data Preprocessing

#### Sample Images from Dataset

![Sample Dataset Images](images/image_cell19_output0.png)

_Example images from the COCO-Flickr Persian dataset_

#### Caption Length Distribution

![Caption Length Histogram](images/image_cell20_output0.png)

_Distribution of caption lengths in the dataset_

### Preprocessing Pipeline

1. **Text Normalization**: Using Hazm library for Persian text normalization
2. **Emoji Removal**: Cleaning text from emoji characters
3. **Tokenization**: Word-level tokenization with Hazm
4. **Vocabulary Building**: Building vocabulary with frequency filtering

### Tokenizer Statistics

- **Context Length**: 40 tokens
- **Total Unique Tokens**: 5,166 (including unknown words)
- **Dictionary Size**: 2,971 tokens (excluding unknown)
- **Unknown Words**: 2,195 tokens (42.5% of unique tokens)
- **Total Tokens in Training Set**: 110,881

### Dataset Structure

The `COCODataset` class:

- Loads images from specified path
- Applies necessary transforms (EfficientNet-B4 preprocessing)
- Converts captions to token IDs
- Allows switching transforms for visualization

---

## Model Architecture

### Architecture Overview

- **Encoder**: EfficientNet-B4 (pre-trained, frozen)
- **Decoder**: LSTM with attention mechanism (trainable)
- **Embedding Dimension**: 300
- **Decoder Hidden Dimension**: 512
- **Attention Dimension**: 512
- **Total Parameters**: 28.4 million
  - Encoder (non-trainable): 17.5 million
  - Decoder (trainable): 10.8 million

### Encoder Design

**Input**: RGB image with dimensions 380×380  
**Output**: Visual features with dimensions `[batch_size, 144, 1792]`

- 144: Number of spatial locations
- 1792: Feature dimension for each location

All EfficientNet parameters are frozen to leverage pre-trained features.

### Decoder Design

The Decoder consists of three main components:

#### 1. Classic Attention Layer

The classic attention mechanism (Additive Attention):

- **Inputs**: Encoder features (a) and decoder hidden state (h)
- **Attention Energy**: Linear combination of features and hidden state
- **Attention Weights**: Softmax over energies
- **Context Vector**: Weighted sum of visual features
- **Beta Coefficient**: A gate to control the importance of the context vector

#### 2. Custom LSTM Cell

A manual implementation of LSTM:

- Four gates: Input (i), Forget (f), Output (o), Gate (g)
- Uses previous word embedding, hidden state, and attention context vector
- Incorporates visual information at each step

#### 3. Output Layers

- **Embedding**: Converts word IDs to dense vectors (300-dim)
- **LSTM**: Processes sequences while considering attention
- **Output**: Combines embedding, hidden state, and context vector

#### Generation Methods

1. **Greedy Decoding**: Selects the word with highest probability at each step
2. **Beam Search**: Maintains k best candidates for better search (beam_width=3)

---

## Training Process

### Loss Function

The loss function combines two components:

1. **Cross-Entropy Loss**: For correct word prediction (main loss)
2. **Regularization Loss**: To ensure attention weights are properly normalized
   $$\mathcal{L}_{reg} = \lambda \cdot \frac{1}{B} \sum_{b=1}^{B} \left(1 - \sum_{i=1}^{L} \alpha_{b,t,i}\right)^2$$

Final loss:
$$\mathcal{L} = \mathcal{L}_{CE} + \lambda \cdot \mathcal{L}_{reg}$$

where $\lambda = 1$ controls regularization importance.

### Training Configuration

- **Optimizer**: Adam with initial learning rate 0.01
- **Learning Rate Scheduler**: ExponentialLR with decay factor 0.98
- **Batch Size**: 64
- **Max Epochs**: 50
- **Early Stopping**: Patience = 10, warmup = 3 epochs
- **Weight Decay**: 1e-5 for regularization

### Training Progress - Base Model

![Training Loss and BLEU](images/image_cell47_output0.png)

_Loss and BLEU score progression during training_

![Validation BLEU](images/image_cell47_output1.png)

_Validation BLEU score improvement over epochs_

### Evaluation Metrics

- **BLEU-1 to BLEU-4**: Measures n-gram overlap between generated and reference captions
- **BLEU-4**: Most accurate metric as it considers sentence structure

---

## Advanced Techniques

### 1. Scheduled Sampling

#### Problem with Teacher Forcing

In standard training, the model uses ground truth words at each step, causing:

1. **Exposure Bias**: During inference, model must use its own predictions
2. **Divergence**: Errors propagate to subsequent steps

#### Solution: Scheduled Sampling

Gradually decreases probability of using ground truth words:

- **Epoch 1**: $p_{teacher} = 1.0$ (always use ground truth)
- **Each Epoch**: $p_{teacher} = p_{teacher} - 0.02$ (gradual decrease)
- **Final Epochs**: $p_{teacher} \approx 0.0$ (only use predictions)

#### Training with Scheduled Sampling

![Scheduled Sampling Training](images/image_cell69_output0.png)

_Loss progression with scheduled sampling_

![Scheduled Sampling BLEU](images/image_cell69_output1.png)

_BLEU score improvement with scheduled sampling_

#### Benefits

- Better adaptation to inference conditions
- More robust to errors
- Improved generation performance

### 2. Scaled Dot-Product Attention

#### Advantages over Classic Attention

1. **Computational Efficiency**: Uses matrix multiplication instead of multiple linear layers
2. **Simplicity**: Fewer parameters
3. **Scaling**: Division by $\sqrt{d_k}$ prevents gradient explosion
4. **Better Performance**: Often outperforms classic attention

#### Implementation

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

Where:

- **Q (Query)**: Extracted from decoder hidden state
- **K (Key)**: Extracted from encoder features
- **V (Value)**: The encoder features themselves

#### Training with Scaled Dot-Product Attention

![Scaled Attention Training](images/image_cell70_output0.png)

_Loss convergence with scaled dot-product attention_

#### Improvements

- Faster convergence
- Higher BLEU scores
- More stable training

---

## Results and Evaluation

### Sample Caption Generation

#### Base Model Results

![Sample Captions - Base Model](images/image_cell49_output0.png)

_Sample generated captions using Greedy decoding_

![Sample Captions with Beam Search](images/image_cell53_output0.png)

_Comparison of Greedy vs Beam Search decoding_

#### Scheduled Sampling Results

![Scheduled Sampling Captions](images/image_cell87_output0.png)

_Generated captions with scheduled sampling_

![Scheduled Sampling Comparison](images/image_cell87_output1.png)

_Improved caption quality with scheduled sampling_

#### Scaled Dot-Product Attention Results

![Scaled Attention Captions](images/image_cell88_output0.png)

_Generated captions with scaled dot-product attention_

### Quantitative Evaluation

#### Base Model Performance

The base model with traditional attention achieves:

- Steady loss decrease during training
- Gradual BLEU score improvement
- Good generalization (small train-val gap)

#### Scheduled Sampling Performance

- Gradual improvement in BLEU scores
- Reduced exposure bias
- More diverse and natural captions
- Longer and more complete descriptions

#### Scaled Dot-Product Attention Performance

- Faster convergence
- Higher BLEU scores
- Better training stability
- Improved caption accuracy

### Attention Map Visualization

Attention maps show which parts of the image the model focuses on when generating each word. Red/orange regions indicate highest attention, while blue/green regions have less attention.

#### Base Model Attention Maps

![Attention Map 1](images/image_cell54_output0.png)
_Attention visualization for sample 1_

![Attention Map 2](images/image_cell55_output0.png)
_Attention visualization for sample 2_

![Attention Map 3](images/image_cell56_output0.png)
_Attention visualization for sample 3_

![Attention Map 4](images/image_cell57_output0.png)
_Attention visualization for sample 4_

#### Scheduled Sampling Attention Maps

![SS Attention Map 1](images/image_cell77_output0.png)
_Attention with scheduled sampling - sample 1_

![SS Attention Map 2](images/image_cell78_output0.png)
_Attention with scheduled sampling - sample 2_

![SS Attention Map 3](images/image_cell79_output0.png)
_Attention with scheduled sampling - sample 3_

#### Scaled Dot-Product Attention Maps

![Scaled Attention Map 1](images/image_cell90_output0.png)
_Scaled dot-product attention - sample 1_

![Scaled Attention Map 2](images/image_cell91_output0.png)
_Scaled dot-product attention - sample 2_

![Scaled Attention Map 3](images/image_cell92_output0.png)
_Scaled dot-product attention - sample 3_

![Scaled Attention Map 4](images/image_cell93_output0.png)
_Scaled dot-product attention - sample 4_

![Scaled Attention Map 5](images/image_cell94_output0.png)
_Scaled dot-product attention - sample 5_

### Detailed Caption Comparison

The following images show detailed comparisons of generated captions:

![Detailed Captions - Base Model](images/image_cell24_output1.png)

_Detailed caption generation analysis_

![Caption Comparison during Training](images/image_cell40_output1.png)
![Caption Comparison during Training 2](images/image_cell40_output3.png)
![Caption Comparison during Training 3](images/image_cell40_output5.png)
![Caption Comparison during Training 4](images/image_cell40_output7.png)
![Caption Comparison during Training 5](images/image_cell40_output9.png)
![Caption Comparison during Training 6](images/image_cell40_output11.png)
![Caption Comparison during Training 7](images/image_cell40_output13.png)
![Caption Comparison during Training 8](images/image_cell40_output15.png)
![Caption Comparison during Training 9](images/image_cell40_output17.png)
![Caption Comparison during Training 10](images/image_cell40_output19.png)
![Caption Comparison during Training 11](images/image_cell40_output21.png)
![Caption Comparison during Training 12](images/image_cell40_output23.png)
![Caption Comparison during Training 13](images/image_cell40_output25.png)
![Caption Comparison during Training 14](images/image_cell40_output27.png)
![Caption Comparison during Training 15](images/image_cell40_output29.png)
![Caption Comparison during Training 16](images/image_cell40_output31.png)
![Caption Comparison during Training 17](images/image_cell40_output33.png)
![Caption Comparison during Training 18](images/image_cell40_output35.png)
![Caption Comparison during Training 19](images/image_cell40_output37.png)
![Caption Comparison during Training 20](images/image_cell40_output39.png)

_Training progress showing generated captions at different epochs_

### Method Comparison

| Method                    | Advantages                                             | Disadvantages                               |
| ------------------------- | ------------------------------------------------------ | ------------------------------------------- |
| **Traditional Attention** | Simple implementation, easy to understand              | May have lower performance in complex cases |
| **Scheduled Sampling**    | Reduced exposure bias, more diverse captions           | Requires careful tuning of decay schedule   |
| **Scaled Dot-Product**    | Higher efficiency, better performance, faster learning | May require more tuning in some cases       |

### Key Observations

1. **Scaled Dot-Product Attention** usually achieves the best BLEU scores
2. **Scheduled Sampling** produces more diverse and natural captions
3. **Beam Search** performs better than Greedy in all methods
4. All methods successfully generate meaningful Persian captions
5. Attention maps show models learn to attend to correct regions

---

## Conclusion

### Project Summary

This project successfully implements a complete Persian image captioning system using deep learning. The system uses an encoder-decoder architecture with EfficientNet-B4 as the encoder and LSTM with attention mechanism as the decoder.

### Main Achievements

1. **Complete Implementation**: All pipeline components from preprocessing to evaluation
2. **Method Comparison**: Examination of three different attention methods and Scheduled Sampling
3. **Comprehensive Evaluation**: Use of BLEU metrics and attention visualization
4. **Persian Processing**: Appropriate implementation for Persian language

### Key Learnings

- **Attention Mechanisms**: Play a key role in generation quality
- **Scheduled Sampling**: Gradually reducing Teacher Forcing improves performance
- **Scaled Dot-Product**: Simplicity and higher efficiency compared to classic attention
- **Beam Search**: Better method than Greedy for caption generation

### Future Directions

- Using Transformer instead of LSTM
- Implementing Self-Attention in decoder
- Using BERT or similar models for improved language processing
- Increasing dataset size
- Fine-tuning Encoder for the specific task

### Final Notes

This project demonstrates that with appropriate architectures and advanced techniques, efficient image captioning systems for Persian language can be built. The key to success lies in the proper combination of architecture, appropriate loss function, and training techniques.

---

## Files and Resources

### Code Files

- `code/NNDL_CAe_2.ipynb`: Complete Persian image captioning implementation with all three attention mechanisms
- `code/NNDL_CAe_2_Complete_IEEE.ipynb`: IEEE format version
- `extract_images.py`: Script to extract images from notebook

### Documentation

- `README.md`: This comprehensive documentation
- `description/NNDL_HWe.pdf`: Assignment description
- `report/NNDL_UT_CA7_2.pdf`: Project report

### Reference Papers

- `paper/1502.03044v3.pdf`: Show and Tell: A Neural Image Caption Generator
- `paper/1506.03099v3.pdf`: Show, Attend and Tell: Neural Image Caption Generation with Visual Attention

### Images Directory

All extracted images from the notebook are stored in the `images/` directory:

- Dataset visualizations
- Training progress plots
- Generated caption examples
- Attention map visualizations
- Method comparison charts

---

## Technical Details

### Requirements

```python
torch>=1.9.0
torchvision>=0.10.0
matplotlib>=3.3.0
numpy>=1.19.0
PIL>=8.0.0
hazm>=0.7.0
arabic-reshaper>=3.0.0
python-bidi>=0.4.2
nltk>=3.6.0
```

### Model Checkpoints

- `checkpoint_loss.pth`: Base model checkpoint
- `checkpoint_scheduled_sampling.pth`: Scheduled sampling model checkpoint
- `scaled_dot_prod_attn.pth`: Scaled dot-product attention model checkpoint

---

## Acknowledgments

This project is part of the Neural Networks and Deep Learning course at the University of Tehran. Special thanks to the course instructors and the developers of the libraries used in this implementation.

---

**Last Updated**: 2024  
**Version**: 1.0
