# NNDL_CAe_image_captioning

This folder contains the implementation of image captioning in Persian (Farsi) using attention-based encoder-decoder models. Part of Neural Networks and Deep Learning course assignment e.

## Concepts Covered

### Image Captioning in Persian

Generating natural language descriptions in Persian for images, handling right-to-left text and Arabic script.

### Attention Mechanisms

- **Scaled Dot-Product Attention**: Standard transformer attention
- **Bahdanau Attention**: Content-based attention for sequence generation
- **Visual Attention**: Focus on relevant image regions during captioning

### Encoder-Decoder with Attention

- **Encoder**: CNN extracts visual features
- **Decoder**: RNN with attention generates Persian captions
- **Attention Layer**: Computes relevance between decoder state and image features

### Persian Text Processing

- **Normalization**: Handle different Arabic forms
- **Tokenization**: Word-level tokenization with Hazm library
- **Bidi Algorithm**: Proper display of right-to-left text

## Implementation Details

### Dataset

- **COCO-Flickr-FA-40k**: Persian captions for COCO images
- **Images**: 40,000 images from COCO dataset
- **Captions**: Multiple Persian descriptions per image
- **Preprocessing**:
  - Persian text normalization and cleaning
  - Remove emojis and special characters
  - Tokenization with Hazm
  - Vocabulary building with frequency filtering

### Model Architecture

#### Encoder

- **CNN Backbone**: ResNet-50 pretrained on ImageNet
- **Feature Extraction**: Final conv layer features (2048x7x7)
- **Adaptive Pooling**: Reduce to fixed dimension

#### Decoder

- **Embedding Layer**: Persian word embeddings (300-dim)
- **LSTM/GRU**: Sequence generation with attention
- **Attention Mechanism**: Scaled dot-product or Bahdanau
- **Output Projection**: Vocabulary-sized softmax

#### Attention Variants

- **Scaled Dot-Product**: Q·K^T / sqrt(d_k) with softmax
- **Bahdanau**: MLP-based attention scoring
- **Multi-Head**: Multiple attention heads for richer context

### Training Parameters

- **Embedding Dimension**: 300
- **Decoder Hidden Size**: 512
- **Attention Dimension**: 512
- **Batch Size**: 32
- **Learning Rate**: 0.01 with exponential decay
- **Epochs**: 50
- **Beam Width**: 3 for inference

### Loss Function

- Cross-Entropy Loss with ignore_index for padding
- Scheduled sampling for better convergence

### Evaluation Metrics

- **BLEU Scores**: BLEU-1, BLEU-2, BLEU-3, BLEU-4
- **Corpus BLEU**: Sentence-level vs. corpus-level
- **Attention Maps**: Qualitative evaluation of focus regions

## Results

### Model Performance

- **BLEU-1**: ~0.65
- **BLEU-4**: ~0.25
- **Corpus BLEU**: ~0.35
- Better performance with attention vs. without

### Attention Analysis

- **Scaled Dot-Product**: More stable training, better BLEU scores
- **Bahdanau Attention**: Slower convergence but potentially better alignment
- **Visual Grounding**: Attention maps highlight relevant objects

### Qualitative Results

- **Generated Captions**: Fluent Persian descriptions
- **Attention Visualization**: Heatmaps showing focus on subjects
- **Diverse Generations**: Beam search produces varied captions

### Training Dynamics

- Loss decreases steadily over 50 epochs
- Scheduled sampling improves stability
- Exponential LR decay prevents overfitting

### Challenges Addressed

- **Persian Script**: Right-to-left rendering and normalization
- **Limited Dataset**: 40k images with Persian captions
- **Attention Stability**: Proper scaling and masking
- **Evaluation**: BLEU smoothing for short Persian sentences

## Applications

- **Multilingual Captioning**: Persian image descriptions
- **Accessibility**: Persian audio descriptions for visually impaired
- **Content Moderation**: Persian text analysis
- **Cultural Adaptation**: Localized captioning systems

## Files

- `code/`: PyTorch implementation with Persian text processing
- `report/`: Analysis with BLEU scores and attention visualizations
- `paper/`: Attention and multilingual captioning papers
- `description/`: Assignment description

## Conclusion

The implementation successfully generates Persian captions with attention mechanisms, achieving competitive BLEU scores and demonstrating effective visual-linguistic alignment for Persian language processing.
