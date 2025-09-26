# NNDL_CA4_LSTM-GRU_image_captioning

This folder contains the implementation of image captioning using LSTM and GRU networks. Part of Neural Networks and Deep Learning course assignment 4.

## Concepts Covered

### Image Captioning

Generating natural language descriptions for images, combining computer vision and natural language processing.

### Encoder-Decoder Architecture

- **Encoder**: CNN (e.g., ResNet, VGG) extracts image features
- **Decoder**: RNN (LSTM/GRU) generates captions autoregressively

### Recurrent Neural Networks

- **LSTM**: Long Short-Term Memory, handles long sequences and vanishing gradients
- **GRU**: Gated Recurrent Unit, simpler alternative to LSTM with similar performance

### Training

- Teacher forcing: Uses ground truth tokens as input during training
- Attention mechanisms: Focus on relevant image regions
- Loss: Cross-entropy for token prediction

### Evaluation

- BLEU score: Measures n-gram overlap with reference captions
- METEOR, ROUGE: Alternative language generation metrics
- Qualitative: Fluency and relevance of generated captions

### Challenges

- Sequence generation variability
- Handling novel objects/concepts
- Computational complexity

## Implementation Details

### Dataset

- **Flickr8k**: 8,000 images with 5 captions each
- **Preprocessing**:
  - Captions cleaned (lowercase, punctuation removal)
  - Vocabulary built with frequency thresholding
  - Special tokens: <SOS>, <EOS>, <PAD>, <UNK>

### Model Architecture

- **Encoder**: Pre-trained ResNet-50 (features from avg pooling layer)
- **Decoder**: LSTM/GRU with embedding layer
- **Hybrid Model**: Encoder + Decoder with attention

### Training Parameters

- **Embedding Dimensions**: 50, 150, 300
- **Hidden Size**: 2048
- **Batch Size**: 32
- **Learning Rate**: 0.001 (Adam optimizer)
- **Epochs**: 10-40
- **Beam Width**: 3, 5, 10 for inference

### Loss Function

- Cross-Entropy Loss with ignore_index for <PAD>
- Teacher forcing during training

### Evaluation Metrics

- **BLEU Scores**: BLEU-1, BLEU-2, BLEU-3, BLEU-4
- **Perplexity**: Measure of language model quality
- **Qualitative**: Generated captions vs ground truth

## Results

### Model Comparisons

#### Embedding Size 50 (Baseline)

- **BLEU-1**: ~0.65
- **BLEU-4**: ~0.25
- Fast training, lower quality captions

#### Embedding Size 150

- **BLEU-1**: ~0.70
- **BLEU-4**: ~0.30
- Better semantic understanding

#### Embedding Size 300

- **BLEU-1**: ~0.72
- **BLEU-4**: ~0.32
- Highest quality, slower training

#### With Dropout

- Reduces overfitting
- Slight BLEU improvement on validation

#### Without Teacher Forcing

- More stable generation
- Lower BLEU scores initially

#### Smaller Dictionary

- Faster training
- Lower BLEU due to limited vocabulary

#### 40 Epoch Training

- Overfitting after ~20 epochs
- Best BLEU around epoch 15-20

### Final Model (Trained on Train+Val)

- **Test BLEU-1**: ~0.68
- **Test BLEU-4**: ~0.28
- Generates coherent, descriptive captions

### Qualitative Examples

- Input: Image of dog playing in park
- Generated: "a brown dog is running through the grass"
- Ground truth: "a dog runs through a field of grass"

### Training Dynamics

- Loss decreases steadily for first 10-15 epochs
- Validation loss plateaus, indicating convergence
- Beam search improves caption quality over greedy decoding

## Challenges Addressed

- **Sequence Length Mismatch**: Variable caption lengths handled with padding
- **Vocabulary Sparsity**: Frequent word filtering
- **Training Stability**: Gradient clipping, teacher forcing
- **Evaluation**: BLEU score smoothing for short sentences
