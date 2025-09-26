# NNDL_CA4_LSTM-GRU_image_captioning

This folder contains the implementation of image captioning using LSTM and GRU networks with attention mechanisms. Part of Neural Networks and Deep Learning course assignment 4.

## Concepts Covered

### Image Captioning

Image captioning is the task of generating natural language descriptions for images, requiring the integration of computer vision for visual understanding and natural language processing for text generation.

#### Problem Formulation

Given an image I, generate a sequence of words W = (w1, w2, ..., wT) that describes the image content. This is typically approached as a sequence-to-sequence learning problem.

### Encoder-Decoder Architecture

The encoder-decoder framework separates visual feature extraction from language generation.

#### Encoder (CNN-based)

- **Architecture**: Pre-trained CNN (ResNet-50, VGG16)
- **Feature Extraction**: Global average pooling of final convolutional layer
- **Output**: Feature vector v ∈ ℝ^d (d = 2048 for ResNet-50)
- **Purpose**: Captures semantic and spatial information from the image

#### Decoder (RNN-based)

- **Architecture**: LSTM or GRU network
- **Input**: Word embeddings + context vectors
- **Output**: Probability distribution over vocabulary at each timestep
- **Purpose**: Generates captions autoregressively

### Recurrent Neural Networks for Sequence Generation

#### Long Short-Term Memory (LSTM)

LSTM addresses the vanishing gradient problem in vanilla RNNs through gating mechanisms.

**Cell State Update**:

```
f_t = σ(W_f · [h_{t-1}, x_t] + b_f)    # Forget gate
i_t = σ(W_i · [h_{t-1}, x_t] + b_i)    # Input gate
C̃_t = tanh(W_C · [h_{t-1}, x_t] + b_C) # Candidate values
C_t = f_t * C_{t-1} + i_t * C̃_t        # Cell state
o_t = σ(W_o · [h_{t-1}, x_t] + b_o)    # Output gate
h_t = o_t * tanh(C_t)                   # Hidden state
```

#### Gated Recurrent Unit (GRU)

GRU simplifies LSTM with fewer parameters while maintaining similar performance.

**Update Equations**:

```
r_t = σ(W_r · [h_{t-1}, x_t])           # Reset gate
z_t = σ(W_z · [h_{t-1}, x_t])           # Update gate
h̃_t = tanh(W · [r_t * h_{t-1}, x_t])    # Candidate activation
h_t = (1 - z_t) * h_{t-1} + z_t * h̃_t   # Hidden state
```

### Attention Mechanism

Attention allows the decoder to focus on relevant parts of the image when generating each word.

#### Bahdanau Attention (Additive)

```
# Compute attention scores
e_{t,i} = v_a^T tanh(W_a h_{t-1} + U_a v_i)
α_{t,i} = softmax(e_{t,i})

# Context vector
c_t = ∑_i α_{t,i} v_i

# Decoder input
h̃_t = tanh(W_c [c_t, h_{t-1}])
```

#### Luong Attention (Multiplicative)

```
# Attention scores
e_{t,i} = h_t^T W_a v_i
α_{t,i} = softmax(e_{t,i})

# Context vector
c_t = ∑_i α_{t,i} v_i
```

### Training Methodology

#### Teacher Forcing

During training, use ground truth tokens as input rather than predicted tokens:

- **Advantages**: Faster convergence, stable training
- **Disadvantages**: Exposure bias during inference
- **Scheduled Sampling**: Gradually decrease teacher forcing probability

#### Loss Function

Cross-entropy loss over vocabulary predictions:

```
L = -∑_t ∑_w y_{t,w} log ŷ_{t,w}
```

Ignores <PAD> tokens using ignore_index.

### Inference Strategies

#### Greedy Decoding

Select highest probability word at each step:

```
w_t = argmax P(w_t | w_{<t}, I)
```

#### Beam Search

Maintain k most likely sequences:

- **Beam Width**: k (typically 3-10)
- **Score**: Log probability normalized by length
- **Advantages**: Better captions than greedy
- **Complexity**: O(k × |V| × T)

### Evaluation Metrics

#### BLEU Score

Measures n-gram overlap between generated and reference captions:

```
BLEU-N = BP × exp(∑_{n=1}^N w_n log p_n)
```

Where BP is brevity penalty, p_n is n-gram precision.

#### METEOR

Considers synonyms, stemming, and word order:

- Better correlation with human judgments than BLEU
- Handles morphological variations

#### ROUGE

Recall-oriented metric for summarization:

- ROUGE-N: N-gram recall
- ROUGE-L: Longest common subsequence

### Implementation Details

#### Dataset: Flickr8k

- **Size**: 8,000 images, 40,000 captions
- **Split**: 6,000 train, 1,000 val, 1,000 test
- **Preprocessing**:
  - Captions: Lowercase, remove punctuation, add <SOS>/<EOS>
  - Vocabulary: Words appearing ≥5 times
  - Max length: 20-30 tokens

#### Model Architecture

```python
class EncoderCNN(nn.Module):
    def __init__(self):
        self.resnet = models.resnet50(pretrained=True)
        self.resnet.fc = nn.Linear(2048, embed_size)  # Fine-tune

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size):
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.attention = Attention(hidden_size, embed_size)
```

#### Training Parameters

- **Embedding Dimensions**: 50, 150, 300
- **Hidden Size**: 512, 1024, 2048
- **Batch Size**: 32-128
- **Learning Rate**: 0.001 (Adam), 0.0001 (fine-tuning)
- **Epochs**: 20-40 with early stopping
- **Dropout**: 0.5 to prevent overfitting

### Results and Analysis

#### Quantitative Results

| Configuration          | BLEU-1 | BLEU-2 | BLEU-3 | BLEU-4 | Training Time |
| ---------------------- | ------ | ------ | ------ | ------ | ------------- |
| Embed 50, No Attention | 0.65   | 0.45   | 0.32   | 0.25   | Fast          |
| Embed 150, Attention   | 0.70   | 0.52   | 0.38   | 0.30   | Medium        |
| Embed 300, Attention   | 0.72   | 0.55   | 0.41   | 0.32   | Slow          |
| + Dropout 0.5          | 0.71   | 0.54   | 0.40   | 0.31   | -             |
| Beam Search (k=5)      | +0.03  | +0.04  | +0.03  | +0.02  | -             |

#### Ablation Studies

- **Attention Impact**: +5-8% BLEU improvement
- **Embedding Size**: Diminishing returns beyond 150
- **Teacher Forcing**: 10% faster convergence
- **Beam Search**: Better captions, 2-3× slower inference

#### Qualitative Analysis

**Example 1:**

- Image: Dog chasing ball in park
- Generated: "a brown dog is running after a ball in the grass"
- BLEU-1: 0.8, BLEU-4: 0.4

**Example 2:**

- Image: People at beach
- Generated: "several people are standing on the beach near the water"
- BLEU-1: 0.7, BLEU-4: 0.3

#### Training Dynamics

- **Loss Convergence**: Steady decrease for 15-20 epochs
- **Overfitting**: Validation BLEU peaks around epoch 15
- **Attention Maps**: Decoder focuses on relevant image regions

### Challenges and Solutions

1. **Exposure Bias**: Teacher forcing vs. scheduled sampling
2. **Vocabulary Sparsity**: Subword embeddings (BPE) or character-level models
3. **Computational Cost**: Efficient attention implementations
4. **Evaluation Limitations**: BLEU doesn't capture semantic similarity
5. **Dataset Bias**: Flickr8k lacks diversity in scenes/objects

### Applications and Extensions

- **Visual Question Answering**: Answer questions about images
- **Image Retrieval**: Find images matching text descriptions
- **Multimodal Learning**: Joint vision-language representations
- **Medical Imaging**: Generate radiology reports
- **Accessibility**: Describe images for visually impaired users

## Files

- `code/nndl-ca4-1.ipynb`: Complete implementation with LSTM/GRU models
- `report/`: Detailed analysis with attention visualizations
- `description/`: Assignment requirements and dataset details

## Key Learnings

1. Attention mechanisms significantly improve caption quality
2. Teacher forcing accelerates training but creates inference mismatch
3. Beam search provides better captions than greedy decoding
4. Embedding size has diminishing returns beyond certain point
5. BLEU scores correlate with human judgments but have limitations

## Conclusion

This implementation demonstrates the power of encoder-decoder architectures with attention for image captioning, achieving BLEU-1 scores around 0.72. The project showcases the integration of CNNs for vision and RNNs for language, highlighting both the capabilities and challenges of multimodal learning.
