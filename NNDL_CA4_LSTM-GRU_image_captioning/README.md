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

### Applications

- Accessibility (describing images for visually impaired)
- Content indexing
- Human-computer interaction

## Files

- `code/`: LSTM/GRU captioning models
- Dataset: MS COCO or Flickr

## Results

The models generate coherent captions, with GRU often providing similar performance to LSTM with fewer parameters.
