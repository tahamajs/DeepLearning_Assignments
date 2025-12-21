"""
Collate function for Image Captioning dataset: pads token sequences and returns lengths.
"""
import torch

PAD_TOKEN = '<pad>'


def pad_sequences(seqs, pad_value=0):
    lengths = [len(s) for s in seqs]
    max_len = max(lengths)
    batch_size = len(seqs)
    out = torch.full((batch_size, max_len), pad_value, dtype=torch.long)
    for i, s in enumerate(seqs):
        out[i, :len(s)] = torch.tensor(s, dtype=torch.long)
    return out, torch.tensor(lengths, dtype=torch.long)


def collate_fn(batch):
    # batch: list of (image, seq, length)
    images, seqs, lengths = zip(*batch)
    images = torch.stack(images, dim=0)
    padded, lengths = pad_sequences(seqs, pad_value=0)
    return images, padded, lengths
