"""Model implementations: BiRNN (for slot filling) and BiLSTMJoint (intent + slot heads)."""
from typing import Optional

import torch
import torch.nn as nn


class BiRNNSlotFiller(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int = 128, hidden_dim: int = 128, num_labels: int = 50, bidirectional: bool = True, dropout: float = 0.5):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.rnn = nn.RNN(embed_dim, hidden_dim, batch_first=True, bidirectional=bidirectional)
        self.dropout = nn.Dropout(dropout)
        self.hidden_dim = hidden_dim
        self.bidirectional = bidirectional
        rnn_out_dim = hidden_dim * (2 if bidirectional else 1)
        self.classifier = nn.Linear(rnn_out_dim, num_labels)

    def forward(self, input_ids: torch.Tensor, lengths: Optional[torch.Tensor] = None):
        emb = self.embedding(input_ids)
        packed = emb
        out, _ = self.rnn(packed)
        out = self.dropout(out)
        logits = self.classifier(out)
        return logits


class BiLSTMJoint(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int = 128, hidden_dim: int = 128, num_slot_labels: int = 50, num_intents: int = 20, dropout: float = 0.5):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.slot_head = nn.Linear(hidden_dim * 2, num_slot_labels)
        # For intent we pool (mean over non-pad positions) or use final hidden state
        self.intent_head = nn.Linear(hidden_dim * 2, num_intents)

    def forward(self, input_ids: torch.Tensor, lengths: Optional[torch.Tensor] = None):
        emb = self.embedding(input_ids)
        out, (hn, cn) = self.lstm(emb)
        out = self.dropout(out)
        slot_logits = self.slot_head(out)
        # intent: mean pooling over sequence length (masking recommended outside)
        pooled = out.mean(dim=1)
        intent_logits = self.intent_head(pooled)
        return slot_logits, intent_logits
