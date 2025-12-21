"""Encoder-Decoder model for SLU (non-aligned slot generation and intent prediction)."""
from typing import Optional

import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int = 128, hidden_dim: int = 128, num_layers: int = 1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=num_layers, batch_first=True, bidirectional=False)

    def forward(self, input_ids: torch.Tensor):
        emb = self.embedding(input_ids)
        outputs, (hn, cn) = self.lstm(emb)
        return outputs, (hn, cn)


class Decoder(nn.Module):
    def __init__(self, slot_vocab_size: int, embed_dim: int = 64, hidden_dim: int = 256, num_layers: int = 1, dropout: float = 0.5):
        super().__init__()
        self.embedding = nn.Embedding(slot_vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.slot_head = nn.Linear(hidden_dim, slot_vocab_size)

    def forward(self, slot_inputs: torch.Tensor, hidden: Optional[tuple] = None):
        emb = self.embedding(slot_inputs)
        outputs, hidden = self.lstm(emb, hidden)
        outputs = self.dropout(outputs)
        logits = self.slot_head(outputs)
        return logits, hidden


class Seq2SeqJoint(nn.Module):
    def __init__(self, encoder: Encoder, decoder: Decoder, intent_hidden_dim: int = 128, num_intents: int = 20):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        # intent head from encoder final hidden state
        self.intent_head = nn.Linear(encoder.lstm.hidden_size, num_intents)

    def forward(self, src_ids: torch.Tensor, tgt_slot_ids: torch.Tensor = None, teacher_forcing: float = 0.5):
        # encode
        enc_out, (hn, cn) = self.encoder(src_ids)
        # predict intent from last encoder hidden state (hn)
        # hn shape: (num_layers * num_directions, B, H)
        last_h = hn[-1]
        intent_logits = self.intent_head(last_h)

        # decode slots (greedy by default if tgt not provided)
        B = src_ids.size(0)
        max_len = tgt_slot_ids.size(1) if tgt_slot_ids is not None else src_ids.size(1)
        device = src_ids.device
        outputs = []

        # start with BOS (assumed id 1) or zeros
        cur_input = torch.full((B, 1), 1, dtype=torch.long, device=device)
        hidden = (hn, cn)
        for t in range(max_len):
            logits, hidden = self.decoder(cur_input, hidden)
            # logits: (B, 1, V)
            outputs.append(logits)
            # next input
            if tgt_slot_ids is not None and torch.rand(1).item() < teacher_forcing:
                cur_input = tgt_slot_ids[:, t].unsqueeze(1)
            else:
                cur_input = logits.argmax(dim=-1)
        slot_logits = torch.cat(outputs, dim=1)
        return slot_logits, intent_logits
