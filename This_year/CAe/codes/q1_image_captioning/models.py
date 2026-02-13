"""
Encoder/Decoder skeletons with Attention for Image Captioning
"""
import torch
import torch.nn as nn
import torchvision.models as models
import warnings

class EncoderCNN(nn.Module):
    def __init__(self, encoded_dim=512, freeze_backbone=True, pretrained=True):
        super().__init__()
        # Prefer pretrained ImageNet weights, but gracefully fall back to
        # random initialization in offline/test environments.
        weights = None
        if pretrained:
            try:
                weights = models.VGG16_Weights.IMAGENET1K_V1
            except AttributeError:
                weights = None
        try:
            vgg = models.vgg16(weights=weights)
        except Exception as exc:
            warnings.warn(
                f"Could not load pretrained VGG16 weights ({exc}); falling back to random init.",
                RuntimeWarning,
            )
            vgg = models.vgg16(weights=None)
        modules = list(vgg.features.children())
        self.feature_extractor = nn.Sequential(*modules)
        if freeze_backbone:
            for p in self.feature_extractor.parameters():
                p.requires_grad = False
        self.pool = nn.AdaptiveAvgPool2d((7,7))
        self.project = nn.Linear(512*7*7, encoded_dim)

    def forward(self, x):
        feat = self.feature_extractor(x)
        feat = self.pool(feat)
        feat = feat.view(feat.size(0), -1)
        out = self.project(feat)
        return out

class Attention(nn.Module):
    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        super().__init__()
        self.enc_att = nn.Linear(encoder_dim, attention_dim)
        self.dec_att = nn.Linear(decoder_dim, attention_dim)
        self.full_att = nn.Linear(attention_dim, 1)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, encoder_out, decoder_hidden):
        # encoder_out: (batch, enc_dim), decoder_hidden: (batch, dec_dim)
        att1 = self.enc_att(encoder_out)
        att2 = self.dec_att(decoder_hidden)
        att = self.full_att(self.relu(att1 + att2))
        alpha = self.softmax(att)
        att_embedding = encoder_out * alpha
        return att_embedding, alpha

class DecoderLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim=300, decoder_dim=512, encoder_dim=512):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.attention = Attention(encoder_dim, decoder_dim, attention_dim=256)
        self.lstm = nn.LSTMCell(embed_dim + encoder_dim, decoder_dim)
        self.fc = nn.Linear(decoder_dim, vocab_size)

    def forward_step(self, word, encoder_out, states):
        embedded = self.embedding(word)
        dec_hidden, dec_cell = states
        att_embed, alpha = self.attention(encoder_out, dec_hidden)
        lstm_input = torch.cat([embedded, att_embed], dim=1)
        h, c = self.lstm(lstm_input, (dec_hidden, dec_cell))
        out = self.fc(h)
        return out, (h, c), alpha

    def forward(self, captions, encoder_out):
        """Teacher-forcing forward pass.

        captions: (batch, max_len) LongTensor with <start> and <end> tokens
        encoder_out: (batch, encoder_dim)
        returns: logits (batch, max_len-1, vocab_size) and attention weights list
        """
        device = captions.device
        batch_size = captions.size(0)
        max_len = captions.size(1)
        vocab_size = self.fc.out_features

        # initialize hidden states (simple zeros) — could use encoder projection
        h = torch.zeros(batch_size, self.lstm.hidden_size, device=device)
        c = torch.zeros(batch_size, self.lstm.hidden_size, device=device)

        inputs = captions[:, :-1]  # input tokens (excluding last token)
        targets = captions[:, 1:]

        logits = []
        alphas = []
        for t in range(inputs.size(1)):
            word = inputs[:, t]
            out, (h, c), alpha = self.forward_step(word, encoder_out, (h, c))
            logits.append(out.unsqueeze(1))
            alphas.append(alpha.unsqueeze(1))
        logits = torch.cat(logits, dim=1)  # (batch, seq_len, vocab)
        alphas = torch.cat(alphas, dim=1)
        return logits, alphas

    def greedy_decode(self, encoder_out, start_token, end_token, max_len=30):
        """Greedy decode a sequence given encoder outputs."""
        device = encoder_out.device
        batch_size = encoder_out.size(0)
        h = torch.zeros(batch_size, self.lstm.hidden_size, device=device)
        c = torch.zeros(batch_size, self.lstm.hidden_size, device=device)

        seqs = torch.full((batch_size, max_len), fill_value=0, dtype=torch.long, device=device)
        seqs[:,0] = start_token
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

        outputs = []
        for t in range(1, max_len):
            prev = seqs[:, t-1]
            out, (h, c), _ = self.forward_step(prev, encoder_out, (h, c))
            probs = torch.softmax(out, dim=1)
            next_tok = torch.argmax(probs, dim=1)
            seqs[:, t] = next_tok
            outputs.append(next_tok.unsqueeze(1))
            finished = finished | (next_tok == end_token)
            if finished.all():
                break
        # return sequences up to first end token per example as list
        result = []
        for i in range(batch_size):
            toks = seqs[i].tolist()
            # truncate at first end token
            if end_token in toks:
                toks = toks[:toks.index(end_token)+1]
            result.append(toks)
        return result
