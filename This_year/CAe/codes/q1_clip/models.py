"""
CLIP-style encoders and InfoNCE loss utilities
"""
import torch
import torch.nn as nn
import timm

class ImageEncoderViT(nn.Module):
    def __init__(self, model_name='vit_small_patch16_224', proj_dim=256, pretrained=True, freeze_backbone=True):
        super().__init__()
        self.vit = timm.create_model(model_name, pretrained=pretrained, num_classes=0, global_pool='avg')
        self.proj = nn.Linear(self.vit.num_features, proj_dim)
        self.norm = nn.LayerNorm(proj_dim)
        if freeze_backbone:
            for p in self.vit.parameters():
                p.requires_grad = False

    def forward(self, x):
        feat = self.vit(x)
        out = self.proj(feat)
        out = out / (out.norm(dim=-1, keepdim=True) + 1e-8)
        return out

class TextEncoderSimple(nn.Module):
    def __init__(self, vocab_size, embed_dim=300, proj_dim=256, hidden_dim=512):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.proj = nn.Linear(hidden_dim, proj_dim)

    def forward(self, x, lengths=None):
        emb = self.embed(x)
        out, _ = self.lstm(emb)
        feat = out[:, -1, :]
        out = self.proj(feat)
        out = out / (out.norm(dim=-1, keepdim=True) + 1e-8)
        return out

class InfoNCE(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = nn.Parameter(torch.tensor(temperature))

    def forward(self, img_emb, txt_emb):
        logits = torch.matmul(img_emb, txt_emb.t()) / self.temperature
        labels = torch.arange(img_emb.size(0), device=img_emb.device)
        loss_i = nn.CrossEntropyLoss()(logits, labels)
        loss_t = nn.CrossEntropyLoss()(logits.t(), labels)
        return (loss_i + loss_t) / 2
