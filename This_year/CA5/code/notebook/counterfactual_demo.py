# Helper script for Counterfactual Attention demo
import torch
import torch.nn.functional as F


def topk_spatial_mask_from_attention(attention_map, Hf, Wf, mask_pct=0.4):
    # attention_map: (T,) or (Hf*Wf,) flattened
    flat = attention_map.view(-1)
    k = max(1, int(flat.numel() * mask_pct))
    vals, idxs = torch.topk(flat, k)
    mask = torch.zeros_like(flat).bool()
    mask[idxs] = True
    return mask.view(Hf, Wf)
