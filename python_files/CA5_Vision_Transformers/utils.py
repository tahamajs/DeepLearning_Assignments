# utils.py - Utility functions

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import math

def visualize_attention(model, img_tensor, original_image):
    model.eval()
    with torch.no_grad():
        output = model(img_tensor.unsqueeze(0))
    
    # Retrieve stored attention map from the MHSA layer
    attn_map = model.mhsa.last_attention_map # Shape: (1, heads, pixels, pixels)
    
    # Average over heads and reshape to spatial dimensions
    H_feat = W_feat = int(math.sqrt(attn_map.shape[-1]))
    attn_map = attn_map.mean(dim=1).view(H_feat, W_feat, H_feat, W_feat)
    
    # Project specific pixel attention or global attention
    # For simplicity, average over all positions
    global_attn = attn_map.mean(dim=(0,1))
    
    # Resize to image size
    global_attn = F.interpolate(global_attn.unsqueeze(0).unsqueeze(0), size=(256, 128), mode='bilinear').squeeze()
    
    # Normalize
    global_attn = (global_attn - global_attn.min()) / (global_attn.max() - global_attn.min())
    
    # Overlay heatmap on original_image
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(original_image)
    plt.title("Original Image")
    
    plt.subplot(1, 2, 2)
    plt.imshow(original_image)
    plt.imshow(global_attn.cpu(), alpha=0.5, cmap='jet')
    plt.title("Attention Heatmap")
    plt.show()

def post_process_sql(text):
    # Extract only the SQL part
    # Look for SELECT ... ;
    if "SELECT" in text:
        start = text.find("SELECT")
        end = text.find(";", start)
        if end != -1:
            return text[start:end+1]
        return text[start:]
    return text