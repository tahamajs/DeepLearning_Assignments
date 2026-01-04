# utils.py - Utility functions

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import math
import numpy as np
from PIL import Image

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

def visualize_feature_maps(model, image_tensor, layer_name='layer4', num_channels=16):
    """
    Visualize feature maps from intermediate layers
    """
    activation = {}

    def hook_fn(module, input, output):
        activation['value'] = output.detach()

    # Register hook
    layer = getattr(model, layer_name)
    hook = layer.register_forward_hook(hook_fn)

    # Forward pass
    with torch.no_grad():
        _ = model(image_tensor.unsqueeze(0))

    hook.remove()

    # Get feature maps
    feature_maps = activation['value'][0]  # Remove batch dimension

    # Plot first num_channels feature maps
    fig, axes = plt.subplots(4, 4, figsize=(12, 12))
    fig.suptitle(f'Feature Maps - {layer_name}')

    for i in range(min(num_channels, feature_maps.shape[0])):
        ax = axes[i // 4, i % 4]
        feature_map = feature_maps[i].cpu().numpy()
        # Normalize for visualization
        feature_map = (feature_map - feature_map.min()) / (feature_map.max() - feature_map.min())
        ax.imshow(feature_map, cmap='viridis')
        ax.axis('off')
        ax.set_title(f'Channel {i}')

    plt.tight_layout()
    plt.show()

def compute_saliency_map(model, image_tensor, target_class):
    """
    Compute saliency map using gradients
    """
    model.eval()
    image_tensor.requires_grad_()

    # Forward pass
    output = model(image_tensor.unsqueeze(0))
    score = output[0, target_class]

    # Backward pass
    model.zero_grad()
    score.backward()

    # Get gradients
    saliency = image_tensor.grad.abs().sum(dim=0)

    # Normalize
    saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min())

    return saliency

def occlusion_sensitivity_analysis(model, image_tensor, target_class, patch_size=16, stride=8):
    """
    Analyze model sensitivity to occlusions
    """
    model.eval()
    height, width = image_tensor.shape[1], image_tensor.shape[2]

    # Get baseline prediction
    with torch.no_grad():
        baseline_output = model(image_tensor.unsqueeze(0))
        baseline_confidence = torch.softmax(baseline_output, dim=1)[0, target_class].item()

    sensitivity_map = torch.zeros((height, width))

    for y in range(0, height - patch_size + 1, stride):
        for x in range(0, width - patch_size + 1, stride):
            # Create occluded image
            occluded_image = image_tensor.clone()
            occluded_image[:, y:y+patch_size, x:x+patch_size] = 0  # Black occlusion

            # Get prediction
            with torch.no_grad():
                output = model(occluded_image.unsqueeze(0))
                confidence = torch.softmax(output, dim=1)[0, target_class].item()

            # Store sensitivity (drop in confidence)
            sensitivity_map[y:y+patch_size, x:x+patch_size] = baseline_confidence - confidence

    return sensitivity_map

def analyze_attention_patterns(model, dataloader, num_samples=10):
    """
    Analyze attention patterns across multiple samples
    """
    model.eval()
    attention_maps = []

    with torch.no_grad():
        for i, (images, labels) in enumerate(dataloader):
            if i >= num_samples:
                break

            output = model(images[:1])  # Single image
            if hasattr(model, 'mhsa') and hasattr(model.mhsa, 'last_attention_map'):
                attn_map = model.mhsa.last_attention_map
                attention_maps.append(attn_map.cpu())

    if attention_maps:
        # Average attention maps
        avg_attention = torch.stack(attention_maps).mean(dim=0)
        print(f"Average attention shape: {avg_attention.shape}")

        # Visualize average attention
        plt.figure(figsize=(8, 6))
        plt.imshow(avg_attention[0, 0].numpy(), cmap='viridis')
        plt.title('Average Attention Pattern')
        plt.colorbar()
        plt.show()

    return attention_maps

def comprehensive_interpretability_analysis(model, image_tensor, original_image, model_name="Model"):
    """
    Run comprehensive interpretability analysis
    """
    print(f"Running interpretability analysis for {model_name}...")

    # 1. Attention visualization (if available)
    if hasattr(model, 'mhsa'):
        print("1. Visualizing attention...")
        visualize_attention(model, image_tensor, original_image)

    # 2. Feature map visualization
    print("2. Visualizing feature maps...")
    visualize_feature_maps(model, image_tensor)

    # 3. Saliency map
    print("3. Computing saliency map...")
    target_class = model(image_tensor.unsqueeze(0)).argmax().item()
    saliency = compute_saliency_map(model, image_tensor, target_class)

    plt.figure(figsize=(8, 6))
    plt.imshow(original_image)
    plt.imshow(saliency.cpu(), alpha=0.5, cmap='hot')
    plt.title('Saliency Map')
    plt.show()

    # 4. Occlusion sensitivity
    print("4. Analyzing occlusion sensitivity...")
    sensitivity = occlusion_sensitivity_analysis(model, image_tensor, target_class)

    plt.figure(figsize=(8, 6))
    plt.imshow(original_image)
    plt.imshow(sensitivity.cpu(), alpha=0.7, cmap='Reds')
    plt.title('Occlusion Sensitivity')
    plt.show()

    return {
        'saliency': saliency,
        'sensitivity': sensitivity
    }