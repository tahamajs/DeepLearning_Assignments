# evaluation.py - Evaluation and analysis functions

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import normalize

def extract_features(model, dataloader, device):
    """
    Extract features from a model for Re-ID evaluation
    """
    model.eval()
    features = []
    labels = []

    with torch.no_grad():
        for imgs, lbls in dataloader:
            imgs = imgs.to(device)
            # For Re-ID, we use features before final classification
            feat = model.stem(imgs)
            feat = model.layer1(feat)
            feat = model.layer2(feat)
            feat = model.layer3(feat)
            feat = model.layer4(feat)
            feat = model.avgpool(feat)
            feat = torch.flatten(feat, 1)
            features.append(feat.cpu())
            labels.append(lbls)

    features = torch.cat(features, dim=0)
    labels = torch.cat(labels, dim=0)
    return features.numpy(), labels.numpy()

def compute_cmc_and_map(query_features, query_labels, gallery_features, gallery_labels, k=10):
    """
    Compute CMC (Cumulative Matching Characteristics) and mAP for Re-ID
    """
    # Normalize features
    query_features = normalize(query_features, axis=1)
    gallery_features = normalize(gallery_features, axis=1)

    num_queries = len(query_features)
    cmc_scores = np.zeros(k)
    ap_scores = []

    for i in range(num_queries):
        # Compute similarities
        similarities = np.dot(gallery_features, query_features[i])

        # Get ranking
        indices = np.argsort(similarities)[::-1]
        ranked_labels = gallery_labels[indices]

        # CMC: Check if correct label appears in top-k
        correct_label = query_labels[i]
        correct_indices = np.where(ranked_labels == correct_label)[0]

        if len(correct_indices) > 0:
            rank = correct_indices[0] + 1  # 1-based ranking
            if rank <= k:
                cmc_scores[rank-1:] += 1

        # mAP: Average precision for this query
        relevant = (ranked_labels == correct_label).astype(int)
        if np.sum(relevant) > 0:
            ap = average_precision_score(relevant, similarities[indices])
            ap_scores.append(ap)

    cmc_scores /= num_queries
    map_score = np.mean(ap_scores) if ap_scores else 0.0

    return cmc_scores, map_score

def plot_cmc_curve(cmc_scores, model_name, k=10):
    """
    Plot CMC curve for Re-ID evaluation
    """
    ranks = np.arange(1, k+1)
    plt.figure(figsize=(10, 6))
    plt.plot(ranks, cmc_scores * 100, 'o-', linewidth=2, markersize=8)
    plt.xlabel('Rank')
    plt.ylabel('Identification Rate (%)')
    plt.title(f'CMC Curve - {model_name}')
    plt.grid(True, alpha=0.3)
    plt.xticks(ranks)
    plt.ylim(0, 100)
    plt.show()

def evaluate_reid_advanced(model, test_loader, device, model_name="Model"):
    """
    Comprehensive Re-ID evaluation with CMC and mAP
    """
    print(f"Evaluating {model_name}...")

    # Extract features
    features, labels = extract_features(model, test_loader, device)

    # For simplicity, use same data as query and gallery
    # In practice, you'd have separate query/gallery sets
    query_features, gallery_features = features, features
    query_labels, gallery_labels = labels, labels

    # Compute metrics
    cmc_scores, map_score = compute_cmc_and_map(
        query_features, query_labels,
        gallery_features, gallery_labels, k=10
    )

    print(".2f")
    print(".2f")
    print(".2f")

    # Plot CMC curve
    plot_cmc_curve(cmc_scores, model_name)

    return cmc_scores, map_score