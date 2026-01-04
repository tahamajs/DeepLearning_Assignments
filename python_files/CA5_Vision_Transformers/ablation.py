# ablation.py - Ablation study functions

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

def create_ablation_model(model_type, **kwargs):
    """
    Create different model variants for ablation studies
    """
    if model_type == 'resnet_no_pretrain':
        model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=False)
        model.fc = nn.Linear(model.fc.in_features, kwargs.get('num_classes', 10))
    elif model_type == 'resnet_pretrained':
        from models import get_resnet50
        model = get_resnet50(kwargs.get('num_classes', 10))
    elif model_type.startswith('botnet'):
        from models import BotNet50
        model = BotNet50(kwargs.get('num_classes', 10))
        # Modify attention heads (simplified - would need to modify MHSA class)
        if model_type == 'botnet_8':
            # This would require modifying the MHSA heads parameter
            pass
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    return model

def ablation_study_reid(model_configs, train_loader, test_loader, epochs=3):
    """
    Run ablation study comparing different model configurations
    """
    results = {}

    for config_name, config in model_configs.items():
        print(f"\nRunning ablation: {config_name}")
        print("-" * 50)

        # Create model
        model = create_ablation_model(**config)
        optimizer = optim.Adam(model.parameters(), lr=1e-4)

        # Quick training
        from train import train_model, evaluate_model
        train_model(model, optimizer, train_loader, epochs=epochs)
        accuracy = evaluate_model(model, test_loader)

        results[config_name] = {
            'accuracy': accuracy,
            'config': config
        }

        print(".2f")

    return results

def plot_ablation_results(results):
    """
    Plot ablation study results
    """
    model_names = list(results.keys())
    accuracies = [results[name]['accuracy'] for name in model_names]

    plt.figure(figsize=(12, 6))
    bars = plt.bar(model_names, accuracies)
    plt.xlabel('Model Configuration')
    plt.ylabel('Accuracy (%)')
    plt.title('Ablation Study Results')
    plt.xticks(rotation=45, ha='right')

    # Add value labels on bars
    for bar, acc in zip(bars, accuracies):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                '.1f', ha='center', va='bottom')

    plt.tight_layout()
    plt.show()

def ablation_llada_masking_schedules():
    """
    Ablation study for different masking schedules in LLaDA
    """
    def cosine_schedule(t):
        return 0.5 * (1 + np.cos(np.pi * t))

    def linear_schedule(t):
        return 1 - t

    def exponential_schedule(t):
        return np.exp(-3 * t)

    # Test schedules
    t_values = np.linspace(0, 1, 100)

    plt.figure(figsize=(10, 6))
    plt.plot(t_values, [cosine_schedule(t) for t in t_values], label='Cosine', linewidth=2)
    plt.plot(t_values, [linear_schedule(t) for t in t_values], label='Linear', linewidth=2)
    plt.plot(t_values, [exponential_schedule(t) for t in t_values], label='Exponential', linewidth=2)

    plt.xlabel('Time Step (t)')
    plt.ylabel('Masking Probability')
    plt.title('LLaDA Masking Schedule Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

    return {
        'cosine': cosine_schedule,
        'linear': linear_schedule,
        'exponential': exponential_schedule
    }