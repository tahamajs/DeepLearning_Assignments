# optimization.py - Model optimization and deployment functions

import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import numpy as np

def benchmark_inference_speed(model, dataloader, device, model_name="Model", num_runs=10):
    """
    Benchmark model inference speed
    """
    model.eval()
    model.to(device)

    # Warm up
    with torch.no_grad():
        for images, _ in dataloader:
            _ = model(images.to(device))
            break

    # Benchmark
    times = []
    total_samples = 0

    with torch.no_grad():
        for run in range(num_runs):
            start_time = time.time()

            for images, _ in dataloader:
                images = images.to(device)
                _ = model(images)
                total_samples += len(images)

            end_time = time.time()
            times.append(end_time - start_time)

    avg_time = np.mean(times)
    std_time = np.std(times)
    throughput = total_samples / avg_time  # samples per second

    print(f"{model_name} Benchmark Results:")
    print(f"  Average time: {avg_time:.4f} seconds")
    print(f"  Std deviation: {std_time:.4f} seconds")
    print(f"  Throughput: {throughput:.2f} samples/second")

    return {
        'avg_time': avg_time,
        'std_time': std_time,
        'throughput': throughput
    }

def quantize_model_dynamic(model, dtype=torch.qint8):
    """
    Apply dynamic quantization to reduce model size
    """
    # Dynamic quantization for linear layers
    quantized_model = torch.quantization.quantize_dynamic(
        model, {torch.nn.Linear}, dtype=dtype
    )

    # Calculate size reduction
    original_size = sum(p.numel() * p.element_size() for p in model.parameters())
    quantized_size = sum(p.numel() * p.element_size() for p in quantized_model.parameters())

    print(f"Original model size: {original_size / 1024 / 1024:.2f} MB")
    print(f"Quantized model size: {quantized_size / 1024 / 1024:.2f} MB")
    print(".2f")

    return quantized_model

def apply_mixed_precision_training():
    """
    Setup mixed precision training with automatic mixed precision (AMP)
    """
    try:
        from torch.cuda.amp import GradScaler, autocast
        scaler = GradScaler()

        print("Mixed precision training enabled")

        return scaler, autocast, True
    except ImportError:
        print("Mixed precision not available, using standard training")
        return None, lambda: torch.enable_grad(), False

def create_knowledge_distillation_teacher_student(teacher_model, student_model, temperature=3.0):
    """
    Setup knowledge distillation from teacher to student model
    """
    def distillation_loss(student_logits, teacher_logits, true_labels, temperature, alpha=0.5):
        """
        Compute distillation loss combining hard and soft targets
        """
        # Soft targets from teacher
        teacher_probs = F.softmax(teacher_logits / temperature, dim=1)
        student_log_probs = F.log_softmax(student_logits / temperature, dim=1)
        soft_loss = F.kl_div(student_log_probs, teacher_probs, reduction='batchmean') * (temperature ** 2)

        # Hard targets from true labels
        hard_loss = F.cross_entropy(student_logits, true_labels)

        # Combine losses
        return alpha * soft_loss + (1 - alpha) * hard_loss

    def train_step_kd(student_model, teacher_model, images, labels, optimizer, temperature, alpha=0.5):
        student_model.train()
        teacher_model.eval()

        with torch.no_grad():
            teacher_logits = teacher_model(images)

        student_logits = student_model(images)
        loss = distillation_loss(student_logits, teacher_logits, labels, temperature, alpha)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return loss.item()

    return distillation_loss, train_step_kd

def export_to_onnx(model, sample_input, onnx_path="model.onnx"):
    """
    Export model to ONNX format for deployment
    """
    try:
        torch.onnx.export(
            model,
            sample_input,
            onnx_path,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
        )
        print(f"Model exported to {onnx_path}")
        return True
    except Exception as e:
        print(f"ONNX export failed: {e}")
        return False

def comprehensive_optimization_analysis():
    """
    Run comprehensive optimization analysis comparing different techniques
    """
    print("Comprehensive Model Optimization Analysis")
    print("=" * 50)

    # This would typically compare different optimization techniques
    # For now, just show the framework

    optimizations = {
        'Quantization': 'Reduces model size and inference time',
        'Pruning': 'Removes unnecessary weights',
        'Knowledge Distillation': 'Transfers knowledge from large to small models',
        'Mixed Precision': 'Uses float16 for faster computation',
        'ONNX Export': 'Enables cross-platform deployment'
    }

    for opt_name, description in optimizations.items():
        print(f"• {opt_name}: {description}")

    print("\nTo implement these optimizations:")
    print("1. Use quantize_model_dynamic() for quantization")
    print("2. Use apply_mixed_precision_training() for faster training")
    print("3. Use create_knowledge_distillation_teacher_student() for model compression")
    print("4. Use export_to_onnx() for deployment")
    print("5. Use benchmark_inference_speed() to measure improvements")

    return optimizations