# main.py - Main script with setup and examples

# Install necessary libraries
# !pip install torch torchvision transformers peft datasets bitsandbytes accelerate scikit-learn matplotlib

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision.models as models
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
import numpy as np
import matplotlib.pyplot as plt
import copy
import math
import os
from PIL import Image
import glob

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Import from modules
from data import ReIDDataset, format_example, normalize_sql, exact_match_score, download_market1501
from models import get_resnet50, MHSA, BotNet50
from utils import visualize_attention, post_process_sql
from train import train_model, evaluate_model, train_step, evaluate_pipeline
from generate import generate_block_diffusion

# Part 1: Re-ID Setup
# Aggressive Augmentation for small datasets
train_transforms = T.Compose([
    T.Resize((256, 128)), # Standard Re-ID size
    T.RandomHorizontalFlip(),
    T.RandomRotation(10),
    T.ColorJitter(brightness=0.2, contrast=0.2),
    T.ToTensor(),
    T.RandomErasing(p=0.5), # Helps with occlusion - applied after ToTensor
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

test_transforms = T.Compose([
    T.Resize((256, 128)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Initialize Datasets and DataLoaders
# Assuming data_path is set, e.g., "/path/to/reid_data"
# For demo, use dummy
data_path = download_market1501()
train_dataset = ReIDDataset("dummy_train", transform=train_transforms)
test_dataset = ReIDDataset("dummy_test", transform=test_transforms)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Initialize models
num_classes = len(train_dataset.classes)
resnet_model = get_resnet50(num_classes)
botnet_model = BotNet50(num_classes)

# Optimizers
resnet_optimizer = optim.Adam(resnet_model.parameters(), lr=1e-4)
botnet_optimizer = optim.Adam(botnet_model.parameters(), lr=1e-4)

# Part 2: LLaDA Setup
# Load Dataset
dataset = load_dataset("gretelai/synthetic_text_to_sql")

# Load Model with Quantization
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4"
)

model_name = "GSAI-ML/LLaDA-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",
    use_cache=False # Important for Diffusion training
)

# Apply LoRA
peft_config = LoraConfig(
    r=16, lora_alpha=32, target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05, bias="none", task_type="CAUSAL_LM"
)
model = get_peft_model(model, peft_config)

optimizer = optim.AdamW(model.parameters(), lr=2e-4)

# Example usage
if __name__ == "__main__":
    # Train Re-ID models
    print("Training ResNet50...")
    train_model(resnet_model, resnet_optimizer, train_loader)
    resnet_acc = evaluate_model(resnet_model, test_loader)

    print("Training BotNet50...")
    train_model(botnet_model, botnet_optimizer, train_loader)
    botnet_acc = evaluate_model(botnet_model, test_loader)

    print(".2f")

    # Example for LLaDA generation
    example = dataset['test'][0]
    prompt, _ = format_example(example, tokenizer)
    generated = generate_block_diffusion(model, tokenizer, prompt)
    print("Generated SQL:", generated)