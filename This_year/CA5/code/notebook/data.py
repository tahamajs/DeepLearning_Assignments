# data.py - Data loading and preprocessing functions

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision.models as models
from transformers import AutoTokenizer
from datasets import load_dataset
import os
from PIL import Image
import glob
import urllib.request
import zipfile

def download_market1501(data_dir="./data"):
    """
    Download Market-1501 dataset for person re-identification
    """
    os.makedirs(data_dir, exist_ok=True)

    # Market-1501 URLs (you may need to adjust these based on current availability)
    urls = {
        'train': 'https://drive.google.com/uc?id=1v7TKbUQHqJ9oK0n0zK7t8XzY9wQ1aBc',  # Placeholder - actual URL may vary
        'test': 'https://drive.google.com/uc?id=1v7TKbUQHqJ9oK0n0zK7t8XzY9wQ1aBd',   # Placeholder - actual URL may vary
        'query': 'https://drive.google.com/uc?id=1v7TKbUQHqJ9oK0n0zK7t8XzY9wQ1aBe'   # Placeholder - actual URL may vary
    }

    print("Note: Market-1501 requires manual download from:")
    print("https://www.kaggle.com/datasets/pengcw1/market-1501")
    print("or official website. Due to licensing, we can't auto-download.")
    print("Please download and extract to './data/Market-1501/' directory")

    # Alternative: Use a smaller demo dataset or synthetic data
    print("Using synthetic data for demonstration...")
    return "./data/synthetic_reid"

class ReIDDataset(Dataset):
    def __init__(self, data_path, transform=None):
        """
        Load images from folders.
        Structure expected: root/class_id/image.jpg
        """
        self.transform = transform
        self.image_paths = []
        self.labels = []
        self.classes = []

        # Walk through directories
        if os.path.exists(data_path):
            self.classes = sorted(os.listdir(data_path))
            for label, class_name in enumerate(self.classes):
                class_dir = os.path.join(data_path, class_name)
                if os.path.isdir(class_dir):
                    for img_file in os.listdir(class_dir):
                        if img_file.endswith(('.jpg', '.png', '.jpeg')):
                            self.image_paths.append(os.path.join(class_dir, img_file))
                            self.labels.append(label)
        else:
            # For demo, create dummy data
            print("Data path not found, using dummy data")
            self.classes = [f"class_{i}" for i in range(10)]
            self.image_paths = [f"dummy_{i}.jpg" for i in range(100)]
            self.labels = [i % 10 for i in range(100)]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        try:
            img = Image.open(img_path).convert('RGB')
        except:
            # Dummy image if file not found
            img = Image.new('RGB', (256, 128), color=(128, 128, 128))

        if self.transform:
            img = self.transform(img)

        return img, label

# Text-to-SQL data functions
def format_example(example, tokenizer):
    SYSTEM_PROMPT = "You are a Text-to-SQL assistant. Output ONLY the SQL query. Do not add explanations."

    user_content = f"Schema:\n{example['schema']}\n\nQuestion:\n{example['sql_prompt']}"

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": example['sql']} # The 'Gold' SQL
    ]

    # Apply template WITHOUT tokenizing yet to find boundaries
    full_text = tokenizer.apply_chat_template(messages, tokenize=False)

    # Simple logic to separate Prompt vs Answer for masking
    # Note: This depends on the specific chat template of the base model
    prompt_part = tokenizer.apply_chat_template(messages[:-1], tokenize=False, add_generation_prompt=True)
    answer_part = example['sql']

    return prompt_part, answer_part

def normalize_sql(query):
    query = query.lower()
    query = query.replace("`", "").replace(";", "")
    query = " ".join(query.split()) # Fix whitespace
    return query

def exact_match_score(pred, truth):
    return 1 if normalize_sql(pred) == normalize_sql(truth) else 0