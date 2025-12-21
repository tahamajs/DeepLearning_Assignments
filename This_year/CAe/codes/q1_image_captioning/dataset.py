"""
Dataset and DataLoader utilities for Image Captioning (PyTorch)
"""
from torch.utils.data import Dataset
from PIL import Image
import os

class ImageCaptionDataset(Dataset):
    def __init__(self, df, images_root, tokenizer, transforms=None):
        """df: DataFrame with columns ['image', 'caption']"""
        self.df = df.reset_index(drop=True)
        self.images_root = images_root
        self.tokenizer = tokenizer
        self.transforms = transforms

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.loc[idx]
        img_path = os.path.join(self.images_root, row['image'])
        image = Image.open(img_path).convert('RGB')
        if self.transforms is not None:
            image = self.transforms(image)
        caption = row['caption']
        seq = self.tokenizer.text_to_sequence(caption)
        return image, seq, len(seq)

# collate_fn will be implemented in train scripts to pad sequences
