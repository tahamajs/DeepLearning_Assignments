"""
Data loading utilities for UrbanSound experiments
"""
import os
import pandas as pd
import librosa
import numpy as np

class UrbanSoundDataset:
    def __init__(self, metadata_csv, audio_root, sr=16000, transform=None):
        self.metadata = pd.read_csv(metadata_csv)
        self.audio_root = audio_root
        self.sr = sr
        self.transform = transform

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        path = os.path.join(self.audio_root, row['slice_file_name'])
        wav, _ = librosa.load(path, sr=self.sr)
        label = int(row['classID'])
        if self.transform:
            wav = self.transform(wav)
        return wav, label
