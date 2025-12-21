"""
Training script (small smoke-run) for Image Captioning.
This script is intended for quick local tests and demos.
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as T
from .dataset import ImageCaptionDataset
from .tokenizer import Tokenizer
from .models import EncoderCNN, DecoderLSTM
from .collate import collate_fn
from utils.utils import save_figure, make_fig_name, save_asset_manifest
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import numpy as np


def make_synthetic_dataset(tokenizer, images_dir, n=100):
    # simple synthetic images and captions
    from pathlib import Path
    import pandas as pd
    Path(images_dir).mkdir(parents=True, exist_ok=True)
    data = []
    for i in range(n):
        caption = np.random.choice(['A cat on the sofa', 'A dog in the park', 'A car on the street'])
        img = Image.new('RGB', (224,224), (int(50+i)%255,100,150))
        d = ImageDraw.Draw(img)
        d.text((10,10), f'Img {i+1}', fill=(255,255,255))
        fname = f'synth_{i+1:04d}.png'
        img.save(Path(images_dir)/fname)
        data.append({'image':fname, 'caption':caption})
    df = pd.DataFrame(data)
    return df


def train_smoke(device='cpu'):
    images_dir = 'q1_image_captioning/images'
    tokenizer = Tokenizer()
    # tiny corpus
    captions = ['A cat on the sofa', 'A dog in the park', 'A car on the street'] * 40
    tokenizer.build_vocab(captions, min_freq=1)

    df = make_synthetic_dataset(tokenizer, images_dir, n=120)
    transform = T.Compose([T.ToTensor(), T.Resize((224,224))])
    ds = ImageCaptionDataset(df, images_dir, tokenizer, transforms=transform)
    dl = DataLoader(ds, batch_size=8, shuffle=True, collate_fn=collate_fn)

    enc = EncoderCNN(encoded_dim=512, freeze_backbone=True).to(device)
    dec = DecoderLSTM(vocab_size=tokenizer.vocab_size(), embed_dim=300, decoder_dim=512, encoder_dim=512).to(device)

    optim = torch.optim.Adam(list(dec.parameters()), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer.word2idx['<pad>'])

    losses = []
    for epoch in range(3):
        dec.train()
        for imgs, caps, lengths in dl:
            imgs = imgs.to(device)
            caps = caps.to(device)
            enc_out = enc(imgs)
            logits, _ = dec(caps, enc_out)
            # logits: (batch, seq_len, vocab)
            batch, seq_len, vocab = logits.size()
            logits_flat = logits.reshape(batch*seq_len, vocab)
            targets = caps[:,1:].reshape(-1)
            loss = loss_fn(logits_flat, targets)
            optim.zero_grad(); loss.backward(); optim.step()
            losses.append(loss.item())
    # plot loss
    fig, ax = plt.subplots(figsize=(3.5,2.5))
    ax.plot(losses)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Loss')
    ax.set_title('Training loss (smoke)')
    fig_path = make_fig_name('captioning','loss','smoke', ext='png', images_dir=images_dir)
    save_figure(fig, fig_path)

    # perform greedy decode on 5 samples and save their images with predicted captions
    dec.eval()
    sample_df = df.sample(5).reset_index(drop=True)
    manifest = [{'filename': fig_path, 'width_in':3.5, 'height_in':2.5, 'dpi':300, 'caption_placeholder':'Smoke training loss'}]
    for i, row in sample_df.iterrows():
        img = Image.open(Path(images_dir)/row['image']).convert('RGB')
        img_t = T.ToTensor()(img).unsqueeze(0).to(device)
        enc_out = enc(img_t)
        seqs = dec.greedy_decode(enc_out, start_token=tokenizer.word2idx['<start>'], end_token=tokenizer.word2idx['<end>'], max_len=20)
        pred_tokens = seqs[0]
        pred_text = tokenizer.sequence_to_text(pred_tokens)
        # render
        out_img = Image.new('RGB', (800,200), (255,255,255))
        d = ImageDraw.Draw(out_img)
        d.text((10,10), f'GT: {row["caption"]}', fill=(0,0,0))
        d.text((10,40), f'Pred: {pred_text}', fill=(0,0,0))
        out_path = Path(images_dir)/f'q1_pred_sample_{i+1:02d}.png'
        out_img.save(out_path)
        manifest.append({'filename': str(out_path), 'width_in':8.0, 'height_in':2.0, 'dpi':300, 'caption_placeholder':f'Example {i+1} prediction'})

    save_asset_manifest(manifest, images_dir)

    # Save tokenizer and model checkpoints for later inference
    import json
    ckpt_dir = Path(images_dir) / 'checkpoints'
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    tokenizer_path = ckpt_dir / 'tokenizer.json'
    with open(tokenizer_path, 'w') as f:
        json.dump(tokenizer.word2idx, f)

    from utils.utils import save_checkpoint
    save_checkpoint(dec, str(ckpt_dir / 'decoder.pt'))
    save_checkpoint(enc, str(ckpt_dir / 'encoder.pt'))

    print('Smoke training complete. Assets and checkpoints saved to', images_dir)

if __name__ == '__main__':
    train_smoke(device='cpu')
