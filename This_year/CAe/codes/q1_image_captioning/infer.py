"""
Inference and BLEU evaluation for Q1 image captioning.
"""
import torch
from .tokenizer import Tokenizer
from .models import EncoderCNN, DecoderLSTM
from .metrics import compute_bleu_scores
from .dataset import ImageCaptionDataset
from .collate import collate_fn
from utils.utils import load_checkpoint
import torchvision.transforms as T
from pathlib import Path
import json
import pandas as pd
from PIL import Image

def load_tokenizer(path):
    with open(path, 'r') as f:
        word2idx = json.load(f)
    tk = Tokenizer()
    tk.word2idx = word2idx
    tk.idx2word = {int(v):k for k,v in word2idx.items()}
    return tk

def run_inference(images_dir, ckpt_dir, n=10, device='cpu'):
    # Load tokenizer
    tokenizer = load_tokenizer(Path(ckpt_dir)/'tokenizer.json')
    enc = EncoderCNN(encoded_dim=512, freeze_backbone=True).to(device)
    dec = DecoderLSTM(vocab_size=tokenizer.vocab_size(), embed_dim=300, decoder_dim=512, encoder_dim=512).to(device)
    load_checkpoint(enc, str(Path(ckpt_dir)/'encoder.pt'))
    load_checkpoint(dec, str(Path(ckpt_dir)/'decoder.pt'))
    enc.eval(); dec.eval()

    # Load test images and captions
    df = pd.read_csv(Path(images_dir).parent/'synth_test.csv') if (Path(images_dir).parent/'synth_test.csv').exists() else None
    if df is None:
        # fallback: sample from train set
        df = pd.DataFrame(list(Path(images_dir).glob('synth_*.png')), columns=['image'])
        df['caption'] = 'A cat on the sofa'
    sample_df = df.sample(n=min(n, len(df))).reset_index(drop=True)
    transform = T.Compose([T.ToTensor(), T.Resize((224,224))])

    references = []
    hypotheses = []
    for i, row in sample_df.iterrows():
        img = Image.open(Path(images_dir)/row['image']).convert('RGB')
        img_t = transform(img).unsqueeze(0).to(device)
        enc_out = enc(img_t)
        seqs = dec.greedy_decode(enc_out, start_token=tokenizer.word2idx['<start>'], end_token=tokenizer.word2idx['<end>'], max_len=20)
        pred_tokens = seqs[0]
        pred_text = tokenizer.sequence_to_text(pred_tokens)
        references.append(row['caption'])
        hypotheses.append(pred_text)
    bleu = compute_bleu_scores(references, hypotheses)
    return {'references': references, 'hypotheses': hypotheses, 'bleu': bleu}

if __name__ == '__main__':
    images_dir = 'q1_image_captioning/images'
    ckpt_dir = str(Path(images_dir)/'checkpoints')
    result = run_inference(images_dir, ckpt_dir, n=10, device='cpu')
    print('BLEU scores:', result['bleu'])
    for ref, hyp in zip(result['references'], result['hypotheses']):
        print('GT:', ref)
        print('PR:', hyp)
        print('---')
