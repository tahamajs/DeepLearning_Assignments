"""Train script skeleton for encoder-decoder joint model (no execution in this commit)."""
import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
import torch.nn as nn

from src.data.preprocess import ATISDataset, collate_fn, load_atis_examples, build_vocab, build_label_vocab
from src.models.encoder_decoder import Encoder, Decoder, Seq2SeqJoint


def train(args):
    train_examples = load_atis_examples(Path(args.data_dir) / "train.json")
    test_examples = load_atis_examples(Path(args.data_dir) / "test.json")
    word2id, id2word = build_vocab([ex["tokens"] for ex in train_examples])
    slot2id, id2slot = build_label_vocab([s for ex in train_examples for s in ex["slots"]])
    intents = sorted({ex["intent"] for ex in train_examples})
    intent2id = {intent: idx for idx, intent in enumerate(intents)}

    train_ds = ATISDataset(train_examples, word2id, slot2id, intent2id, add_bos_eos=True)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    enc = Encoder(len(word2id), embed_dim=args.enc_embed, hidden_dim=args.enc_hidden)
    dec = Decoder(len(slot2id), embed_dim=args.dec_embed, hidden_dim=args.dec_hidden)
    model = Seq2SeqJoint(enc, dec, num_intents=len(intent2id))
    model.to(device)

    slot_criterion = nn.CrossEntropyLoss(ignore_index=0)
    intent_criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Training loop left as an exercise; notebook contains a demo cell for a short run


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='data/processed')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--enc_embed', type=int, default=128)
    parser.add_argument('--enc_hidden', type=int, default=128)
    parser.add_argument('--dec_embed', type=int, default=64)
    parser.add_argument('--dec_hidden', type=int, default=256)
    parser.add_argument('--lr', type=float, default=1e-3)
    args = parser.parse_args()
    train(args)
