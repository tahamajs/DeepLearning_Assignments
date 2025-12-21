"""Train BiRNN slot filler (baseline).
Usage:
    python scripts/train_birnn.py --data_dir data/raw --out_dir outputs/birnn
"""
import argparse
import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from tqdm import tqdm

from src.data.preprocess import load_atis_examples, build_vocab, build_label_vocab, ATISDataset, collate_fn
from src.models.baseline import BiRNNSlotFiller
from src.utils.metrics import slot_f1, slot_classification_report


def train(args):
    train_examples = load_atis_examples(Path(args.data_dir) / "train.json")
    dev_examples = load_atis_examples(Path(args.data_dir) / "dev.json") if (Path(args.data_dir) / "dev.json").exists() else []
    test_examples = load_atis_examples(Path(args.data_dir) / "test.json")

    word2id, id2word = build_vocab([ex["tokens"] for ex in train_examples])
    slot2id, id2slot = build_label_vocab([s for ex in train_examples for s in ex["slots"]])

    train_ds = ATISDataset(train_examples, word2id, slot2id, {"unknown": 0})
    test_ds = ATISDataset(test_examples, word2id, slot2id, {"unknown": 0})

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BiRNNSlotFiller(len(word2id), embed_dim=args.embed_dim, hidden_dim=args.hidden_dim, num_labels=len(slot2id), bidirectional=args.bidirectional, dropout=args.dropout)
    model.to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=0)  # assume PAD is 0
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_f1 = 0.0
    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}"):
            input_ids = batch["input_ids"].to(device)
            slot_ids = batch["slot_ids"].to(device)
            optimizer.zero_grad()
            logits = model(input_ids)
            # logits: (B, T, C) -> reshape
            b, t, c = logits.shape
            loss = criterion(logits.view(-1, c), slot_ids.view(-1))
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch+1} train loss: {avg_loss:.4f}")

        # eval
        model.eval()
        preds, trues = [], []
        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch["input_ids"].to(device)
                slot_ids = batch["slot_ids"].to(device)
                logits = model(input_ids)
                pred = logits.argmax(dim=-1).cpu().tolist()
                true = slot_ids.cpu().tolist()
                # convert ids to labels when reporting outside
                preds.extend(pred)
                trues.extend(true)
        # convert ids to string labels
        id2slot = {v: k for k, v in slot2id.items()}
        pred_labels = [[id2slot.get(pid, "O") for pid in seq] for seq in preds]
        true_labels = [[id2slot.get(tid, "O") for tid in seq] for seq in trues]
        f1 = slot_f1(true_labels, pred_labels)
        print(f"Epoch {epoch+1} test slot F1: {f1:.4f}")
        if f1 > best_f1:
            best_f1 = f1
            out_dir = Path(args.out_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), out_dir / "best_birnn.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data/raw")
    parser.add_argument("--out_dir", default="outputs/birnn")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--embed_dim", type=int, default=128)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--bidirectional", type=bool, default=True)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()
    train(args)
