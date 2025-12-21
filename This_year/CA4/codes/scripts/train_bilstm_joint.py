"""Train BiLSTM joint model (intent + slot heads).
Usage:
    python scripts/train_bilstm_joint.py --data_dir data/raw --out_dir outputs/bilstm_joint
"""
import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from tqdm import tqdm

from src.data.preprocess import load_atis_examples, build_vocab, build_label_vocab, ATISDataset, collate_fn
from src.models.baseline import BiLSTMJoint
from src.utils.metrics import slot_f1, slot_classification_report, intent_accuracy


def train(args):
    train_examples = load_atis_examples(Path(args.data_dir) / "train.json")
    test_examples = load_atis_examples(Path(args.data_dir) / "test.json")

    word2id, id2word = build_vocab([ex["tokens"] for ex in train_examples])
    slot2id, id2slot = build_label_vocab([s for ex in train_examples for s in ex["slots"]])
    intents = sorted({ex["intent"] for ex in train_examples})
    intent2id = {intent: idx for idx, intent in enumerate(intents)}

    train_ds = ATISDataset(train_examples, word2id, slot2id, intent2id)
    test_ds = ATISDataset(test_examples, word2id, slot2id, intent2id)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BiLSTMJoint(len(word2id), embed_dim=args.embed_dim, hidden_dim=args.hidden_dim, num_slot_labels=len(slot2id), num_intents=len(intent2id), dropout=args.dropout)
    model.to(device)

    slot_criterion = nn.CrossEntropyLoss(ignore_index=0)
    intent_criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_f1 = 0.0
    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}"):
            input_ids = batch["input_ids"].to(device)
            slot_ids = batch["slot_ids"].to(device)
            intents = batch["intent"].to(device)
            optimizer.zero_grad()
            slot_logits, intent_logits = model(input_ids)
            b, t, c = slot_logits.shape
            slot_loss = slot_criterion(slot_logits.view(-1, c), slot_ids.view(-1))
            intent_loss = intent_criterion(intent_logits, intents)
            loss = slot_loss + intent_loss
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f"Epoch {epoch+1} train loss: {running_loss/len(train_loader):.4f}")

        # eval
        model.eval()
        preds, trues = [], []
        intent_trues, intent_preds = [], []
        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch["input_ids"].to(device)
                slot_ids = batch["slot_ids"].to(device)
                intents = batch["intent"].to(device)
                slot_logits, intent_logits = model(input_ids)
                pred = slot_logits.argmax(dim=-1).cpu().tolist()
                true = slot_ids.cpu().tolist()
                preds.extend(pred)
                trues.extend(true)
                intent_preds.extend(intent_logits.argmax(dim=-1).cpu().tolist())
                intent_trues.extend(intents.cpu().tolist())

        id2slot = {v: k for k, v in slot2id.items()}
        pred_labels = [[id2slot.get(pid, "O") for pid in seq] for seq in preds]
        true_labels = [[id2slot.get(tid, "O") for tid in seq] for seq in trues]
        f1 = slot_f1(true_labels, pred_labels)
        acc = intent_accuracy(intent_trues, intent_preds)
        print(f"Epoch {epoch+1} test slot F1: {f1:.4f}  intent acc: {acc:.4f}")

        if f1 > best_f1:
            best_f1 = f1
            out_dir = Path(args.out_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), out_dir / "best_bilstm_joint.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data/raw")
    parser.add_argument("--out_dir", default="outputs/bilstm_joint")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=24)
    parser.add_argument("--embed_dim", type=int, default=128)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()
    train(args)
