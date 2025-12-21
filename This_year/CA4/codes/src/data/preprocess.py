"""Data loading and preprocessing utilities for ATIS dataset.

Provides:
- Tokenizer (whitespace)
- Vocab builders for words, slots, intents
- ATISDataset (torch.utils.data.Dataset)
- collate_fn for dynamic batch padding

Usage example:
>>> from src.data.preprocess import load_atis_dataset, build_vocab, ATISDataset, collate_fn
>>> train_examples = load_atis_dataset('data/raw/train.json')
>>> word2id, id2word = build_vocab([ex['tokens'] for ex in train_examples])
"""
from collections import Counter
from pathlib import Path
from typing import List, Tuple, Dict, Iterable, Optional

import json
import csv

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset


# Special tokens
PAD = "<PAD>"
UNK = "<UNK>"
BOS = "<BOS>"
EOS = "<EOS>"


class WhitespaceTokenizer:
    def __init__(self, lowercase: bool = True):
        self.lowercase = lowercase

    def tokenize(self, text: str) -> List[str]:
        if self.lowercase:
            text = text.lower()
        return text.strip().split()


def read_json_or_csv(path: Path) -> List[Dict]:
    path = Path(path)
    if path.is_dir():
        # guess train/test files inside dir
        files = list(path.glob("**/*.json")) + list(path.glob("**/*.csv"))
        if not files:
            raise FileNotFoundError(f"No json/csv files found in {path}")
        data = []
        for f in files:
            data.extend(read_json_or_csv(f))
        return data

    if path.suffix == ".json":
        with open(path, "r", encoding="utf8") as f:
            j = json.load(f)
            # Accept list of examples or dict with splits
            if isinstance(j, list):
                return j
            # If dictionary with 'train' 'test'
            examples = []
            for k, v in j.items():
                if isinstance(v, list):
                    examples.extend(v)
            return examples

    elif path.suffix == ".csv":
        data = []
        with open(path, "r", encoding="utf8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                data.append(row)
        return data

    else:
        raise ValueError("Unsupported file type: " + str(path))


def load_atis_examples(path: str) -> List[Dict]:
    """Load ATIS examples returning standardized structure:
    [{'tokens': [...], 'slots': [...], 'intent': '...'}, ...]

    The function can adapt to a few common dataset file formats.
    """
    path = Path(path)
    raw = read_json_or_csv(path)
    examples = []
    for ex in raw:
        # Various formats: either tokens/slot_labels/intent, or text/slot_labels
        if "tokens" in ex and "slot_labels" in ex and "intent" in ex:
            tokens = ex["tokens"]
            slots = ex["slot_labels"]
            intent = ex["intent"]
        elif "text" in ex and "slots" in ex and "intent" in ex:
            # slots could be list of labels aligned to tokens or a dict
            tokens = WhitespaceTokenizer().tokenize(ex["text"])
            slots = ex["slots"]
            intent = ex["intent"]
        elif "text" in ex and "slot_labels" in ex and "intent" in ex:
            tokens = WhitespaceTokenizer().tokenize(ex["text"])
            slots = ex["slot_labels"]
            intent = ex["intent"]
        else:
            # Try to infer from CSV with fields
            if "text" in ex:
                tokens = WhitespaceTokenizer().tokenize(ex["text"])
            elif "utterance" in ex:
                tokens = WhitespaceTokenizer().tokenize(ex["utterance"])
            else:
                # skip malformed
                continue
            slots = ex.get("slot_labels") or ex.get("slots") or ex.get("labels")
            intent = ex.get("intent") or ex.get("intent_label") or ex.get("label")
            if isinstance(slots, str):
                # assume space-separated labels aligned to tokens
                slots = slots.strip().split()
            if isinstance(intent, dict):
                # sometimes intent is a mapping like {'intent': 'atis_flight'}
                # try to extract a string value
                if "intent" in intent:
                    intent = intent["intent"]
                else:
                    # take the first value
                    vals = list(intent.values())
                    intent = vals[0] if vals else "unknown"

        # ensure tokens/slots align
        if slots is None:
            # default to all O labels
            slots = ["O"] * len(tokens)
        # convert any strings of slots to list
        if isinstance(slots, str):
            slots = slots.strip().split()
        # final check lengths
        if len(slots) != len(tokens):
            # try to align using tokenized text if possible
            # fallback: pad/truncate slots
            if len(slots) < len(tokens):
                slots = slots + ["O"] * (len(tokens) - len(slots))
            else:
                slots = slots[: len(tokens)]

        examples.append({"tokens": tokens, "slots": slots, "intent": intent})

    return examples


def build_vocab(seqs: Iterable[Iterable[str]], min_freq: int = 1, specials: Optional[List[str]] = None) -> Tuple[Dict[str, int], Dict[int, str]]:
    if specials is None:
        specials = [PAD, UNK, BOS, EOS]
    counter = Counter()
    for seq in seqs:
        counter.update(seq)
    # filter by frequency
    items = [tok for tok, cnt in counter.items() if cnt >= min_freq]
    # build mappings with specials first
    itoks = list(specials) + sorted(items)
    token2id = {tok: idx for idx, tok in enumerate(itoks)}
    id2token = {idx: tok for tok, idx in token2id.items()}
    return token2id, id2token


def build_label_vocab(labels: Iterable[str], specials: Optional[List[str]] = None) -> Tuple[Dict[str, int], Dict[int, str]]:
    if specials is None:
        specials = [PAD]
    uniq = sorted(set(labels))
    itoks = list(specials) + uniq
    label2id = {lab: idx for idx, lab in enumerate(itoks)}
    id2label = {idx: lab for lab, idx in label2id.items()}
    return label2id, id2label


def encode_example(example: Dict, word2id: Dict[str, int], slot2id: Dict[str, int], intent2id: Dict[str, int], add_bos_eos: bool = False) -> Dict:
    tokens = example["tokens"]
    slots = example["slots"]
    intent = example["intent"]
    if add_bos_eos:
        tokens = [BOS] + tokens + [EOS]
        slots = ["O"] + slots + ["O"]
    token_ids = [word2id.get(t, word2id.get(UNK)) for t in tokens]
    slot_ids = [slot2id.get(s, slot2id.get("O", 0)) for s in slots]
    intent_id = intent2id.get(intent, intent2id.get("unknown", 0))
    return {"token_ids": token_ids, "slot_ids": slot_ids, "intent_id": intent_id, "tokens": tokens}


class ATISDataset(Dataset):
    def __init__(self, examples: List[Dict], word2id: Dict[str, int], slot2id: Dict[str, int], intent2id: Dict[str, int], add_bos_eos: bool = False):
        self.examples = [encode_example(ex, word2id, slot2id, intent2id, add_bos_eos) for ex in examples]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        return {"token_ids": torch.tensor(ex["token_ids"], dtype=torch.long), "slot_ids": torch.tensor(ex["slot_ids"], dtype=torch.long), "intent": torch.tensor(ex["intent_id"], dtype=torch.long), "tokens": ex["tokens"]}


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """Pad sequences in a batch to the max length in the batch.

    Returns tensors: input_ids (B, T), slot_ids (B, T), intent (B), lengths (B)
    """
    token_seqs = [item["token_ids"] for item in batch]
    slot_seqs = [item["slot_ids"] for item in batch]
    intents = torch.stack([item["intent"] for item in batch])
    lengths = torch.tensor([len(s) for s in token_seqs], dtype=torch.long)
    # pad
    pad_val = 0  # assume PAD token id is 0
    input_ids = pad_sequence(token_seqs, batch_first=True, padding_value=pad_val)
    slot_pad_val = 0
    slot_ids = pad_sequence(slot_seqs, batch_first=True, padding_value=slot_pad_val)
    return {"input_ids": input_ids, "slot_ids": slot_ids, "intent": intents, "lengths": lengths}


__all__ = [
    "WhitespaceTokenizer",
    "load_atis_examples",
    "build_vocab",
    "build_label_vocab",
    "ATISDataset",
    "collate_fn",
]


def save_vocab(out_dir: Path, word2id, id2word, slot2id, id2slot, intent2id, id2intent):
    out_dir.mkdir(parents=True, exist_ok=True)
    vocab = {
        "word2id": word2id,
        "id2word": id2word,
        "slot2id": slot2id,
        "id2slot": id2slot,
        "intent2id": intent2id,
        "id2intent": id2intent,
    }
    with open(out_dir / "vocab.json", "w", encoding="utf8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)


def save_examples(out_dir: Path, name: str, examples: List[Dict]):
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / f"{name}.json", "w", encoding="utf8") as f:
        json.dump(examples, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    import argparse
    import random

    parser = argparse.ArgumentParser(description="Preprocess ATIS dataset: tokenization, build vocab, split and save processed files")
    parser.add_argument("--input", type=str, default="data/raw", help="Path to raw dataset (file or dir)")
    parser.add_argument("--out", type=str, default="data/processed", help="Output directory for processed files")
    parser.add_argument("--val_frac", type=float, default=0.1)
    parser.add_argument("--test_frac", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    raw_path = Path(args.input)
    out_dir = Path(args.out)

    # Try to read explicit train/dev/test files if present
    if (raw_path / "train.json").exists() and (raw_path / "test.json").exists():
        train_examples = load_atis_examples(raw_path / "train.json")
        test_examples = load_atis_examples(raw_path / "test.json")
        dev_examples = load_atis_examples(raw_path / "dev.json") if (raw_path / "dev.json").exists() else []
    else:
        all_examples = load_atis_examples(raw_path)
        random.seed(args.seed)
        random.shuffle(all_examples)
        n = len(all_examples)
        n_test = int(n * args.test_frac)
        n_val = int(n * args.val_frac)
        test_examples = all_examples[:n_test]
        dev_examples = all_examples[n_test : n_test + n_val]
        train_examples = all_examples[n_test + n_val :]

    # Build vocabs from train
    word2id, id2word = build_vocab([ex["tokens"] for ex in train_examples])
    slot2id, id2slot = build_label_vocab([s for ex in train_examples for s in ex["slots"]])
    intents = sorted({ex["intent"] for ex in train_examples})
    intent2id = {intent: idx for idx, intent in enumerate(intents)}
    id2intent = {idx: intent for intent, idx in intent2id.items()}

    # Save processed
    save_examples(out_dir, "train", train_examples)
    save_examples(out_dir, "dev", dev_examples)
    save_examples(out_dir, "test", test_examples)
    save_vocab(out_dir, word2id, id2word, slot2id, id2slot, intent2id, id2intent)

    print(f"Wrote processed datasets to {out_dir} (train={len(train_examples)} dev={len(dev_examples)} test={len(test_examples)})")
