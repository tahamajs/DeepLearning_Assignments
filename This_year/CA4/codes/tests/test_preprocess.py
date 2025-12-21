import pytest
import torch
from src.data.preprocess import WhitespaceTokenizer, build_vocab, build_label_vocab, ATISDataset, collate_fn


def test_tokenizer():
    t = WhitespaceTokenizer()
    assert t.tokenize("Show me Flights") == ["show", "me", "flights"]


def test_vocab_and_dataset():
    examples = [{"tokens": ["show", "me", "flights"], "slots": ["O", "O", "O"], "intent": "atis_flight"}]
    w2i, i2w = build_vocab([ex["tokens"] for ex in examples])
    s2i, i2s = build_label_vocab([s for ex in examples for s in ex["slots"]])
    intents = {"atis_flight": 0}
    ds = ATISDataset(examples, w2i, s2i, intents)
    sample = ds[0]
    assert "token_ids" in sample and "slot_ids" in sample and "intent" in sample


def test_collate():
    examples = [{"tokens": ["show", "me"], "slots": ["O", "O"], "intent": "a"}, {"tokens": ["hello"], "slots": ["O"], "intent": "a"}]
    w2i, _ = build_vocab([ex["tokens"] for ex in examples])
    s2i, _ = build_label_vocab([s for ex in examples for s in ex["slots"]])
    intents = {"a": 0}
    ds = ATISDataset(examples, w2i, s2i, intents)
    batch = [ds[0], ds[1]]
    coll = collate_fn(batch)
    assert coll["input_ids"].shape[0] == 2
    assert coll["slot_ids"].shape[0] == 2
    assert coll["intent"].shape[0] == 2
