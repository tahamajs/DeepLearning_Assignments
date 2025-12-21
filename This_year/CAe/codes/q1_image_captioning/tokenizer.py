"""
Tokenizer utilities for Image Captioning (build vocab, tokenization, detokenization)
"""
from collections import Counter
import re

class Tokenizer:
    def __init__(self, specials=['<pad>', '<start>', '<end>', '<unk>']):
        self.word2idx = {}
        self.idx2word = {}
        self.specials = specials
        for i, tok in enumerate(specials):
            self.word2idx[tok] = i
            self.idx2word[i] = tok
        self.next_idx = len(specials)

    def build_vocab(self, captions, min_freq=1):
        """Build vocab from list of captions (strings)."""
        cnt = Counter()
        for s in captions:
            toks = self._tokenize_text(s)
            cnt.update(toks)
        for w, c in cnt.items():
            if c >= min_freq and w not in self.word2idx:
                self.word2idx[w] = self.next_idx
                self.idx2word[self.next_idx] = w
                self.next_idx += 1

    def text_to_sequence(self, text):
        toks = self._tokenize_text(text)
        seq = [self.word2idx.get('<start>')]
        for t in toks:
            seq.append(self.word2idx.get(t, self.word2idx['<unk>']))
        seq.append(self.word2idx.get('<end>'))
        return seq

    def sequence_to_text(self, seq):
        words = [self.idx2word.get(i, '<unk>') for i in seq]
        # trim special tokens
        words = [w for w in words if w not in ['<start>', '<end>', '<pad>']]
        return ' '.join(words)

    def _tokenize_text(self, s):
        # Basic cleaning: remove punctuation except emojis and split on whitespace
        s = s.lower().strip()
        s = re.sub(r"[\"#$%&'()*+,-./:;<=>?@\[\\\]^_`{|}~]", '', s)
        toks = s.split()
        return toks

    def vocab_size(self):
        return len(self.word2idx)
